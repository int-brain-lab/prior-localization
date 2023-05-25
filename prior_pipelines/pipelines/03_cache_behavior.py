# Standard library
import logging
import pickle
from datetime import datetime as dt
from pathlib import Path

# Third party libraries
import dask
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dask.distributed import LocalCluster

# IBL libraries
from one.api import ONE
from brainwidemap import bwm_query
from prior_pipelines.params import CACHE_PATH, SETTINGS_PATH
CACHE_PATH.mkdir(parents=True, exist_ok=True)
from prior_pipelines.pipelines.utils_common_pipelines import load_behavior
from prior_pipelines.pipelines.utils_common_pipelines import cache_behavior
from prior_pipelines.decoding.functions.utils import check_settings

_logger = logging.getLogger('prior_pipelines')


@dask.delayed
def delayed_load(eid, target):
    try:
        return load_behavior(eid, target)
    except KeyError:
        pass


@dask.delayed(pure=False, traverse=False)
def delayed_save(subject, eid, target, outputs):
    return cache_behavior(subject, eid, target, outputs)


# load settings as a dict
settings = yaml.safe_load(open(SETTINGS_PATH))
kwargs = check_settings(settings)

dataset_futures = []

one = ONE()
one.alyx.clear_rest_cache()
alignment_resolved = True if kwargs['criterion'] == 'aligned-behavior' else False
bwm_df = bwm_query(one, alignment_resolved=alignment_resolved).set_index(['subject', 'eid'])

for eid in bwm_df.index.unique(level='eid'):
    session_df = bwm_df.xs(eid, level='eid')
    subject = session_df.index[0]
    load_outputs = delayed_load(eid, kwargs['target'])
    if not load_outputs['skip']:
        save_future = delayed_save(subject, eid, kwargs['target'], load_outputs)
        dataset_futures.append([subject, eid, save_future])

N_CORES = 4

cluster = SLURMCluster(cores=N_CORES,
                       memory='32GB',
                       processes=1,
                       queue="shared-cpu",
                       walltime="01:15:00",
                       log_directory='~/dask-worker-logs',
                       interface='ib0',
                       extra=["--lifetime", "60m", "--lifetime-stagger", "10m"],
                       job_cpu=N_CORES,
                       env_extra=[
                           f'export OMP_NUM_THREADS={N_CORES}',
                           f'export MKL_NUM_THREADS={N_CORES}',
                           f'export OPENBLAS_NUM_THREADS={N_CORES}'
                       ])

# cluster = LocalCluster()
cluster.scale(20)

client = Client(cluster)

tmp_futures = [client.compute(future[3]) for future in dataset_futures]

# Run below code AFTER futures have finished!
dataset = [{
    'subject': x[0],
    'eid': x[1],
    'probe_name': x[2],
    'meta_file': tmp_futures[i].result()[0],
    'reg_file': tmp_futures[i].result()[1]
} for i, x in enumerate(dataset_futures) if tmp_futures[i].status == 'finished']
dataset = pd.DataFrame(dataset)

outdict = {'params': kwargs, 'dataset_filenames': dataset}
outfile = Path(CACHE_PATH).joinpath(kwargs['date'] + '_%s_metadata.pkl' % kwargs['target'])
with open(outfile, 'wb') as fw:
    pickle.dump(outdict, fw)
