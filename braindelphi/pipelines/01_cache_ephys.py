# Standard library
import logging
import pickle
from datetime import datetime as dt
from pathlib import Path
from braindelphi.pipelines.utils_common_pipelines import load_ephys
from braindelphi.pipelines.utils_common_pipelines import cache_regressors

# Third party libraries
import dask
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dask.distributed import LocalCluster

# IBL libraries
from one.api import ONE
from brainwidemap import bwm_query
from braindelphi.params import CACHE_PATH
CACHE_PATH.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger('braindelphi')

@dask.delayed
def delayed_load(eid, pids, params):
    try:
        return load_ephys(eid, pids, **params)
    except KeyError:
        pass


@dask.delayed(pure=False, traverse=False)
def delayed_save(subject, eid, probe_name, params, outputs):
    return cache_regressors(subject, eid, probe_name, params, outputs)


# Parameters
ALGN_RESOLVED = True
DATE = str(dt.today())
MAX_LEN = None
T_BEF = 0.6
T_AFT = 0.6
BINWIDTH = 0.02
ABSWHEEL = False
WHEEL = False
QC = True
TYPE = 'primaries'
MERGE_PROBES = False
# End parameters

# Construct params dict from above
params = {
    'max_len': MAX_LEN,
    't_before': T_BEF,
    't_after': T_AFT,
    'binwidth': BINWIDTH,
    'abswheel': ABSWHEEL,
    'ret_qc': QC,
    'wheel': WHEEL,
}

dataset_futures = []

one = ONE()
bwm_df = bwm_query(one, alignment_resolved=ALGN_RESOLVED).set_index(['subject', 'eid'])

for eid in bwm_df.index.unique(level='eid'):
    session_df = bwm_df.xs(eid, level='eid')
    subject = session_df.index[0]
    # If there are two probes, there are two options:
    # load and save data from each probe independently, or merge the data from both probes
    pids = session_df.pid.to_list()
    probe_names = session_df.probe_name.to_list()
    if MERGE_PROBES:
        load_outputs = delayed_load(eid, pids, params)
        save_future = delayed_save(subject, eid, 'merged_probes', {**params, 'type': TYPE}, load_outputs)
        dataset_futures.append([subject, eid, 'merged_probes', save_future])
    else:
        for (pid, probe_name) in zip(pids, probe_names):
            load_outputs = delayed_load(eid, [pid], params)
            save_future = delayed_save(subject, eid, probe_name, {**params, 'type': TYPE}, load_outputs)
            dataset_futures.append([subject, eid, probe_name, save_future])

N_CORES = 4

cluster = SLURMCluster(cores=N_CORES,
                       memory='32GB',
                       processes=1,
                       queue="shared-cpu",
                       walltime="01:15:00",
                       log_directory='/srv/beegfs/scratch/users/f/findling/dask-worker-logs',
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

outdict = {'params': params, 'dataset_filenames': dataset}
with open(Path(CACHE_PATH).joinpath(DATE + '_ephys_metadata.pkl'), 'wb') as fw:
    pickle.dump(outdict, fw)

"""
import numpy as np
failures = [(i, x) for i, x in enumerate(tmp_futures) if x.status == 'error']
for i, failure in failures:
    print(i, failure.exception(), failure.key)
print(len(failures))
print(np.array(failures)[:,1])
import traceback
tb = failure.traceback()
traceback.print_tb(tb)
print(len([(i, x) for i, x in enumerate(tmp_futures) if x.status == 'cancelled']))
print(len([(i, x) for i, x in enumerate(tmp_futures) if x.status == 'error']))
print(len([(i, x) for i, x in enumerate(tmp_futures) if x.status == 'lost']))
print(len([(i, x) for i, x in enumerate(tmp_futures) if x.status == 'pending']))

import numpy as np
nb_trials_per_df = np.zeros(dataset.index.size)
for i_filepath, filepath in enumerate(dataset.reg_file):
    nb_trials_per_df[i_filepath] = pickle.load(open(filepath, 'rb'))['trials_df'].index.size
"""