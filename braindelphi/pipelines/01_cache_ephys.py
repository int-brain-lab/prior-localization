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
from braindelphi.params import CACHE_PATH

# braindelphi repo imports
from braindelphi.utils_root import query_sessions

_logger = logging.getLogger('braindelphi')

@dask.delayed
def delayed_load(eid, pids, params):
    try:
        return load_ephys(eid, pids, **params)
    except KeyError:
        pass


@dask.delayed(pure=False, traverse=False)
def delayed_save(subject, eid, probes, params, outputs):
    return cache_regressors(subject, eid, probes, params, outputs)


# Parameters
SESS_CRITERION = 'resolved-behavior'
DATE = str(dt.today())
MAX_LEN = 2.
T_BEF = 0.6
T_AFT = 0.6
BINWIDTH = 0.02
ABSWHEEL = True
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
    'ret_qc': QC
}

dataset_futures = []

sessdf = query_sessions(SESS_CRITERION).set_index(['subject', 'eid'])

for eid in sessdf.index.unique(level='eid'):
    xsdf = sessdf.xs(eid, level='eid')
    subject = xsdf.index[0]
    pids_lst = [[pid] for pid in xsdf.pid.to_list()] if not MERGE_PROBES else [xsdf.pid.to_list()]
    probe_lst = [[n] for n in xsdf.probe.to_list()] if not MERGE_PROBES else [xsdf.probe.to_list()]
    for (probes, pids) in zip(probe_lst, pids_lst):
        load_outputs = delayed_load(eid, pids, params)
        save_future = delayed_save(subject, eid, probes, {**params, 'type': TYPE, 'merge_probes':MERGE_PROBES},
                                   load_outputs)
        dataset_futures.append([subject, eid, probes, save_future])

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
    'probes': x[2],
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
print(np.array(failures)[:,0])
import traceback
tb = failure.traceback()
traceback.print_tb(tb)
print(len([(i, x) for i, x in enumerate(tmp_futures) if x.status == 'cancelled']))
print(len([(i, x) for i, x in enumerate(tmp_futures) if x.status == 'error']))
print(len([(i, x) for i, x in enumerate(tmp_futures) if x.status == 'lost']))
print(len([(i, x) for i, x in enumerate(tmp_futures) if x.status == 'finished']))
"""