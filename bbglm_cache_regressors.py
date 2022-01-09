import dask
import re
import pickle
import pandas as pd
from params import GLM_CACHE
from pathlib import Path
from datetime import datetime as dt
from one.api import ONE
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from bbglm_sessfit import load_regressors, cache_regressors
from .decoding.decoding_utils import query_sessions


@dask.delayed
def delayed_load(session_id, probes, params, force_load=False):
    try:
        return load_regressors(session_id, probes, **params)
    except KeyError as e:
        if force_load:
            params['resolved_alignment'] = False
            return load_regressors(session_id, probes, **params)
        else:
            raise e

@dask.delayed(pure=False, traverse=False)
def delayed_save(subject, session_id, probes, params, outputs):
    return cache_regressors(subject, session_id, probes, params, *outputs)


# Parameters
SESS_CRITERION = 'resolved-behavior'
DATE = str(dt.today())
MAX_LEN = 2.
T_BEF = 0.6
T_AFT = 0.6
BINWIDTH = 0.02
ABSWHEEL = False
QC = True
FORCE = True  # If load_spike_sorting_fast doesn't return _channels, use _channels function
# End parameters

#Construct params dict from above
params = {'max_len': MAX_LEN,
          't_before': T_BEF,
          't_after': T_AFT,
          'binwidth': BINWIDTH,
          'abswheel': ABSWHEEL,
          'resolved_alignment': True if re.match('resolved.*', SESS_CRITERION) else False,
          'ret_qc': QC}

one = ONE()
dataset_futures = []

sessdf = query_sessions(SESS_CRITERION).set_index(['subject', 'eid'])

for eid in sessdf.index.unique(level='eid'):
    xsdf = sessdf.xs(eid, level='eid')
    subject = xsdf.index[0]
    probes = xsdf.probe.to_list()
    load_outputs = delayed_load(eid, probes, params, force_load=FORCE)
    save_future = delayed_save(subject, eid, probes, params, load_outputs)
    dataset_futures.append([subject, eid, probes, save_future])

N_CORES = 4
cluster = SLURMCluster(cores=N_CORES, memory='32GB', processes=1, queue="shared-cpu",
                       walltime="01:15:00",
                       log_directory='/home/gercek/dask-worker-logs',
                       interface='ib0',
                       extra=["--lifetime", "60m", "--lifetime-stagger", "10m"],
                       job_cpu=N_CORES, env_extra=[f'export OMP_NUM_THREADS={N_CORES}',
                                                   f'export MKL_NUM_THREADS={N_CORES}',
                                                   f'export OPENBLAS_NUM_THREADS={N_CORES}'])
cluster.adapt(minimum_jobs=0, maximum_jobs=20)
client = Client(cluster)

tmp_futures = [client.compute(future[3]) for future in dataset_futures]
dataset = [{'subject': x[0], 'eid': x[1], 'probes': x[2], 'meta_file': tmp_futures[i].result()[0],
            'reg_file': tmp_futures[i].result()[1]}
           for i, x in enumerate(dataset_futures) if tmp_futures[i].status == 'finished']
dataset = pd.DataFrame(dataset)

outdict = {'params': params, 'dataset_filenames': dataset}
with open(Path(GLM_CACHE).joinpath(DATE + '_dataset_metadata.pkl'), 'wb') as fw:
    pickle.dump(outdict, fw)
