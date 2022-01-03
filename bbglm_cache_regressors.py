import dask
import re
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
    return cache_regressors(subject, session_id, probes, params, **outputs)


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
          'abshweel': ABSWHEEL,
          'resolved_alignment': True if re.match('resolved.*', SESS_CRITERION) else False,
          'ret_qc': QC}

one = ONE()
dataset_futures = []

sessdf = query_sessions(SESS_CRITERION).set_index(['subject', 'eid'])

for eid in sessdf.unique('eid', level='eid'):
    xsdf = sessdf.xs(eid, level='eid')
    subject = xsdf.index[0]
    probes = xsdf.probes.to_list()
    load_outputs = delayed_load(eid, probes, params, force_load=FORCE)
    save_future = delayed_save(subject, eid, probes, params, load_outputs)
    dataset_futures.append([subject, eid, probes, save_future])
    
