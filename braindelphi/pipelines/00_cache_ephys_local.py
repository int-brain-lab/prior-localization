# Standard library
import logging
import pickle
from datetime import datetime as dt
from pathlib import Path

# Third party libraries
import pandas as pd

# IBL libraries
from braindelphi.params import CACHE_PATH, SETTINGS_PATH
from braindelphi.pipelines.utils_common_pipelines import load_ephys
from braindelphi.pipelines.utils_common_pipelines import cache_regressors
from braindelphi.decoding.functions.utils import check_settings

# braindelphi repo imports
from braindelphi.utils_root import query_sessions

_logger = logging.getLogger('braindelphi')


def delayed_load(eid, pids, params):
    try:
        return load_ephys(eid, pids, **params)
    except KeyError:
        pass


def delayed_save(subject, eid, probes, params, outputs):
    return cache_regressors(subject, eid, probes, params, outputs)


# Parameters
SESS_CRITERION = 'aligned-behavior'
DATE = '2022-06-02'  # str(dt.today())
MAX_LEN = 5.
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
    'wheel': WHEEL,
    'ret_qc': QC,
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
        save_future = delayed_save(
            subject, eid, probes,
            {**params, 'type': TYPE, 'merge_probes': MERGE_PROBES},
            load_outputs)
        dataset_futures.append([subject, eid, probes, save_future])


# Run below code AFTER futures have finished!
dataset = [{
    'subject': x[0],
    'eid': x[1],
    'probes': x[2],
    'meta_file': x[3][0],
    'reg_file': x[3][1]
} for i, x in enumerate(dataset_futures)]
dataset = pd.DataFrame(dataset)

outdict = {'params': params, 'dataset_filenames': dataset}
with open(Path(CACHE_PATH).joinpath(DATE + '_ephys_metadata.pkl'), 'wb') as fw:
    pickle.dump(outdict, fw)
