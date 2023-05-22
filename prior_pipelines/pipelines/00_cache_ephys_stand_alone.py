# Standard library
import logging
import pickle
from datetime import datetime as dt
from pathlib import Path
from urllib.error import HTTPError
import yaml

# Third party libraries
import pandas as pd

# IBL libraries
from brainwidemap import bwm_query
from one.api import ONE

# braindelphi repo imports
from braindelphi.params import CACHE_PATH, SETTINGS_PATH
from braindelphi.pipelines.utils_common_pipelines import load_ephys
from braindelphi.pipelines.utils_common_pipelines import cache_regressors
from braindelphi.decoding.functions.utils import check_settings

_logger = logging.getLogger('braindelphi')


def delayed_load(eid, pids, params):
    try:
        return load_ephys(eid, pids, **params)
    except KeyError:
        pass


def delayed_save(subject, eid, probes, params, outputs):
    return cache_regressors(subject, eid, probes, params, outputs)


# load settings as a dict
settings = yaml.safe_load(open(SETTINGS_PATH))
kwargs = check_settings(settings)

# Parameters
SESS_CRITERION = kwargs['criterion']
DATE = kwargs['date']
MAX_LEN = 5.
T_BEF = 0.6
T_AFT = 0.6
BINWIDTH = kwargs['binsize']
ABSWHEEL = False
WHEEL = False
QC = True
TYPE = 'primaries'
MERGE_PROBES = kwargs['merge_probes']
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

one = ONE()
alignment_resolved = True if SESS_CRITERION.find('algined') > -1 else False
bwm_df = bwm_query(one, alignment_resolved=alignment_resolved).set_index(['subject', 'eid'])

for i, eid in enumerate(bwm_df.index.unique(level='eid')):
    if i <= 142:
        continue
    session_df = bwm_df.xs(eid, level='eid')
    subject = session_df.index[0]
    print(eid)
    print(subject)
    pids = session_df.pid.to_list()
    probe_names = session_df.probe_name.to_list()
    if MERGE_PROBES:
        try:
            load_outputs = delayed_load(eid, pids, params)
            save_future = delayed_save(
                subject, eid, 'merged_probes', {**params, 'type': TYPE}, load_outputs)
            dataset_futures.append([subject, eid, 'merged_probes', save_future])
        except HTTPError as e:
            print('Caught HTTPError for eid: %s' % eid)
            print(e)
        except AttributeError as e:
            print('Caught AttributeError for eid: %s' % eid)
            print(e)
        except IndexError as e:
            print('Caught IndexError for eid: %s' % eid)
            print(e)
    else:
        for (pid, probe_name) in zip(pids, probe_names):
            try:
                load_outputs = delayed_load(eid, [pid], params)
                save_future = delayed_save(
                    subject, eid, probe_name, {**params, 'type': TYPE}, load_outputs)
                dataset_futures.append([subject, eid, probe_name, save_future])
            except HTTPError as e:
                print('Caught HTTPError for eid: %s' % eid)
                print(e)
            except AttributeError as e:
                print('Caught AttributeError for eid: %s' % eid)
                print(e)
            except IndexError as e:
                print('Caught IndexError for eid: %s' % eid)
                print(e)


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
