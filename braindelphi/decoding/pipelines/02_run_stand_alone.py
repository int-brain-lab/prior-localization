"""
On the second part of the pipeline example, we loop over the dataframe
The analysis tools load the downloaded and cached version of the data.
"""
import copy
import glob
import logging
import numpy as np
from pathlib import Path
import pandas as pd
import sys
import yaml

from one.api import ONE

from braindelphi.utils_root import load_pickle_data
from braindelphi.params import braindelphi_PATH, CACHE_PATH, SETTINGS_PATH, FIT_PATH
from braindelphi.pipelines.utils_common_pipelines import load_ephys, load_behavior
from braindelphi.decoding.functions.decoding import fit_eid
from braindelphi.decoding.functions.utils import check_settings


# True if braindelphi/
LOAD_FROM_CACHE = True

# sessions to be excluded (by Olivier Winter)
excludes = [
    'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',  # inconsistent trials object: relaunched on 31-12-2021
    'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',  # same same
    '7a887357-850a-4378-bd2a-b5bc8bdd3aac',  # same same
    '56b57c38-2699-4091-90a8-aba35103155e',  # load obect pickle error
    '09394481-8dd2-4d5c-9327-f2753ede92d7',  # same same
]


# load settings as a dict
settings = yaml.safe_load(open(SETTINGS_PATH))
kwargs = check_settings(settings)
# add path info
kwargs['add_to_saving_path'] = '_binsize=%i_lags=%i_mergedProbes_%i' % (
    1000 * kwargs['binsize'], kwargs['n_bins_lag'], kwargs['merge_probes'],
)
kwargs['neuralfit_path'] = FIT_PATH
print(kwargs)

# set up logging
log_file = FIT_PATH.joinpath('decoding_%s_%s.log' % (kwargs['target'], kwargs['date']))
logging.basicConfig(
    filename=log_file, filemode='w', level=logging.DEBUG,
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # add logging to console

# load insertion data
if LOAD_FROM_CACHE:
    metadata_file = list(Path(CACHE_PATH).glob('*%s_metadata.pkl' % kwargs['neural_dtype']))[0]
    cache_data = load_pickle_data(metadata_file)
    bwm_df = cache_data['dataset_filenames']
    eids = bwm_df['eid'].unique()
else:
    insdf = pd.read_parquet(braindelphi_PATH.joinpath('decoding', 'insertions.pqt'))
    insdf = insdf[insdf.spike_sorting != ""]
    eids = insdf['eid'].unique()

# load imposter session df
if kwargs.get('n_pseudo', 0) > 0:
    ephys_str = '_beforeRecording' if not kwargs['imposter_generate_from_ephys'] else ''
    filename = 'imposterSessions_%s%s.pqt' % (kwargs['target'], ephys_str)
    imposter_path = braindelphi_PATH.joinpath('decoding', filename)
    imposter_df = pd.read_parquet(imposter_path)
    kwargs['imposter_df'] = imposter_df

# loop over sessions: load data and decode
one = ONE()
IMIN = 0
IMAX = 10
for i, eid in enumerate(eids):

    # determine if we should proceed with decoding session
    if i < IMIN:
        continue
    if i >= IMAX:
        continue
    if eid in excludes:
        continue

    if LOAD_FROM_CACHE:
        # use pre-computed pkl files
        curr_df = bwm_df[bwm_df['eid'] == eid]
        subject = curr_df.subject.iloc[0]

        # format probe/pid info depending on whether or not we want to merge units across probes
        filename = curr_df.meta_file.iloc[0]
        meta = load_pickle_data(filename)
        pids_lst = [[None]]
        probe_lst = [meta['probes']]

        # load behavioral data (same for all probes)
        filename = '%s_%s_regressors.pkl' % (kwargs['date'], kwargs['target'])
        beh_cache = Path(CACHE_PATH).joinpath('behavior', subject, eid, filename)
        # need to check if this exists since we're iterating through the neural data cache
        if beh_cache.exists():
            dlc_dict = load_pickle_data(beh_cache)
        else:
            logging.log(
                logging.DEBUG,
                f"{i}, session: {eid}, subject: {subject} - NO {kwargs['target']} DATA")
            continue

    else:
        # load with one using a pre-computed sessions table
        curr_df = insdf[insdf['eid'] == eid]
        subject = curr_df.subject.iloc[0]

        if np.any(curr_df['spike_sorting'] == ""):
            logging.log(
                logging.DEBUG, f"{i}, session: {eid}, subject: {subject} - NO SPIKE SORTING")
            continue

        # format probe/pid info depending on whether or not we want to merge units across probes
        pids_lst = [
            [pid] for pid in curr_df.pid.to_list()] if not kwargs['merge_probes'] \
            else [curr_df.pid.to_list()]
        probe_lst = [
            [n] for n in curr_df.probe.to_list()] if not kwargs['merge_probes'] \
            else [curr_df.probe.to_list()]

        # load behavioral data (same for all probes); logging occurs within load_behavior function
        dlc_dict = load_behavior(eid, kwargs['target'], one=one)

    if dlc_dict['skip']:
        continue

    logging.log(logging.DEBUG, f"{i}, session: {eid}, subject: {subject}")

    for (probes, pids) in zip(probe_lst, pids_lst):

        # load neural data
        try:
            if kwargs['neural_dtype'] == 'ephys':
                if LOAD_FROM_CACHE:
                    neural_dict = load_pickle_data(curr_df.reg_file.iloc[0])
                else:
                    neural_dict = load_ephys(
                        eid, pids, one=one, ret_qc=True, max_len=5, abswheel=False, wheel=False)
            else:
                raise NotImplementedError
        except Exception as e:
            logging.log(logging.CRITICAL, e)
            continue

        # fit model
        metadata = {
            'eid': eid,
            'subject': subject,
            'probes': probes,
            'merge_probes': kwargs['merge_probes'],
        }

        filenames = fit_eid(
            neural_dict=neural_dict,
            trials_df=neural_dict['trials_df'],
            metadata=metadata,
            dlc_dict=dlc_dict,
            pseudo_ids=[-1],
            # pseudo_ids=1 + np.arange(kwargs['n_pseudo']),
            **copy.copy(kwargs)
        )
