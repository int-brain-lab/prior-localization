import pandas as pd
from prior_localization.params import kwargs
from prior_localization.functions.decoding import fit_session
import numpy as np
from prior_localization.params import IMPOSTER_SESSION_PATH
from one.api import ONE
from brainwidemap import bwm_query, load_good_units
from brainbox.io.one import SessionLoader

T_BEF = 0.6
T_AFT = 0.6
BINWIDTH = 0.02
idx = 0
pseudo_ids = np.arange(-1, 49)  # if this contains -1, fitting will also be run on the real session

one = ONE()
bwm_df = bwm_query(one)

metadata = {
    'subject': bwm_df.iloc[idx]['subject'],
    'eid': bwm_df.iloc[idx]['eid'],
    'probe_name': bwm_df.iloc[idx]['probe_name']
}

# Load trials df and add start and end times (for now)
sess_loader = SessionLoader(one, metadata['eid'])
sess_loader.load_trials()
sess_loader.trials['trial_start'] = sess_loader.trials['stimOn_times'] - T_BEF
sess_loader.trials['trial_end'] = sess_loader.trials['stimOn_times'] + T_AFT


# Load spike sorting data and put it in a dictionary for now
spikes, clusters = load_good_units(one, bwm_df.iloc[idx]['pid'], eid=bwm_df.iloc[idx]['eid'],
                                   pname=bwm_df.iloc[idx]['probe_name'])

neural_dict = {
    'spk_times': spikes['times'],
    'spk_clu': spikes['clusters'],
    'clu_regions': clusters['acronym'],
    'clu_qc': {k: np.asarray(v) for k, v in clusters.to_dict('list').items()},
    'clu_df': clusters
}

if kwargs['use_imposter_session']:
    kwargs['imposterdf'] = pd.read_parquet(IMPOSTER_SESSION_PATH.joinpath('imposterSessions_beforeRecordings.pqt'))
else:
    kwargs['imposterdf'] = None

results_fit_session = fit_session(neural_dict=neural_dict, trials_df=sess_loader.trials, metadata=metadata,
                                  pseudo_ids=pseudo_ids, **kwargs)

