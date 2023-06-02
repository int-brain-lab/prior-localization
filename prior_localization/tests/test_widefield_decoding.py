'''
import pandas as pd
import sys
from prior_localization.decoding.settings import kwargs
from prior_localization.decoding.functions.decoding import fit_eid
import numpy as np
from prior_localization.pipelines.wfi_utils import prepare_widefield_data
from one.api import ONE
import brainbox.io.one as bbone

# using caching function from Mayo
eid = 'ff7a70f5-a2b6-4e7e-938e-e7208e0678c2'
one=ONE(base_url='https://alyx.internationalbrainlab.org')
neural_dict = prepare_widefield_data(eid=eid, one=one)
neural_dict['timings']['stimOn_times'] = neural_dict['timings']['stimOn_times'].astype(int)
sess_loader = bbone.SessionLoader(one, eid)
if sess_loader.trials.empty:
    sess_loader.load_trials()

kwargs['set_seed_for_DEBUG'] = True


# using my caching function for ground truth
i_eid = 0
HEMISPHERES = ['left', 'right']
kwargs['set_seed_for_DEBUG'] = True
kwargs['run_integration_test'] = True # if you put False, you will overwrite the existing "groundtruth"
subjects = glob.glob(WIDE_FIELD_PATH.joinpath('CSK-im-*').as_posix())
nb_eids_per_subject = np.array([np.load(s + '/behavior.npy', allow_pickle=True).size for s in subjects])
subj_id = np.sum(i_eid >= nb_eids_per_subject.cumsum())
sess_id = i_eid - np.hstack((0, nb_eids_per_subject)).cumsum()[:-1][subj_id]
sessiondf, wideFieldImaging_dict, metadata = load_wfi_session(subjects[subj_id], sess_id, HEMISPHERES, WIDE_FIELD_PATH.as_posix())


metadata = {
    'eid': 'wfi9s0',
    'subject': 'wfi9',
    'hemispheres': 'both_hemispheres',
    'eids_train': ['wfi9s0']
}

regressors = [sess_loader.trials, neural_dict]

# decoding function
def decode(regressors, metadata, nb_pseudo_sessions=3):
    if kwargs['neural_dtype'] == 'widefield':
        trials_df, neural_dict = regressors
    else:
        trials_df, neural_dict = regressors['trials_df'], regressors

    # pseudo_id=-1 decodes the prior of the session, pseudo_id>0 decodes pseudo-priors
    pseudo_ids = np.concatenate((-np.ones(1), np.arange(1, nb_pseudo_sessions))).astype('int64')
    results_fit_eid = fit_eid(neural_dict=neural_dict, trials_df=trials_df, metadata=metadata,
                            pseudo_ids=pseudo_ids, **kwargs)
    return results_fit_eid # these are filenames

# launch decoding
results_fit_eid = []
results_fit_eid.extend(decode(regressors, metadata))

if kwargs['run_integration_test']:
    import pickle
    for f in results_fit_eid:        
        predicteds = pickle.load(open(f, 'rb'))
        targets = pickle.load(open(f.parent.joinpath(f.name.replace('_to_be_tested', '')), 'rb'))
        targets_keys = list(targets['fit'].keys())
        for k in targets_keys:
            if k not in ['Rsquared_test_full', 'predictions_test']:
                targets.pop(k)
        for (fit_pred, fit_targ) in zip(predicteds['fit'], targets['fit']):
            np.testing.assert_allclose(fit_pred['Rsquared_test_full'], fit_targ['Rsquared_test_full'], rtol=1e-2, atol=0)
            np.testing.assert_allclose(fit_pred['predictions_test'], fit_targ['predictions_test'], rtol=5e-2, atol=0)
        f.unlink() # remove predicted path

print('job successful')

'''