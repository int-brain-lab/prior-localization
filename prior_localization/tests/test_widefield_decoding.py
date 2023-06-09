import pickle
from prior_localization.functions.decoding import fit_session
import numpy as np
from prior_localization.functions.wfi_utils import prepare_widefield_data
from one.api import ONE
import brainbox.io.one as bbone


# function to extract tested variable from the output of the decoding script
def extract_tested_variables(fit_content):
    predictions = np.vstack([np.array(fit_content['fit'][k]['predictions_test']).squeeze() for k in range(len(fit_content['fit']))])
    Rsquareds = np.array([fit_content['fit'][k]['Rsquared_test_full'] for k in range(len(fit_content['fit']))])
    return predictions, Rsquareds

# settings and metadata
from prior_localization.params import kwargs
kwargs['set_seed_for_DEBUG'] = True
kwargs['neural_dtype'] = 'widefield'
kwargs['nb_runs'] = 2
kwargs["single_region"] = True
build_test = False

# decoding function
def decode(regressors, metadata, nb_pseudo_sessions=3):
    if kwargs['neural_dtype'] == 'widefield':
        trials_df, neural_dict = regressors
    else:
        trials_df, neural_dict = regressors['trials_df'], regressors

    # pseudo_id=-1 decodes the prior of the session, pseudo_id>0 decodes pseudo-priors
    pseudo_ids = np.concatenate((-np.ones(1), np.arange(1, nb_pseudo_sessions))).astype('int64')
    results_fit_eid = fit_session(neural_dict=neural_dict, trials_df=trials_df, metadata=metadata,
                            pseudo_ids=pseudo_ids, **kwargs)
    return results_fit_eid # these are filenames

if build_test:
    # using the source files from Chris caching function for ground truth
    from pathlib import Path
    import glob
    from prior_localization.functions.wfi_utils import load_wfi_session
    WIDE_FIELD_PATH = Path('/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/prior_pipelines/wide_field_imaging/')
    i_eid = 0
    HEMISPHERES = ['left', 'right']
    subjects = glob.glob(WIDE_FIELD_PATH.joinpath('CSK-im-*').as_posix())
    nb_eids_per_subject = np.array([np.load(s + '/behavior.npy', allow_pickle=True).size for s in subjects])
    subj_id = np.sum(i_eid >= nb_eids_per_subject.cumsum())
    sess_id = i_eid - np.hstack((0, nb_eids_per_subject)).cumsum()[:-1][subj_id]
    sessiondf, wideFieldImaging_dict, metadata = load_wfi_session(subjects[subj_id], sess_id, HEMISPHERES, WIDE_FIELD_PATH.as_posix())
    regressors = [sessiondf, wideFieldImaging_dict]
    metadata = {**metadata, **{'hemispheres': 'both_hemispheres'}}

    import pickle
    results_fit_eid = decode(regressors, metadata)
    for fit_file in results_fit_eid:
        fit_content = pickle.load(open(fit_file, 'rb'))
        predictions, Rsquareds = extract_tested_variables(fit_content=fit_content)
        np.save('fixtures/wfi_{}_predictions.npy'.format(fit_file.name.split('_')[1]), predictions)
        np.save('fixtures/wfi_{}_rsquareds.npy'.format(fit_file.name.split('_')[1]), Rsquareds)
else:
    # using caching function from Mayo
    eid = 'ff7a70f5-a2b6-4e7e-938e-e7208e0678c2'
    one=ONE(base_url='https://alyx.internationalbrainlab.org')
    neural_dict = prepare_widefield_data(eid=eid, one=one)
    neural_dict['timings']['stimOn_times'] = neural_dict['timings']['stimOn_times'].astype(int)
    sess_loader = bbone.SessionLoader(one, eid)
    if sess_loader.trials.empty:
        sess_loader.load_trials()
    trialsdf = sess_loader.trials
    # Chris did not have any NA in his contrasts
    trialsdf['choice'] = trialsdf['choice'].astype(int)

    neural_dict['regions'] = wideFieldImaging_dict['regions']
    neural_dict['activity'] = wideFieldImaging_dict['activity']
    neural_dict['atlas'] = wideFieldImaging_dict['atlas']
    neural_dict['timings'] = wideFieldImaging_dict['timings']
    regressors = [trialsdf, wideFieldImaging_dict]
    metadata = {
        'eid': 'wfi9s0',
        'subject': 'wfi9',
        'hemispheres': 'both_hemispheres',
        'eids_train': ['wfi9s0']
    }
    results_fit_eid = decode(regressors, metadata)
    for fit_file in results_fit_eid:
        predicteds = pickle.load(open(fit_file, 'rb'))
        predictions, Rsquareds = extract_tested_variables(predicteds)
        predictions_expected = np.load('fixtures/wfi_{}_predictions.npy'.format(fit_file.name.split('_')[1]))
        Rsquareds_expected = np.load('fixtures/wfi_{}_rsquareds.npy'.format(fit_file.name.split('_')[1]))
        np.testing.assert_allclose(predictions_expected, predictions, rtol=5e-2, atol=0)
        np.testing.assert_allclose(Rsquareds, Rsquareds_expected, rtol=1e-2, atol=0)
        fit_file.unlink() # remove predicted path


#from matplotlib import pyplot as plt
#plt.imshow(neural_dict['regions'] - wideFieldImaging_dict['regions'])