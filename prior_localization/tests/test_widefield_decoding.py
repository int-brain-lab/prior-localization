import pickle
from prior_pipelines.decoding.functions.decoding import fit_eid
import numpy as np
from prior_pipelines.pipelines.wfi_utils import prepare_widefield_data, get_original_timings
from one.api import ONE
import brainbox.io.one as bbone
from one.remote import aws
from pathlib import Path 

# function to extract tested variable from the output of the decoding script
def extract_tested_variables(fit_content):
    predictions = np.vstack([np.array(fit_content['fit'][k]['predictions_test']).squeeze() for k in range(len(fit_content['fit']))])
    Rsquareds = np.array([fit_content['fit'][k]['Rsquared_test_full'] for k in range(len(fit_content['fit']))])
    return predictions, Rsquareds

# settings and metadata
from prior_pipelines.decoding.settings import kwargs
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
    results_fit_eid = fit_eid(neural_dict=neural_dict, trials_df=trials_df, metadata=metadata,
                            pseudo_ids=pseudo_ids, **kwargs)
    return results_fit_eid # these are filenames

# download from aws
one = ONE(base_url='https://alyx.internationalbrainlab.org')
eid = 'ff7a70f5-a2b6-4e7e-938e-e7208e0678c2'
file_path = Path(f'resources/widefield_prior/{eid}_img.npy')
save_path = Path('aws_download_widefield_for_test')

if not save_path.exists():
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_file(file_path, save_path, s3=s3, bucket_name=bucket_name)

# using caching function from Mayo
kwargs['run_integration_test'] = True
eid = 'ff7a70f5-a2b6-4e7e-938e-e7208e0678c2'
neural_dict = prepare_widefield_data(eid=eid, one=one)
neural_dict['timings']['stimOn_times'] = neural_dict['timings']['stimOn_times'].astype(int)
sess_loader = bbone.SessionLoader(one, eid)
if sess_loader.trials.empty:
    sess_loader.load_trials()
trialsdf = sess_loader.trials
trialsdf['choice'] = trialsdf['choice'].astype(int)

neural_dict['regions'] = np.load('./fixtures/regions_wfi.npy') # take Chris' regions
neural_dict['activity'] = np.load('aws_download_widefield_for_test') # take Chris' activity
neural_dict['timings'] = get_original_timings(eid) # take Chris' times
regressors = [trialsdf, neural_dict]
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

