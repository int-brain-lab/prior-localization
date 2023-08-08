import pickle
from prior_localization.functions.decoding import fit_session_widefield
import numpy as np
from one.api import ONE


def extract_tested_variables(fit_content):
    predictions = np.vstack([np.array(fit_content['fit'][k]['predictions_test']).squeeze() for k in range(len(fit_content['fit']))])
    Rsquareds = np.array([fit_content['fit'][k]['Rsquared_test_full'] for k in range(len(fit_content['fit']))])
    return predictions, Rsquareds


root_path = '/home/julia/workspace/int-brain-lab/prior-localization/prior_localization/tests/fixtures/decoding/wfield'
one = ONE()
eid = 'ff7a70f5-a2b6-4e7e-938e-e7208e0678c2'
subject = 'wfi9'
pseudo_ids = np.concatenate((-np.ones(1), np.arange(1, 3))).astype('int64')
# file_path = Path(f'resources/widefield_prior/{eid}_img.npy')
# save_path = Path('/home/julia/data/prior_review/aws_download_widefield_for_test')
# if not save_path.exists():
#     s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
#     aws.s3_download_file(file_path, save_path, s3=s3, bucket_name=bucket_name)

# using caching function from Mayo
# neural_dict = prepare_widefield_data(eid=eid, one=one)
# neural_dict['timings']['stimOn_times'] = neural_dict['timings']['stimOn_times'].astype(int)
# sess_loader = SessionLoader(one, eid)
# sess_loader.load_trials()
# trialsdf = sess_loader.trials
# trialsdf['choice'] = trialsdf['choice'].astype(int)
#
# neural_dict['regions'] = np.load('/home/julia/workspace/int-brain-lab/prior-localization/prior_localization/tests/fixtures/regions_wfi.npy') # take Chris' regions
# neural_dict['activity'] = np.load('/home/julia/data/prior_review/aws_download_widefield_for_test') # take Chris' activity
# neural_dict['timings'] = get_original_timings(eid) # take Chris' times
# # regressors = [trialsdf, neural_dict]
# metadata = {
#     'eid': 'wfi9s0',
#     'subject': 'wfi9',
#     'hemispheres': 'both_hemispheres',
#     'eids_train': ['wfi9s0']
# }
results_fit_eid = fit_session_widefield(
    one, eid, subject, hemisphere=("left", "right"), model='optBay',  pseudo_ids=pseudo_ids, target='pLeft',
        align_event='stimOn_times', frame_window=(-2, 2), output_dir=None, min_trials=150,
        motor_residuals=False, compute_neurometrics=False, stage_only=False, integration_test=True)

for fit_file in results_fit_eid:
    predicteds = pickle.load(open(fit_file, 'rb'))
    predictions, Rsquareds = extract_tested_variables(predicteds)
    predictions_expected = np.load(root_path.joinpath(f"wfi_{fit_file.name.split('_')[0]}_predictions.npy"))
    Rsquareds_expected = np.load(root_path.joinpath(f"wfi_{fit_file.name.split('_')[0]}_rsquareds.npy"))
    np.testing.assert_allclose(predictions_expected, predictions, rtol=5e-2, atol=0)
    np.testing.assert_allclose(Rsquareds, Rsquareds_expected, rtol=1e-2, atol=0)
    fit_file.unlink()  # remove predicted path
