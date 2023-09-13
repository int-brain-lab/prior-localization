import pickle
from prior_localization.functions.decoding import fit_session_widefield
import numpy as np
from one.api import ONE
from pathlib import Path
import matplotlib.pyplot as plt

root_path = Path('/home/julia/workspace/int-brain-lab/prior-localization/prior_localization/tests/fixtures/decoding/wfield')
one = ONE()
eid = 'ff7a70f5-a2b6-4e7e-938e-e7208e0678c2'
subject = 'wfi9'
pseudo_ids = np.concatenate((-np.ones(1), np.arange(1, 3))).astype('int64')

results_fit_eid = fit_session_widefield(
    one, eid, subject, hemisphere=("left", "right"), model='optBay',  pseudo_ids=pseudo_ids, target='pLeft',
        align_event='stimOn_times', frame_window=(-2, -2), output_dir=None, min_trials=150,
        stage_only=False, integration_test=True)

for fit_file in results_fit_eid:
    region = fit_file.name.split('_')[0]
    fit_content = pickle.load(open(fit_file, 'rb'))
    predictions = np.vstack(
        [np.array(fit_content['fit'][k]['predictions_test']).squeeze() for k in range(len(fit_content['fit']))])
    Rsquareds = np.array([fit_content['fit'][k]['Rsquared_test_full'] for k in range(len(fit_content['fit']))])
    predictions_expected = np.load(root_path.joinpath(f"wfi_{region}_predictions.npy"))
    Rsquareds_expected = np.load(root_path.joinpath(f"wfi_{region}_rsquareds.npy"))
    for i in range(predictions.shape[0]):
        fig = plt.figure()
        plt.title(f'{region}, {round(Rsquareds_expected[0], 2)}, {round(Rsquareds[0], 2)}')
        plt.plot(predictions_expected[i, :], label='expected', color='blue', alpha=0.5)
        plt.plot(predictions[i, :], label='actual', color='red', alpha=0.5)
        plt.legend()
        plt.savefig(f'/home/julia/data/prior_review/new_wfield/{region}.png')
        plt.close(fig)
    #np.testing.assert_allclose(predictions_expected, predictions, rtol=1e-5, atol=0)
    #np.testing.assert_allclose(Rsquareds, Rsquareds_expected, rtol=1e-5, atol=0)
    # fit_file.unlink()  # remove predicted path