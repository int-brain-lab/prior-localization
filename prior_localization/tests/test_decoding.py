import unittest
import pickle
import tempfile
from pathlib import Path
import numpy as np

from one.api import ONE
from prior_localization.fit_data import fit_session_ephys, fit_session_pupil, fit_session_motor, fit_session_widefield


class TestEphysDecoding(unittest.TestCase):
    """
    Test decoding on ephys data of a single session with merged and unmerged probes.
    We compare Rsquared and predicted values of the test set.
    """
    def setUp(self) -> None:
        self.one = ONE(base_url='https://openalyx.internationalbrainlab.org')
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        _, self.probe_names = self.one.eid2pid(self.eid)
        self.qc = 1
        self.subject = self.one.eid2ref(self.eid)['subject']
        self.pseudo_ids = [-1, 1, 2]
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.fixtures_dir = Path(__file__).parent.joinpath('fixtures', 'decoding', 'ephys')

    def compare_target_test(self, results_fit_session, probe):
        for f in results_fit_session:
            region = f.name.split('_')[0]
            with open(f, 'rb') as fb:
                predicted_fit = pickle.load(fb)['fit']
            for key in ['Rsquared_test_full', 'predictions_test']:
                test = np.asarray([p[key] for p in predicted_fit]).squeeze()
                target = np.load(self.fixtures_dir.joinpath(f'{probe}_{region}_{key.split("_")[0].lower()}.npy'))
                self.assertTrue(np.allclose(test, target, rtol=1e-05))

    def test_merged_probes(self):
        results_fit_session = fit_session_ephys(
            one=self.one, session_id=self.eid, subject=self.subject, probe_name=self.probe_names,
            output_dir=Path(self.tmp_dir.name), pseudo_ids=self.pseudo_ids, n_runs=2, integration_test=True
        )
        self.compare_target_test(results_fit_session, 'merged')

    def test_single_probes(self):
        for probe_name in self.probe_names:
            results_fit_session = fit_session_ephys(
                one=self.one, session_id=self.eid, subject=self.subject, probe_name=probe_name,
                output_dir=Path(self.tmp_dir.name), pseudo_ids=self.pseudo_ids, n_runs=2, integration_test=True
            )
            self.compare_target_test(results_fit_session, probe_name)

    def test_stage_only(self):
        results_fit_session = fit_session_ephys(
            one=self.one, session_id=self.eid, subject=self.subject, probe_name=self.probe_names,
            output_dir=Path(self.tmp_dir.name), pseudo_ids=self.pseudo_ids, n_runs=2, stage_only=True,
            integration_test=True
        )
        self.assertIsNone(results_fit_session)

    def test_actKernel(self):
        results_fit_session = fit_session_ephys(
            one=self.one, session_id=self.eid, subject=self.subject, probe_name=self.probe_names,
            output_dir=Path(self.tmp_dir.name), pseudo_ids=self.pseudo_ids, model='actKernel', n_runs=2,
            integration_test=True
        )
        self.assertEqual(len(results_fit_session), 5)

    def test_motor_residuals(self):
        # TODO: get actual results?
        results_fit_session = fit_session_ephys(
            one=self.one, session_id=self.eid, subject=self.subject, probe_name=self.probe_names,
            output_dir=Path(self.tmp_dir.name), pseudo_ids=self.pseudo_ids, n_runs=2, motor_residuals=True,
            integration_test=True
        )
        self.assertEqual(len(results_fit_session), 5)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()


class TestMotorEyeDecoding(unittest.TestCase):
    """
    Test decoding of motor regressors and eye movement for controls
    """

    # TODO: test actual fit?
    def setUp(self) -> None:
        self.one = ONE(base_url='https://openalyx.internationalbrainlab.org')
        self.eid = '4a45c8ba-db6f-4f11-9403-56e06a33dfa4'
        self.subject = self.one.eid2ref(self.eid)['subject']
        self.pseudo_ids = [-1, 1, 2]
        self.tmp_dir = tempfile.TemporaryDirectory()

    def test_decode_pupil(self):
        # We need an eid that has LP data
        results_fit_session = fit_session_pupil(
            one=self.one, session_id=self.eid, subject=self.subject, output_dir=Path(self.tmp_dir.name),
            n_runs=2, integration_test=True
        )
        self.assertTrue(results_fit_session.name.startswith('pupil'))
        with open(results_fit_session, 'rb') as fb:
            predicted = pickle.load(fb)
        self.assertEqual(predicted['subject'], self.subject)
        self.assertEqual(predicted['eid'], self.eid)
        # Reset to original eid for other tests

    def test_decode_motor(self):
        results_fit_session = fit_session_motor(
            one=self.one, session_id=self.eid, subject=self.subject, output_dir=Path(self.tmp_dir.name),
            n_runs=2, integration_test=True
        )
        self.assertTrue(results_fit_session.name.startswith('motor'))
        with open(results_fit_session, 'rb') as fb:
            predicted = pickle.load(fb)
        self.assertEqual(predicted['subject'], self.subject)
        self.assertEqual(predicted['eid'], self.eid)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()


class TestWidefieldDecoding(unittest.TestCase):
    """
    Test decoding on widefield data of a single session. We compare Rsquared and predicted values of the test set.
    """
    def setUp(self) -> None:
        self.one = ONE(base_url='https://openalyx.internationalbrainlab.org')
        self.eid = 'ff7a70f5-a2b6-4e7e-938e-e7208e0678c2'
        self.subject = self.one.eid2ref(self.eid)['subject']
        self.pseudo_ids = [-1, 1, 2]
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.fixtures_dir = Path(__file__).parent.joinpath('fixtures', 'decoding', 'wfield')

    def compare_target_test(self, results_fit_session, fixtures_dir):
        for f in results_fit_session:
            region = f.name.split('_')[0]
            with open(f, 'rb') as fb:
                predicted_fit = pickle.load(fb)['fit']
            for key in ['Rsquared_test_full', 'predictions_test']:
                test = np.asarray([p[key] for p in predicted_fit]).squeeze()
                target = np.load(fixtures_dir.joinpath(f'wfi_{region}_{key.split("_")[0].lower()}.npy'))
                self.assertTrue(np.allclose(test, target, rtol=1e-04))

    def test_ONE_data(self):
        results_fit_session = fit_session_widefield(
            one=self.one, session_id=self.eid, subject=self.subject, output_dir=Path(self.tmp_dir.name),
            pseudo_ids=self.pseudo_ids, n_runs=2, integration_test=True
        )
        self.compare_target_test(results_fit_session, self.fixtures_dir.joinpath('new'))

    @unittest.skip("Currently only testing this on my machine because the test data is too big to ship on github")
    def test_chris_data(self):
        results_fit_session = fit_session_widefield(
            one=self.one, session_id=self.eid, subject=self.subject, output_dir=Path(self.tmp_dir.name),
            pseudo_ids=self.pseudo_ids, n_runs=2, integration_test=True, old_data=self.fixtures_dir.joinpath('old')
        )
        self.compare_target_test(results_fit_session, self.fixtures_dir.joinpath('old'))

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
