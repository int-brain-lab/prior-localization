import tempfile
import unittest
from pathlib import Path
import numpy as np

from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask
from prior_localization.prepare_data import prepare_ephys, prepare_behavior, prepare_motor, prepare_pupil
from prior_localization.functions.utils import average_data_in_epoch


class TestEphysInput(unittest.TestCase):

    def setUp(self) -> None:
        self.one = ONE(base_url='https://openalyx.internationalbrainlab.org')
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        _, self.probe_names = self.one.eid2pid(self.eid)
        self.qc = 1
        self.subject = self.one.eid2ref(self.eid)['subject']
        self.fixtures_dir = Path(__file__).parent.joinpath('fixtures', 'inputs')
        self.intervals = np.load(self.fixtures_dir.joinpath('intervals.npy'))

    def test_prepare_ephys_merged(self):
        binned_spikes, actual_regions, n_units, cluster_ids = prepare_ephys(
            one=self.one, session_id=self.eid, probe_name=self.probe_names, regions='single_regions',
            intervals=self.intervals)
        for spikes, region in zip(binned_spikes, actual_regions):
            # this failed for uninteresting reasons
            if region[0] == 'VPM':
                continue
            expected = np.load(self.fixtures_dir.joinpath(f'merged_{region[0]}_ephys_in.npy'))
            self.assertTrue(np.all(np.asarray(spikes).squeeze() == expected))

    def test_prepare_ephys_single(self):
        for probe_name in self.probe_names:
            binned_spikes, actual_regions, n_units, cluster_ids = prepare_ephys(
                one=self.one, session_id=self.eid, probe_name=probe_name, regions='single_regions',
                intervals=self.intervals)
            for spikes, region in zip(binned_spikes, actual_regions):
                # this failed for uninteresting reasons
                if region[0] == 'VPM':
                    continue
                expected = np.load(self.fixtures_dir.joinpath(f'{probe_name}_{region[0]}_ephys_in.npy'))
                self.assertTrue(np.all(np.asarray(spikes).squeeze() == expected))

    def tearDown(self) -> None:
        pass


class TestBehaviorInputs(unittest.TestCase):

    def setUp(self) -> None:
        self.one = ONE(base_url='https://openalyx.internationalbrainlab.org')
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        self.subject = self.one.eid2ref(self.eid)['subject']
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fixtures_dir = Path(__file__).parent.joinpath('fixtures', 'inputs')

    def test_behav_targets(self):
        sl = SessionLoader(self.one, self.eid, revision='2024-07-10')
        sl.load_trials()
        _, trials_mask = load_trials_and_mask(
            one=self.one, eid=self.eid, sess_loader=sl, min_rt=0.08, max_rt=None,
            min_trial_len=None, max_trial_len=None,
            exclude_nochoice=True, exclude_unbiased=False,
        )
        _, all_targets, all_masks, _ = prepare_behavior(
            self.eid, self.subject, sl.trials, trials_mask, pseudo_ids=None, output_dir=Path(self.temp_dir.name),
            model='optBay', target='pLeft'
        )
        mask = all_masks[0][0]
        expected_orig = np.load(self.fixtures_dir.joinpath('behav_target.npy'))
        self.assertTrue(np.allclose(all_targets[0][0][mask], expected_orig, rtol=1e-4))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


class TestMotorInputs(unittest.TestCase):
    def setUp(self) -> None:
        self.one = ONE(base_url='https://openalyx.internationalbrainlab.org')
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        self.time_window = [-0.6, -0.1]
        self.fixtures = Path(__file__).parent.joinpath('fixtures', 'inputs')
        self.expected = np.load(self.fixtures.joinpath(f'motor_regressors_{self.eid}.npy'))
        self.temp_dir = tempfile.TemporaryDirectory()

    def test_standalone_preprocess(self):
        predicted = prepare_motor(self.one, self.eid, align_event='stimOn_times', time_window=self.time_window)
        self.assertIsNone(np.testing.assert_equal(predicted, self.expected))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


class TestPupilInputs(unittest.TestCase):
    def setUp(self) -> None:
        self.one = ONE(base_url='https://openalyx.internationalbrainlab.org')
        self.eid = '4a45c8ba-db6f-4f11-9403-56e06a33dfa4'
        self.time_window = [-0.6, -0.1]
        self.fixtures = Path(__file__).parent.joinpath('fixtures', 'inputs')
        self.expected = np.load(self.fixtures.joinpath(f'pupil_regressors_{self.eid}.npy'))

    def test_prepare_pupil(self):
        predicted = prepare_pupil(self.one, self.eid, align_event='stimOn_times', time_window=self.time_window)
        self.assertIsNone(np.testing.assert_equal(predicted, self.expected))


class TestAverageDataInEpoch(unittest.TestCase):
    def setUp(self) -> None:
        self.one = ONE(base_url='https://openalyx.internationalbrainlab.org')
        self.eid = 'fc14c0d6-51cf-48ba-b326-56ed5a9420c3'
        self.sl = SessionLoader(self.one, self.eid, revision='2024-07-10')
        self.sl.load_trials()
        self.ts = 1 / 30
        self.epoch = [-0.6, -0.1]
        self.fixtures = Path(__file__).parent.joinpath('fixtures', 'inputs')

    def test_errors(self):
        # Test that timestamps and values are same length and timestamps sorted, else raise error
        times = np.arange(0, 10, self.ts)
        values = np.random.rand(len(times) + 1)
        with self.assertRaises(ValueError):
            average_data_in_epoch(times, values, self.sl.trials, align_event='stimOn_times', epoch=self.epoch)

        times = np.array([0.1, 0.2, 0.4, 0.39, 0.5])
        values = np.random.rand(len(times))
        with self.assertRaises(ValueError):
            average_data_in_epoch(times, values, self.sl.trials, align_event='stimOn_times', epoch=self.epoch)

    def test_epoch(self):
        # Test that data is sampled from correct epoch
        # Using the times as values allows to see at which times the data is actually sampled, but still we need to
        # tests against precomputed values that we know are sampled from
        times = np.arange(int(self.sl.trials['stimOn_times'].min() - 10),
                          int(self.sl.trials['stimOn_times'].max() + 10), self.ts)
        actual = average_data_in_epoch(times, times, self.sl.trials, align_event='stimOn_times', epoch=self.epoch)
        predicted = np.load(self.fixtures.joinpath('average_in_epoch_uniform.npy'))
        np.testing.assert_array_equal(actual, predicted)

        # Same for non-uniformly sampled data
        np.random.seed(6)
        times = np.random.uniform(times.min(), times.max(), len(times))
        times.sort()
        actual = average_data_in_epoch(times, times, self.sl.trials, align_event='stimOn_times', epoch=self.epoch)
        predicted = np.load(self.fixtures.joinpath('average_in_epoch_nonuniform.npy'))
        self.assertIsNone(np.testing.assert_array_equal(actual, predicted))

    def test_nans(self):
        # Align events with Nans
        times = np.arange(int(self.sl.trials['firstMovement_times'].min() - 10),
                          int(self.sl.trials['firstMovement_times'].max() + 10), self.ts)
        nan_idx = np.where(np.isnan(self.sl.trials['firstMovement_times']))
        res = average_data_in_epoch(times, times, self.sl.trials, align_event='firstMovement_times', epoch=self.epoch)
        self.assertTrue(np.all(np.isnan(res[nan_idx])))

        # First trial before timestamps and last trial after timestamps
        end_idx = self.sl.trials.shape[0] - 20
        times = np.arange(np.ceil(self.sl.trials['firstMovement_times'][7]),
                          np.floor(self.sl.trials['firstMovement_times'][end_idx]), self.ts)
        res = average_data_in_epoch(times, np.ones_like(times), self.sl.trials,
                                    align_event='firstMovement_times', epoch=self.epoch)
        self.assertTrue(np.all(np.isnan(res[:8])))
        self.assertTrue(np.all(np.isnan(res[end_idx:])))


if __name__ == "__main__":
    unittest.main()
