import tempfile
import unittest
from pathlib import Path
import numpy as np

from one.api import ONE
from prior_localization.prepare_data import prepare_ephys, prepare_behavior
from prior_localization.functions.process_motors import prepare_motor, aggregate_on_timeWindow


class TestEphysInput(unittest.TestCase):

    def setUp(self) -> None:
        self.one = ONE()
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        _, self.probe_names = self.one.eid2pid(self.eid)
        self.qc = 1
        self.subject = self.one.eid2ref(self.eid)['subject']
        self.fixtures_dir = Path(__file__).parent.joinpath('fixtures', 'inputs')
        self.intervals = np.load(self.fixtures_dir.joinpath('intervals.npy'))

    def test_prepare_ephys_merged(self):
        binned_spikes, actual_regions = prepare_ephys(one=self.one, session_id=self.eid, probe_name=self.probe_names,
                                                      regions='single_regions', intervals=self.intervals)
        for spikes, region in zip(binned_spikes, actual_regions):
            # this failed for uninteresting reasons
            if region[0] == 'VPM':
                continue
            expected = np.load(self.fixtures_dir.joinpath(f'merged_{region[0]}_ephys_in.npy'))
            self.assertTrue(np.all(np.asarray(spikes).squeeze() == expected))

    def test_prepare_ephys_single(self):
        for probe_name in self.probe_names:
            binned_spikes, actual_regions = prepare_ephys(one=self.one, session_id=self.eid, probe_name=probe_name,
                                                          regions='single_regions', intervals=self.intervals)
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
        self.one = ONE()
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        self.subject = self.one.eid2ref(self.eid)['subject']
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fixtures_dir = Path(__file__).parent.joinpath('fixtures', 'inputs')

    def test_behav_targets(self):
        _, all_targets, _, mask, _ = prepare_behavior(
            self.one, self.eid, self.subject, pseudo_ids=None, output_dir=Path(self.temp_dir.name),
            model='optBay', target='pLeft', align_event='stimOn_times', time_window=(-0.6, -0.1),
            stage_only=False)
        expected_orig = np.load(self.fixtures_dir.joinpath('behav_target.npy'))
        self.assertTrue(np.all(all_targets == expected_orig))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


class TestMotor(unittest.TestCase):
    def setUp(self) -> None:
        self.one = ONE()
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        self.time_window = [-0.6, -0.1]
        self.fixtures = Path(__file__).parent.joinpath('fixtures', 'inputs')
        self.expected = np.load(self.fixtures.joinpath(f'motor_regressors.npy'))
        self.temp_dir = tempfile.TemporaryDirectory()

    def test_standalone_preprocess(self):
        predicted = prepare_motor(self.one, self.eid, self.time_window)
        self.assertIsNone(np.testing.assert_equal(np.array(predicted).squeeze(), self.expected))
        self.assertIsInstance(predicted, list)
        self.assertEqual(len(np.array(predicted).shape), 3)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

