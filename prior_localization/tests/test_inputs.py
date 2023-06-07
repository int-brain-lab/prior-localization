import tempfile
import unittest
from pathlib import Path
import numpy as np

from one.api import ONE
from prior_localization.prepare_data import prepare_ephys, prepare_behavior
from prior_localization.functions.process_targets import optimal_Bayesian


class TestEphysInput(unittest.TestCase):

    def setUp(self) -> None:
        self.one = ONE()
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        _, self.probe_names = self.one.eid2pid(self.eid)
        self.qc = 1
        self.subject = self.one.eid2ref(self.eid)['subject']
        self.fixtures_dir = Path(__file__).parent.joinpath('fixtures')
        self.intervals = np.load(self.fixtures_dir.joinpath('intervals.npy'))

    def test_prepare_ephys_merged(self):
        binned_spikes, actual_regions, n_units = prepare_ephys(self.one, self.eid, self.probe_names,
                                                               'single_regions', self.intervals)
        for spikes, region in zip(binned_spikes, actual_regions):
            if region[0] == 'VPM':
                continue
            expected = np.load(self.fixtures_dir.joinpath(f'merged_{region[0]}_ephys_in.npy'))
            self.assertTrue(np.all(np.asarray(spikes).squeeze() == expected))

    def test_prepare_ephys_single(self):
        for probe_name in self.probe_names:
            binned_spikes, actual_regions, n_units = prepare_ephys(self.one, self.eid, probe_name,
                                                                   'single_regions', self.intervals)
            for spikes, region in zip(binned_spikes, actual_regions):
                if region[0] == 'VPM':
                    continue
                expected = np.load(self.fixtures_dir.joinpath(f'{probe_name}_{region[0]}_ephys_in.npy'))
                self.assertTrue(np.all(np.asarray(spikes).squeeze() == expected))


class TestBehaviorInputs(unittest.TestCase):

    def setUp(self) -> None:
        self.one = ONE()
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        self.subject = self.one.eid2ref(self.eid)['subject']
        self.temp_dir = tempfile.TemporaryDirectory()

    def test_with_pseudo(self):
        _, all_targets, _, mask, _ = prepare_behavior(
            self.one, self.eid, self.subject, Path(self.temp_dir.name), model=optimal_Bayesian,
            pseudo_ids=np.concatenate((-np.ones(1), np.arange(1, 3))).astype('int64'),
            target='pLeft', align_event='stimOn_times', time_window=(-0.6, -0.1), integration_test=True)
        expected_orig = np.load(Path(__file__).parent.joinpath('fixtures', 'behav_target.npy'))
        expected_1 = np.load(Path(__file__).parent.joinpath('fixtures', 'behav_target_pseudo_merged_BMA_1.npy'))
        expected_2 = np.load(Path(__file__).parent.joinpath('fixtures', 'behav_target_pseudo_merged_BMA_2.npy'))

        self.assertTrue(np.all(np.asarray([all_targets[0][m] for m in np.squeeze(np.where(mask))]) == expected_orig))
        # self.assertTrue(np.all(np.asarray([all_targets[1][m] for m in np.squeeze(np.where(mask))]) == expected_1))
        # self.assertTrue(np.all(np.asarray([all_targets[2][m] for m in np.squeeze(np.where(mask))]) == expected_2))
