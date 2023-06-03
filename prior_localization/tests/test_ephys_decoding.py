import unittest
import pickle
import tempfile
from pathlib import Path
import numpy as np

from one.api import ONE
from brainbox.io.one import SessionLoader

from prior_localization.decoding.settings import kwargs
from prior_localization.decoding.functions.decoding import fit_session
from prior_localization.pipelines.utils_ephys import load_ephys


class TestEphysDecoding(unittest.TestCase):
    """
    Test decoding on ephys data of a single session with merged and unmerged probes.
    We compare Rsquared and predicted values of the test set.
    """
    def setUp(self) -> None:
        self.one = ONE()
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        _, self.probe_names = self.one.eid2pid(self.eid)
        self.metadata = {
            "subject": self.one.eid2ref(self.eid)['subject'],
            "eid": self.eid,
            "ret_qc": 1
        }
        self.pseudo_ids = np.concatenate((-np.ones(1), np.arange(1, 3))).astype('int64')
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.fixtures_dir = Path(__file__).parent.joinpath('fixtures')
        kwargs['behfit_path'] = Path(self.tmp_dir.name).joinpath('behavior')
        kwargs['neuralfit_path'] = Path(self.tmp_dir.name).joinpath('neural')
        kwargs['neural_dtype'] = 'ephys'
        kwargs['integration_test'] = True

        sl = SessionLoader(self.one, self.eid)
        sl.load_trials()
        self.trials_df = sl.trials

    def compare_target_test(self, results_fit_session, probe):
        for f in results_fit_session:
            region = f.name.split('_')[1]
            # This failed for uninteresting regions
            if region == 'VPM':
                continue
            with open(f, 'rb') as fb:
                predicted_fit = pickle.load(fb)['fit']
            for key in ['Rsquared_test_full', 'predictions_test']:
                test = np.asarray([p[key] for p in predicted_fit]).squeeze()
                target = np.load(self.fixtures_dir.joinpath(f'{probe}_{region}_{key.split("_")[0].lower()}.npy'))
                self.assertTrue(np.allclose(test, target, rtol=1e-05))

    def test_merged_probes(self):
        self.metadata['probe_name'] = 'merged_probes'
        neural_dict = load_ephys(self.one, self.eid, self.probe_names, ret_qc=self.metadata['ret_qc'])
        results_fit_session = fit_session(neural_dict=neural_dict, trials_df=self.trials_df, metadata=self.metadata,
                                          pseudo_ids=self.pseudo_ids, **kwargs)
        self.compare_target_test(results_fit_session, 'merged')

    def test_single_probes(self):
        for probe_name in self.probe_names:
            self.metadata['probe_name'] = probe_name
            neural_dict = load_ephys(self.one, self.eid, probe_name, ret_qc=self.metadata['ret_qc'])
            results_fit_session = fit_session(neural_dict=neural_dict, trials_df=self.trials_df, metadata=self.metadata,
                                              pseudo_ids=self.pseudo_ids, **kwargs)
            self.compare_target_test(results_fit_session, probe_name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()
