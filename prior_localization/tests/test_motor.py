import unittest
import tempfile
import numpy as np
from pathlib import Path

from prior_localization.decoding.functions.process_motors import preprocess_motors, aggregate_on_timeWindow
cache_motor_functions = __import__('prior_localization.pipelines.04_cache_motor', fromlist=('prior_localization.pipelines'))


class TestMotor(unittest.TestCase):
    def setUp(self) -> None:
        self.eid = '56956777-dca5-468c-87cb-78150432cc57'
        self.kwargs = {'time_window': [-0.6, -0.1]}
        self.fixtures = Path(__file__).parent.joinpath('fixtures')
        self.expected = np.load(self.fixtures.joinpath(f'motor_regressors_{self.eid}.npy'))
        self.temp_dir = tempfile.TemporaryDirectory()
        #self.expected = np.load(Path(__file__).parent.joinpath(f'fixtures/motor_regressors_{self.eid}.npy'))

    def test_cached_preprocess(self):
        # test the function that generates the motor signals from the pre-generated cache stored in fixtures
        predicted = preprocess_motors(eid=self.eid, kwargs=self.kwargs, cache_path=self.fixtures)
        self.assertIsNone(np.testing.assert_equal(np.array(predicted).squeeze(), self.expected))
        self.assertIsInstance(predicted, list)
        self.assertEqual(len(np.array(predicted).shape), 3)

    def test_standalone_preprocess(self):
        regressors = cache_motor_functions.load_motor(self.eid)
        motor_signals_of_interest = ['licking', 'whisking_l', 'whisking_r', 'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']
        predicted = aggregate_on_timeWindow(regressors, self.kwargs, motor_signals_of_interest)
        self.assertIsNone(np.testing.assert_equal(np.array(predicted).squeeze(), self.expected))
        self.assertIsInstance(predicted, list)
        self.assertEqual(len(np.array(predicted).shape), 3)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()




