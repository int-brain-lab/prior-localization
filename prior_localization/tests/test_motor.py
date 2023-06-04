import pandas as pd
import sys
import pickle
from prior_localization.decoding.functions.process_motors import preprocess_motors, aggregate_on_timeWindow
import numpy as np
from one.api import ONE
import brainbox.io.one as bbone
from scipy import stats
from pathlib import Path

# eid and time window
eid = '56956777-dca5-468c-87cb-78150432cc57'
kwargs = {'time_window': [-0.6, -0.1]}

# load final groundtruth motor_signals
expected_motor_signals = np.load(f'./fixtures/motor_regressors_{eid}.npy')

# test the function that generates the motor signals from the pre-generated cache stored in fixtures
predicted_motor_signals = preprocess_motors(eid=eid, kwargs=kwargs, cache_path=Path('./fixtures/'))
np.testing.assert_equal(expected_motor_signals, np.array(predicted_motor_signals).squeeze())
np.testing.assert_(type(predicted_motor_signals) == list)
np.testing.assert_(len(np.array(predicted_motor_signals).shape) == 3)

# generating without pre-generated caching <- this approach should be the one implemented in the fit_session method
cache_motor_functions = __import__('prior_localization.pipelines.04_cache_motor', fromlist=('prior_localization.pipelines'))
regressors = cache_motor_functions.load_motor(eid)
motor_signals_of_interest = ['licking', 'whisking_l', 'whisking_r', 'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']
predicted_motor_signals = aggregate_on_timeWindow(regressors, kwargs, motor_signals_of_interest)

np.testing.assert_equal(expected_motor_signals, np.array(predicted_motor_signals).squeeze())
np.testing.assert_(type(predicted_motor_signals) == list)
np.testing.assert_(len(np.array(predicted_motor_signals).shape) == 3)


