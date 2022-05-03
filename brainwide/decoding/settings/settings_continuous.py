"""Parameter settings for decoding continuous values in multiple bins per trial."""

from datetime import date
import logging
import numpy as np
from pathlib import Path
import sklearn.linear_model as sklm

from functions import utils as dut

logger = logging.getLogger('ibllib')
logger.disabled = True


# options for decoding target
TARGET_OPTIONS = [
    'wheel-vel',
    'wheel-speed',
    'pupil',
    'l-paw-pos',
    'l-paw-vel',
    'l-paw-speed',
    'l-whisker-me',
    'r-paw-pos',
    'r-paw-vel',
    'r-paw-speed',
    'r-whisker-me',
]

# options for decoder
DECODER_OPTIONS = [
    sklm.Lasso,
    # sklm.LassoCV: "LassoCV",
    sklm.Ridge,
    # sklm.RidgeCV: "RidgeCV",
    sklm.LinearRegression,
    # sklm.LogisticRegression: "Logistic",
]

# -------------------------------------------------------------------------------------------------
# IO params
# -------------------------------------------------------------------------------------------------
LOCAL = True  # run locally or on remote machine
DECODING_PATH = Path("/media/mattw/ibl/decoding")
DATE = '2022-05-03'  # str(date.today())

# -------------------------------------------------------------------------------------------------
# data processing params
# -------------------------------------------------------------------------------------------------

TARGET = 'wheel-speed'

# time window params
ALIGN_TIME = 'firstMovement_times'  # firstMovement_times | stimOn_times | feedback_times
TIME_WINDOW = (-0.25, 1.0)  # sec; relative to ALIGN_TIME
BINSIZE = 0.020  # sec; size of individual bins within time window
N_BINS_LAG = 10  # number of bins to use for prediction

# decoder params
ESTIMATOR = sklm.Ridge
# ESTIMATOR_KWARGS = {'tol': 0.0001, 'max_iter': 10000, 'fit_intercept': True}
HPARAM_GRID = {'alpha': np.array([1, 10, 100, 1000, 10000])}
N_PSEUDO = 0
N_RUNS = 5
SHUFFLE = True  # interleaved cross validation
USE_IMPOSTER_SESSION = False  # if false, it uses pseudosessions
# NORMALIZE_INPUT = False  # take out mean of the neural activity per unit across trials
# NORMALIZE_OUTPUT = False  # take out mean of output to predict

# cluster params
MIN_UNITS = 10
QC_CRITERIA = 3 / 3  # 3 / 3  # In {None, 1/3, 2/3, 3/3}
SINGLE_REGION = True  # perform decoding on region-wise or whole brain analysis
MERGED_PROBES = False  # merge probes before performing analysis

# session/behavior params
# aligned -> histology was performed by one experimenter
# resolved -> histology was performed by 2-3 experiments
SESS_CRITERION = 'aligned-behavior'  # aligned and behavior
MIN_BEHAV_TRIAS = 300  # default BWM setting
MIN_RT = 0.08  # 0.08  # Float (s) or None
NO_UNBIAS = False  # take out unbiased trials


# -------------------------------------------------------------------------------------------------
# widefield params
# -------------------------------------------------------------------------------------------------
WIDE_FIELD_IMAGING = False
WFI_HEMISPHERES = ['left']  # 'left' and/or 'right'
WFI_NB_FRAMES = -1  # signed number of frames from ALIGN_TIME. can not be zero


# -------------------------------------------------------------------------------------------------
# save all settings in a dict
# -------------------------------------------------------------------------------------------------

# Error checking
if not LOCAL:
    raise NotImplementedError

if not SINGLE_REGION and not MERGED_PROBES:
    raise ValueError('full probes analysis can only be done with merged probes')

if ESTIMATOR not in DECODER_OPTIONS:
    raise ValueError('ESTIMATOR can only be one of {}'.format(DECODER_OPTIONS))

if TARGET not in TARGET_OPTIONS:
    raise ValueError('TARGET can only be on of {}'.format(TARGET_OPTIONS))

fit_metadata = {
    'target': TARGET,
    'align_time': ALIGN_TIME,
    'time_window': TIME_WINDOW,
    'binsize': BINSIZE,
    'n_bins_lag': N_BINS_LAG,

    'estimator': ESTIMATOR,
    'hyperparameter_grid': HPARAM_GRID,
    'n_pseudo': N_PSEUDO,
    'n_runs': N_RUNS,
    'shuffle': SHUFFLE,
    'use_imposter_session': USE_IMPOSTER_SESSION,
    # 'normalize_input': NORMALIZE_INPUT,
    # 'normalize_output': NORMALIZE_OUTPUT,

    'min_units': MIN_UNITS,
    'qc_criteria': QC_CRITERIA,
    'single_region': SINGLE_REGION,
    'merged_probes': MERGED_PROBES,

    'criterion': SESS_CRITERION,
    'min_behav_trials': MIN_BEHAV_TRIAS,
    'min_rt': MIN_RT,
    'no_unbias': NO_UNBIAS,

    'wide_field_imaging': WIDE_FIELD_IMAGING,

    'today': DATE,
    'output_path': DECODING_PATH,
    'add_to_saving_path': '_binsize=%i_lags=%i_mergedProbes_%i' % \
                          (1000 * BINSIZE, N_BINS_LAG, MERGED_PROBES),
}

if WIDE_FIELD_IMAGING:
    fit_metadata['wfi_hemispheres'] = WFI_HEMISPHERES
    fit_metadata['wfi_nb_frames'] = WFI_NB_FRAMES
