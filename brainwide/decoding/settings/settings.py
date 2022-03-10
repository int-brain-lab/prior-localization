import logging
import numpy as np
from functions import utils as dut
import sklearn.linear_model as sklm
from datetime import date
from pathlib import Path
import warnings

logger = logging.getLogger('ibllib')
logger.disabled = True

strlut = {sklm.Lasso: "Lasso",
          sklm.LassoCV: "LassoCV",
          sklm.Ridge: "Ridge",
          sklm.RidgeCV: "RidgeCV",
          sklm.LinearRegression: "PureLinear",
          sklm.LogisticRegression: "Logistic"}

# %% Run param definitions
LOCAL = False
if LOCAL:
    DECODING_PATH = Path("/Users/csmfindling/Documents/Postdoc-Geneva/IBL/behavior/prior-localization/decoding/")
else:
    DECODING_PATH = Path("/home/users/f/findling/scratch/")

# aligned -> histology was performed by one experimenter
# resolved -> histology was performed by 2-3 experiments
SESS_CRITERION = 'aligned-behavior'  # aligned and behavior
DATE = str(date.today())
ALIGN_TIME = 'goCue_times'
TARGET = 'signcont'  # 'signcont' or 'pLeft'
CONTINUOUS_TARGET = False  # True  # is target continuous or not
# NB: if TARGET='signcont', MODEL with define how the neurometric curves will be generated. else MODEL computes TARGET
MODEL = dut.optimal_Bayesian  # expSmoothing_prevAction  or None # or dut.modeldispatcher.
TIME_WINDOW = (-0.6, -0.1)  # (0, 0.1)  #
ESTIMATOR = sklm.Lasso  # Must be in keys of strlut above
ESTIMATOR_KWARGS = {'tol': 0.0001, 'max_iter': 10000, 'fit_intercept': True}
N_PSEUDO = 100
N_PSEUDO_PER_JOB = 10
N_JOBS_PER_SESSION = N_PSEUDO // N_PSEUDO_PER_JOB
N_RUNS = 10
MIN_UNITS = 10
MIN_BEHAV_TRIAS = 400  # default BWM setting
MIN_RT = 0.08  # 0.08  # Float (s) or None
SINGLE_REGION = False  # perform decoding on region-wise or whole brain analysis
MERGED_PROBES = True  # merge probes before performing analysis
NO_UNBIAS = False  # take out unbiased trials
SHUFFLE = True  # interleaved cross validation
BORDER_QUANTILES_NEUROMETRIC = [.3, .7]  # [.3, .4, .5, .6, .7]
COMPUTE_NEUROMETRIC = True if TARGET == 'signcont' else False
FORCE_POSITIVE_NEURO_SLOPES = False

# Basically, quality metric on the stability of a single unit. Should have 1 metric per neuron
QC_CRITERIA = 3 / 3  # 3 / 3  # In {None, 1/3, 2/3, 3/3}
NORMALIZE_INPUT = False  # take out mean of the neural activity per unit across trials
NORMALIZE_OUTPUT = False  # take out mean of output to predict
if NORMALIZE_INPUT or NORMALIZE_OUTPUT:
    warnings.warn('This feature has not been tested')
USE_IMPOSTER_SESSION = False  # if false, it uses pseudosessions

BALANCED_WEIGHT = True  # seems to work better with BALANCED_WEIGHT=False, but putting True is important
USE_OPENTURNS = False  # uses openturns to perform kernel density estimation
BIN_SIZE_KDE = 0.05  # size of the kde bin
HPARAM_GRID = {'alpha': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
SAVE_BINNED = False  # Debugging parameter, not usually necessary
COMPUTE_NEURO_ON_EACH_FOLD = False  # if True, expect a script that is 5 times slower
ADD_TO_SAVING_PATH = 'pseudoSessions_mergedProbes_wholeBrain'


# session to be excluded (by Olivier Winter)
excludes = [
    'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',  # inconsistent trials object: relaunched task on 31-12-2021
    'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',  # same same
    '7a887357-850a-4378-bd2a-b5bc8bdd3aac',  # same same
    '56b57c38-2699-4091-90a8-aba35103155e',  # load object pickle error
    '09394481-8dd2-4d5c-9327-f2753ede92d7',  # same same
]

# ValueErrors and NotImplementedErrors
if not SINGLE_REGION and not MERGED_PROBES:
    raise ValueError('full probes analysis can only be done with merged probes')

if TARGET not in ['signcont', 'pLeft']:
    raise NotImplementedError('this TARGET is not supported yet')

if MODEL not in list(dut.modeldispatcher.keys()):
    raise NotImplementedError('this MODEL is not supported yet')

if COMPUTE_NEUROMETRIC and TARGET != 'signcont':
    raise ValueError('the target should be signcont to compute neurometric curves')

if len(BORDER_QUANTILES_NEUROMETRIC) == 0 and MODEL is not None:
    raise ValueError('BORDER_QUANTILES_NEUROMETRIC must be at least of 1 when MODEL is specified')

if len(BORDER_QUANTILES_NEUROMETRIC) != 0 and MODEL is None:
    raise ValueError('BORDER_QUANTILES_NEUROMETRIC must be empty when MODEL is not specified - oracle pLeft used')

fit_metadata = {
    'criterion': SESS_CRITERION,
    'target': TARGET,
    'model_type': dut.modeldispatcher[MODEL],
    'decoding_path': DECODING_PATH,
    'align_time': ALIGN_TIME,
    'time_window': TIME_WINDOW,
    'estimator': strlut[ESTIMATOR],
    'n_pseudo': N_PSEUDO,
    'min_units': MIN_UNITS,
    'min_behav_trials': MIN_BEHAV_TRIAS,
    'qc_criteria': QC_CRITERIA,
    'date': DATE,
    'shuffle': SHUFFLE,
    'no_unbias': NO_UNBIAS,
    'hyperparameter_grid': HPARAM_GRID,
    'save_binned': SAVE_BINNED,
    'balanced_weight': BALANCED_WEIGHT,
    'force_positive_neuro_slopes': FORCE_POSITIVE_NEURO_SLOPES,
    'compute_neurometric': COMPUTE_NEUROMETRIC,
    'n_runs': N_RUNS,
    'normalize_output': NORMALIZE_OUTPUT,
    'normalize_input': NORMALIZE_INPUT,
    'single_region': SINGLE_REGION,
    'use_imposter_session': USE_IMPOSTER_SESSION,
    'continuous_target': CONTINUOUS_TARGET,
    'use_openturns': USE_OPENTURNS,
    'bin_size_kde': BIN_SIZE_KDE
}
