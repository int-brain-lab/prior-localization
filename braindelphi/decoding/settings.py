import logging
import numpy as np
from braindelphi.decoding.functions.process_targets import optimal_Bayesian
from behavior_models.models.expSmoothing_prevAction import expSmoothing_prevAction
from behavior_models.models.expSmoothing_stimside import expSmoothing_stimside
from braindelphi.params import FIT_PATH as NEURAL_MOD_PATH
from braindelphi.params import BEH_MOD_PATH as BEHAVIOR_MOD_PATH
import sklearn.linear_model as sklm
import warnings

logger = logging.getLogger('ibllib')
logger.disabled = True

strlut = {
    sklm.Lasso: "Lasso",
    sklm.LassoCV: "LassoCV",
    sklm.Ridge: "Ridge",
    sklm.RidgeCV: "RidgeCV",
    sklm.LinearRegression: "PureLinear",
    sklm.LogisticRegression: "Logistic"
}

NEURAL_DTYPE = 'ephys'  #  'ephys' or 'widefield'

# aligned -> histology was performed by one experimenter
# resolved -> histology was performed by 2-3 experiments
SESS_CRITERION = 'aligned-behavior'  # aligned and behavior
ALIGN_TIME = 'goCue_times'
TARGET = 'pLeft'  # 'signcont' or 'pLeft'
if TARGET not in ['pLeft', 'signcont', 'choice', 'feedback']:
    raise ValueError('TARGET can only be pLeft, signcont, choice or feedback')
# NB: if TARGET='signcont', MODEL with define how the neurometric curves will be generated. else MODEL computes TARGET
MODEL = expSmoothing_prevAction  # expSmoothing_prevAction, optimal_Bayesian or None(=Oracle)
BEH_MOUSELEVEL_TRAINING = False  # if True, trains the behavioral model session-wise else mouse-wise
TIME_WINDOW = (-0.6, -0.1)  # (0, 0.1)  #
ESTIMATOR = sklm.Lasso  # Must be in keys of strlut above
BINARIZATION_VALUE = None  # to binarize the target -> could be useful with logistic regression estimator
ESTIMATOR_KWARGS = {'tol': 0.0001, 'max_iter': 10000, 'fit_intercept': True}
N_PSEUDO = 200
N_PSEUDO_PER_JOB = 50
N_JOBS_PER_SESSION = N_PSEUDO // N_PSEUDO_PER_JOB
N_RUNS = 10
MIN_UNITS = 10
MIN_BEHAV_TRIAS = 400  # default BWM setting
MIN_RT = 0.08  # 0.08  # Float (s) or None
MIN_LEN = 1 # min length of trial
MAX_LEN = 5 # max length of trial
SINGLE_REGION = True  # perform decoding on region-wise or whole brain analysis
MERGED_PROBES = False  # merge probes before performing analysis
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
USE_IMPOSTER_SESSION = False  # if false, it uses pseudosessions and simulates the model when action are necessary
CONSTRAIN_IMPOSTER_SESSION_WITH_BEH = False
USE_IMPOSTER_SESSION_FOR_BALANCING = False  # if false, it simulates the model (should be False)
SIMULATE_NEURAL_DATA = False

BALANCED_WEIGHT = False  # seems to work better with BALANCED_WEIGHT=False, but putting True is important
BALANCED_CONTINUOUS_TARGET = True  # is target continuous or discrete FOR BALANCED WEIGHTING
USE_OPENTURNS = False  # uses openturns to perform kernel density estimation
BIN_SIZE_KDE = 0.05  # size of the kde bin
HPARAM_GRID = ({
    'alpha': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
} if not (sklm.LogisticRegression == ESTIMATOR) else {
    'C': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
})
SAVE_BINNED = False  # Debugging parameter, not usually necessary
COMPUTE_NEURO_ON_EACH_FOLD = False  # if True, expect a script that is 5 times slower
ADD_TO_SAVING_PATH = (
    'imposterSess_%i_balancedWeight_%i_RegionLevel_%i_mergedProbes_%i_behMouseLevelTraining_%i_simulated_%i_constrainImpSess_%i'
    % (USE_IMPOSTER_SESSION, BALANCED_WEIGHT, SINGLE_REGION, MERGED_PROBES,
       BEH_MOUSELEVEL_TRAINING, SIMULATE_NEURAL_DATA, CONSTRAIN_IMPOSTER_SESSION_WITH_BEH))

# WIDE FIELD IMAGING
WFI_HEMISPHERES = ['left']  # 'left' and/or 'right'
WFI_NB_FRAMES = -1  # signed number of frames from ALIGN_TIME. can not be zero

# session to be excluded (by Olivier Winter)
excludes = [
    'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',  # inconsistent trials object: relaunched task on 31-12-2021
    'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',  # same same
    '7a887357-850a-4378-bd2a-b5bc8bdd3aac',  # same same
    '56b57c38-2699-4091-90a8-aba35103155e',  # load object pickle error
    '09394481-8dd2-4d5c-9327-f2753ede92d7',  # same same
]

modeldispatcher = {
    expSmoothing_prevAction: expSmoothing_prevAction.name,
    expSmoothing_stimside: expSmoothing_stimside.name,
    optimal_Bayesian: 'optBay',
    None: 'oracle'
}

if ESTIMATOR == sklm.LogisticRegression and BALANCED_CONTINUOUS_TARGET:
    raise ValueError('you can not have a continuous target with logistic regression')

# ValueErrors and NotImplementedErrors
if not SINGLE_REGION and not MERGED_PROBES:
    raise ValueError('full probes analysis can only be done with merged probes')

if MODEL not in list(modeldispatcher.keys()):
    raise NotImplementedError('this MODEL is not supported yet')

if COMPUTE_NEUROMETRIC and TARGET != 'signcont':
    raise ValueError('the target should be signcont to compute neurometric curves')

if len(BORDER_QUANTILES_NEUROMETRIC) == 0 and MODEL is not None:
    raise ValueError('BORDER_QUANTILES_NEUROMETRIC must be at least of 1 when MODEL is specified')

if len(BORDER_QUANTILES_NEUROMETRIC) != 0 and MODEL is None:
    raise ValueError(
        'BORDER_QUANTILES_NEUROMETRIC must be empty when MODEL is not specified - oracle pLeft used'
    )


fit_metadata = {
    'criterion': SESS_CRITERION,
    'target': TARGET,
    'model_type': modeldispatcher[MODEL],
    'align_time': ALIGN_TIME,
    'time_window': TIME_WINDOW,
    'estimator': ESTIMATOR,
    'nb_runs': N_RUNS,
    'n_pseudo': N_PSEUDO,
    'min_units': MIN_UNITS,
    'min_behav_trials': MIN_BEHAV_TRIAS,
    'min_rt': MIN_RT,
    'qc_criteria': QC_CRITERIA,
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
    'balanced_continuous_target': BALANCED_CONTINUOUS_TARGET,
    'use_openturns': USE_OPENTURNS,
    'bin_size_kde': BIN_SIZE_KDE,
    'use_imposter_session_for_balancing': USE_IMPOSTER_SESSION_FOR_BALANCING,
    'beh_mouseLevel_training': BEH_MOUSELEVEL_TRAINING,
    'simulate_neural_data': SIMULATE_NEURAL_DATA,
    'constrain_imposter_session_with_beh': CONSTRAIN_IMPOSTER_SESSION_WITH_BEH,
    'neural_dtype': NEURAL_DTYPE,
    'modeldispatcher': modeldispatcher,
    'behfit_path': BEHAVIOR_MOD_PATH,
    'neuralfit_path': NEURAL_MOD_PATH,
    'estimator_kwargs': ESTIMATOR_KWARGS,
    'hyperparam_grid': HPARAM_GRID,
    'add_to_saving_path': ADD_TO_SAVING_PATH,
}

if NEURAL_DTYPE == 'widefield':
    fit_metadata['wfi_hemispheres'] = WFI_HEMISPHERES
    fit_metadata['wfi_nb_frames'] = WFI_HEMISPHERES

kwargs = {
    'nb_runs': N_RUNS,
    'single_region': SINGLE_REGION,
    'merged_probes': MERGED_PROBES,
    'neuralfit_path': NEURAL_MOD_PATH,
    'behfit_path': BEHAVIOR_MOD_PATH,
    'modeldispatcher': modeldispatcher,
    'estimator_kwargs': ESTIMATOR_KWARGS,
    'hyperparam_grid': HPARAM_GRID,
    'save_binned': SAVE_BINNED,
    'shuffle': SHUFFLE,
    'balanced_weight': BALANCED_WEIGHT,
    'normalize_input': NORMALIZE_INPUT,
    'normalize_output': NORMALIZE_OUTPUT,
    'compute_on_each_fold': COMPUTE_NEURO_ON_EACH_FOLD,
    'balanced_continuous_target': BALANCED_CONTINUOUS_TARGET,
    'force_positive_neuro_slopes': FORCE_POSITIVE_NEURO_SLOPES,
    'estimator': ESTIMATOR,
    'target': TARGET,
    'model': MODEL,
    'align_time': ALIGN_TIME,
    'no_unbias': NO_UNBIAS,
    'min_rt': MIN_RT,
    'min_behav_trials': MIN_BEHAV_TRIAS,
    'qc_criteria': QC_CRITERIA,
    'min_units': MIN_UNITS,
    'time_window': TIME_WINDOW,
    'use_imposter_session': USE_IMPOSTER_SESSION,
    'compute_neurometric': COMPUTE_NEUROMETRIC,
    'border_quantiles_neurometric': BORDER_QUANTILES_NEUROMETRIC,
    'add_to_saving_path': ADD_TO_SAVING_PATH,
    'use_openturns': USE_OPENTURNS,
    'bin_size_kde': BIN_SIZE_KDE,
    'neural_dtype': NEURAL_DTYPE,
    'wfi_hemispheres': WFI_HEMISPHERES,
    'wfi_nb_frames': WFI_NB_FRAMES,
    'use_imposter_session_for_balancing': USE_IMPOSTER_SESSION_FOR_BALANCING,
    'beh_mouseLevel_training': BEH_MOUSELEVEL_TRAINING,
    'binarization_value': BINARIZATION_VALUE,
    'simulate_neural_data': SIMULATE_NEURAL_DATA,
    'constrain_imposter_session_with_beh': CONSTRAIN_IMPOSTER_SESSION_WITH_BEH,
    'min_len': MIN_LEN,
    'max_len': MAX_LEN,
}