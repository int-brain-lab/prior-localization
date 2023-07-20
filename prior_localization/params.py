import numpy as np
import sklearn.linear_model as sklm

# These are settings
NEURAL_DTYPE = "ephys"  # 'ephys' or 'widefield'
DATE = "30-01-2023"

# These might be params
ESTIMATOR = sklm.Ridge
USE_NATIVE_SKLEARN_FOR_HYPERPARAMETER_ESTIMATION = (ESTIMATOR == sklm.Ridge)
BINARIZATION_VALUE = None  # to binarize the target -> could be useful with logistic regression estimator
ESTIMATOR_KWARGS = {"tol": 0.0001, "max_iter": 20000, "fit_intercept": True}
N_PSEUDO = 200
N_PSEUDO_PER_JOB = 10
N_JOBS_PER_SESSION = N_PSEUDO // N_PSEUDO_PER_JOB
N_RUNS = 2
MIN_UNITS = 10
MERGED_PROBES = False  # merge probes before performing analysis
SHUFFLE = True  # interleaved cross validation
BORDER_QUANTILES_NEUROMETRIC = [0.3, 0.7]  # [.3, .4, .5, .6, .7]
COMPUTE_NEURO_ON_EACH_FOLD = False  # if True, expect a script that is 5 times slower
SAVE_PREDICTIONS = True
QC_CRITERIA = 3 / 3  # 3 / 3  # In {None, 1/3, 2/3, 3/3}
BALANCED_WEIGHT = True  # seems to work better with BALANCED_WEIGHT=False, but putting True is important
HPARAM_GRID = (
    {"alpha": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, ])}
    # 'alpha': np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
    # lasso , 0.01, 0.1
    if not (sklm.LogisticRegression == ESTIMATOR)
    else {"C": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
)
ADD_TO_PATH = {'balanced_weighting': BALANCED_WEIGHT}

# WIDE FIELD IMAGING
# WFI_NB_FRAMES_START = -2  # left signed number of frames from ALIGN_TIME (frame included)
# WFI_NB_FRAMES_END = -2  # right signed number of frames from ALIGN_TIME (frame included). If 0 align time frame included
WFI_AVERAGE_OVER_FRAMES = False
# if NEURAL_DTYPE == "widefield" and WFI_NB_FRAMES_START > WFI_NB_FRAMES_END:
#     raise ValueError("there is a problem in the specification of the timing of the widefield")


# if TARGET in ["choice", "feedback"] and (
#     MODEL != 'actKernel'
# ):
#     raise ValueError(
#         "if you want to decode choice or feedback, you must use the actionKernel model and frankenstein sessions"
#     )
#
# if COMPUTE_NEUROMETRIC and TARGET != "signcont":
#     raise ValueError("the target should be signcont to compute neurometric curves")
#
# if len(BORDER_QUANTILES_NEUROMETRIC) == 0 and MODEL is not None:
#     raise ValueError(
#         "BORDER_QUANTILES_NEUROMETRIC must be at least of 1 when MODEL is specified"
#     )
#
# if len(BORDER_QUANTILES_NEUROMETRIC) != 0 and MODEL is None:
#     raise ValueError(
#         "BORDER_QUANTILES_NEUROMETRIC must be empty when MODEL is not specified - oracle pLeft used"
#     )

# def check_settings(settings):
#     """Error check on pipeline settings.
#
#     Parameters
#     ----------
#     settings : dict
#
#     Returns
#     -------
#     dict
#
#     """
#
#     from behavior_models.models import ActionKernel
#     from prior_localization.functions.behavior_targets import optimal_Bayesian
#
#     # options for decoding targets
#     target_options_singlebin = [
#         "prior",  # some estimate of the block prior
#         "choice",  # subject's choice (L/R)
#         "feedback",  # correct/incorrect
#         "signcont",  # signed contrast of stimulus
#     ]
#     target_options_multibin = [
#         "wheel-vel",
#         "wheel-speed",
#         "pupil",
#         "l-paw-pos",
#         "l-paw-vel",
#         "l-paw-speed",
#         "l-whisker-me",
#         "r-paw-pos",
#         "r-paw-vel",
#         "r-paw-speed",
#         "r-whisker-me",
#     ]
#
#     # options for behavioral models
#     behavior_model_options = {
#         "ActionKernel": ActionKernel,
#         "optimal_Bayesian": optimal_Bayesian,
#         "oracle": None,
#     }

    # options for align events
    # align_event_options = [
    #     "firstMovement_times",
    #     "goCue_times",
    #     "stimOn_times",
    #     "feedback_times",
    # ]
    #
    # # options for decoder
    # decoder_options = {
    #     "linear": sklm.LinearRegression,
    #     "lasso": sklm.Lasso,
    #     "ridge": sklm.Ridge,
    #     "logistic": sklm.LogisticRegression,
    # }
    #
    #
    #
    # if params["target"] not in target_options_singlebin + target_options_multibin:
    #     raise NotImplementedError(
    #         "provided target option '{}' invalid; must be in {}".format(
    #             params["target"], target_options_singlebin + target_options_multibin
    #         )
    #     )
    #
    # if params["model"] not in behavior_model_options.keys():
    #     raise NotImplementedError(
    #         "provided beh model option '{}' invalid; must be in {}".format(
    #             params["model"], behavior_model_options.keys()
    #         )
    #     )
    #
    # if params["align_time"] not in align_event_options:
    #     raise NotImplementedError(
    #         "provided align event '{}' invalid; must be in {}".format(
    #             params["align_time"], align_event_options
    #         )
    #     )
    #
    # if not params["single_region"] and not params["merge_probes"]:
    #     raise ValueError("full probes analysis can only be done with merged probes")
    #
    # if params["compute_neurometric"] and kwargs["target"] != "signcont":
    #     raise ValueError("the target should be signcont to compute neurometric curves")
    #
    # if len(params["border_quantiles_neurometric"]) == 0 and params["model"] != "oracle":
    #     raise ValueError(
    #         "border_quantiles_neurometric must be at least of 1 when behavior model is specified"
    #     )
    #
    # if len(params["border_quantiles_neurometric"]) != 0 and params["model"] == "oracle":
    #     raise ValueError(
    #         f"border_quantiles_neurometric must be empty when behavior model is not specified"
    #         f"- oracle pLeft used"
    #     )
    #
    # # map behavior model string to model class
    # if params["model"] == "logistic" and params["balanced_continuous_target"]:
    #     raise ValueError(
    #         "you can not have a continuous target with logistic regression"
    #     )
    #
    # params["model"] = behavior_model_options[params["model"]]
    #
    # # map estimator string to sklm class
    # if params["estimator"] == "logistic":
    #     params["hyperparam_grid"] = {"C": params["hyperparam_grid"]["C"]}
    # else:
    #     params["hyperparam_grid"] = {"alpha": params["hyperparam_grid"]["alpha"]}
    # params["estimator"] = decoder_options[params["estimator"]]