import logging
import yaml
from pathlib import Path
import numpy as np
import sklearn.linear_model as sklm
from behavior_models.utils import build_path

logger = logging.getLogger('prior_localization')


def create_neural_path(output_path, date, neural_dtype, subject, session_id, probe,
                       region_str, target, time_window, pseudo_ids, add_to_path=None):
    full_path = Path(output_path).joinpath('neural', date, neural_dtype, subject, session_id, probe)
    full_path.mkdir(exist_ok=True, parents=True)

    config = check_config()

    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    time_str = f'{time_window[0]}_{time_window[1]}'.replace('.', '_')
    file_name = f'{region_str}_target_{target}_timeWindow_{time_str}_pseudo_id_{pseudo_str}'
    if add_to_path:
        for a in add_to_path:
            file_name = f'{file_name}_{a}_{config[a]}'
    return full_path.joinpath(f'{file_name}.pkl')


def check_bhv_fit_exists(subject, model, eids, resultpath, single_zeta):
    """
    Check if the fit for a given model exists for a given subject and session.

    Parameters
    ----------
    subject: str
        Subject nick name
    model: str
        Model class name
    eids: list
        List of session ids for sessions on which model was fitted
    resultpath: str
        Path to the results
    single_zeta: bool
        Whether or not the model was fitted with a single zeta

    Returns
    -------
    bool
        Whether or not the fit exists
    Path
        Path to the fit
    """
    path_results_mouse = f'model_{model}_single_zeta_{single_zeta}'
    trunc_eids = [eid.split('-')[0] for eid in eids]
    filen = build_path(path_results_mouse, trunc_eids)
    subjmodpath = Path(resultpath).joinpath(Path(subject))
    fullpath = subjmodpath.joinpath(filen)
    return fullpath.exists(), fullpath


def compute_mask(trials_df, align_time, time_window, min_len=None, max_len=None, no_unbias=False,
                 min_rt=0.08, max_rt=None, n_trials_crop_end=0):
    """Create a mask that denotes "good" trials which will be used for further analysis.

    Parameters
    ----------
    trials_df : dict
        contains relevant trial information like goCue_times, firstMovement_times, etc.
    align_time : str
        event in trial on which to align intervals
        'firstMovement_times' | 'stimOn_times' | 'feedback_times'
    time_window : tuple
        (window_start, window_end), relative to align_time
    min_len : float, optional
        minimum length of trials to keep (seconds), bypassed if trial_start column not in trials_df
    max_len : float, original
        maximum length of trials to keep (seconds), bypassed if trial_start column not in trials_df
    no_unbias : bool
        True to remove unbiased block trials, False to keep them
    min_rt : float
        minimum reaction time; trials with fast reactions will be removed
    n_trials_crop_end : int
        number of trials to crop from the end of the session

    Returns
    -------
    pd.Series
        boolean mask of good trials

    """

    # define reaction times
    if "react_times" not in trials_df.keys():
        trials_df["react_times"] = (
            trials_df.firstMovement_times - trials_df.stimOn_times
        )

    # successively build a mask that defines which trials we want to keep

    # ensure align event is not a nan
    mask = trials_df[align_time].notna()

    # ensure animal has moved
    mask = mask & trials_df.firstMovement_times.notna()

    # get rid of unbiased trials
    if no_unbias:
        mask = mask & (trials_df.probabilityLeft != 0.5).values

    # keep trials with reasonable reaction times
    if min_rt is not None:
        mask = mask & (~(trials_df.react_times < min_rt)).values
    if max_rt is not None:
        mask = mask & (~(trials_df.react_times > max_rt)).values

    if (
        "goCue_times" in trials_df.columns
        and max_len is not None
        and min_len is not None
    ):
        # get rid of trials that are too short or too long
        start_diffs = trials_df.goCue_times.diff()
        start_diffs.iloc[0] = 2
        mask = mask & ((start_diffs > min_len).values & (start_diffs < max_len).values)

        # get rid of trials with decoding windows that overlap following trial
        tmp = (
            trials_df[align_time].values[:-1] + time_window[1]
        ) < trials_df.trial_start.values[1:]
        tmp = np.concatenate([tmp, [True]])  # include final trial, no following trials
        mask = mask & tmp

    # get rid of trials where animal does not respond
    mask = mask & (trials_df.choice != 0)

    if n_trials_crop_end > 0:
        mask[-int(n_trials_crop_end):] = False

    return mask


def average_data_in_epoch(times, values, trials_df, align_event='stimOn_times', epoch=(-0.6, -0.1)):
    """
    Aggregate values in a given epoch relative to align_event for each trial. For trials for which the align_event
    is NaN or the epoch contains timestamps outside the times array, the value is set to NaN.

    Parameters
    ----------
    times: np.array
        Timestamps associated with values, assumed to be sorted
    values: np.array
        Data to be aggregated in epoch per trial, one value per timestamp
    trials_df: pd.DataFrame
        Dataframe with trials information
    align_event: str
        Event to align to, must be column in trials_df
    epoch: tuple
        Start and stop of time window to aggregate values in, relative to align_event in seconds


    Returns
    -------
    epoch_array: np.array
        Array of average values in epoch, one per trial in trials_df
    """

    # Make sure timestamps and values are arrays and of same size
    times = np.asarray(times)
    values = np.asarray(values)
    if not len(times) == len(values):
        raise ValueError(f'Inputs to times and values must be same length but are {len(times)} and {len(values)}')
    # Make sure times are sorted
    if not np.all(np.diff(times) >= 0):
        raise ValueError('Times must be sorted')
    # Get the events to align to and compute the ideal intervals for each trial
    events = trials_df[align_event].values
    intervals = np.c_[events + epoch[0], events + epoch[1]]
    # Make a mask to exclude trials were the event is nan, or interval starts before or ends after bin_times
    valid_trials = (~np.isnan(events)) & (intervals[:, 0] >= times[0]) & (intervals[:, 1] <= times[-1])
    # This is the first index to include to be sure to get values >= epoch[0]
    epoch_idx_start = np.searchsorted(times, intervals[valid_trials, 0], side='left')
    # This is the first index to exclude (NOT the last to include) to be sure to get values <= epoch[1]
    epoch_idx_stop = np.searchsorted(times, intervals[valid_trials, 1], side='right')
    # Create an array to fill in with the average epoch values for each trial
    epoch_array = np.full(events.shape, np.nan)
    epoch_array[valid_trials] = np.asarray([np.nanmean(values[start:stop]) for start, stop in
                                            zip(epoch_idx_start, epoch_idx_stop)], dtype=float)

    return epoch_array


def check_inputs(model, pseudo_ids, target, output_dir, config, logger):
    if output_dir is None:
        output_dir = Path.cwd()
        logger.info(f"No output directory specified, setting to current working directory {Path.cwd()}")

    pseudo_ids = [-1] if pseudo_ids is None else pseudo_ids
    if 0 in pseudo_ids:
        raise ValueError("pseudo id can only be -1 (None, actual session) or strictly greater than 0 (pseudo session)")
    if not np.all(np.sort(pseudo_ids) == pseudo_ids):
        raise ValueError("pseudo_ids must be sorted")

    if target in ['choice', 'feedback'] and model != 'actKernel':
        raise ValueError("If you want to decode choice or feedback, you must use the actionKernel model")

    if config['compute_neurometrics'] and target != "signcont":
        raise ValueError("The target should be signcont when compute_neurometrics is set to True in config file")

    if config['compute_neurometrics'] and len(config['border_quantiles_neurometrics']) == 0 and model != 'oracle':
        raise ValueError(
            "If compute_neurometrics is set to True in config file, and model is not oracle, "
            "border_quantiles_neurometrics must be a list of at least length 1"
        )

    if config['compute_neurometrics'] and len(config['border_quantiles_neurometrics']) != 0 and model == 'oracle':
        raise ValueError(
            "If compute_neurometrics is set to True in config file, and model is oracle, "
            "border_quantiles_neurometrics must be set to an empty list"
        )

    return pseudo_ids, output_dir


def check_config():
    # Get settings, need for some things
    with open(Path(__file__).parent.parent.joinpath('settings.yml'), "r") as settings_yml:
        settings = yaml.safe_load(settings_yml)
    # Get config
    with open(Path(__file__).parent.parent.joinpath('config.yml'), "r") as config_yml:
        config = yaml.safe_load(config_yml)
    # Estimator from scikit learn
    try:
        config['estimator'] = getattr(sklm, config['estimator'])
    except AttributeError as e:
        logger.error(f'The estimator {config["estimator"]} specified in config.yaml is not a function of scikit-learn'
                     f'linear_model.')
        raise e
    # Hyperparameter estimation
    config['use_native_sklearn_for_hyperparam_estimation'] = (config['estimator'] == sklm.Ridge)
    config['hparam_grid'] = ({"C": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
                             if config['estimator'] == sklm.LogisticRegression
                             else {"alpha": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])})

    # Add to path
    if settings['add_to_path'] is not None:
        config['add_to_path'] = {i: config[i] for i in settings['add_to_path']}

    return config
