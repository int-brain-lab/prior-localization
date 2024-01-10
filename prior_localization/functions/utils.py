import logging
import yaml
import pickle
from pathlib import Path
import numpy as np
import sklearn.linear_model as sklm
from behavior_models.utils import build_path

logger = logging.getLogger('prior_localization')


def check_bhv_fit_exists(subject, model, eids, resultpath, single_zeta):
    """
    Check if the fit for a given model exists for a given subject and session.

    Parameters
    ----------
    subject: str
        Subject nick name
    model: str
        Model class name
    eids: str or list
        session id or list of session ids for sessions on which model was fitted
    resultpath: str
        Path to the results

    Returns
    -------
    bool
        Whether or not the fit exists
    Path
        Path to the fit
    """
    if isinstance(eids, str):
        eids = [eids]
    fullpath = f'model_{model}'
    if single_zeta:
        fullpath += '_single_zeta'
    fullpath = build_path(fullpath, [eid.split('-')[0] for eid in eids])
    fullpath = Path(resultpath).joinpath(subject, fullpath)
    return fullpath.exists(), fullpath


def compute_mask(trials_df, align_event, min_rt=0.08, max_rt=None, n_trials_crop_end=0):
    """Create a mask that denotes "good" trials which will be used for further analysis.

    Parameters
    ----------
    trials_df : dict
        contains relevant trial information like goCue_times, firstMovement_times, etc.
    align_event : str
        event in trial on which to align intervals
        'firstMovement_times' | 'stimOn_times' | 'feedback_times'
    min_rt : float
        minimum reaction time; trials with faster reactions will be removed
    max_rt : float
        maximum reaction time; trials with slower reactions will be removed
    n_trials_crop_end : int
        number of trials to crop from the end of the session

    Returns
    -------
    pd.Series
        boolean mask of good trials
    """

    # define reaction times
    react_times = trials_df.firstMovement_times - trials_df.stimOn_times
    # successively build a mask that defines which trials we want to keep

    # ensure align event is not a nan
    mask = trials_df[align_event].notna()

    # ensure animal has moved
    mask = mask & trials_df.firstMovement_times.notna()

    # keep trials with reasonable reaction times
    if min_rt is not None:
        mask = mask & (~(react_times < min_rt)).values
    if max_rt is not None:
        mask = mask & (~(react_times > max_rt)).values

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
    epoch_array[valid_trials] = np.asarray(
        [np.nanmean(values[start:stop]) if ~np.all(np.isnan(values[start:stop])) else np.nan
         for start, stop in zip(epoch_idx_start, epoch_idx_stop)],
        dtype=float)

    return epoch_array


def check_inputs(
        model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics=None, motor_residuals=None
):
    """Perform some basic checks and/or corrections on inputs to the main decoding functions"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        try:
           output_dir.mkdir(parents=True)
           logger.info(f"Created output_dir: {output_dir}")
        except PermissionError:
            raise PermissionError(f"Following output_dir cannot be created, insufficient permissions: {output_dir}")

    pseudo_ids = [-1] if pseudo_ids is None else pseudo_ids
    if 0 in pseudo_ids:
        raise ValueError("pseudo id can only be -1 (None, actual session) or strictly greater than 0 (pseudo session)")
    if not np.all(np.sort(pseudo_ids) == pseudo_ids):
        raise ValueError("pseudo_ids must be sorted")

    if target in ['choice', 'feedback'] and model != 'actKernel':
        raise ValueError("If you want to decode choice or feedback, you must use the actKernel model")

    if compute_neurometrics and target != "signcont":
        raise ValueError("The target should be signcont when compute_neurometrics is set to True in config file")

    if compute_neurometrics and len(config['border_quantiles_neurometrics']) == 0 and model != 'oracle':
        raise ValueError(
            "If compute_neurometrics is set to True in config file, and model is not oracle, "
            "border_quantiles_neurometrics must be a list of at least length 1"
        )

    if compute_neurometrics and len(config['border_quantiles_neurometrics']) != 0 and model == 'oracle':
        raise ValueError(
            "If compute_neurometrics is set to True in config file, and model is oracle, "
            "border_quantiles_neurometrics must be set to an empty list"
        )

    if motor_residuals and model != 'optBay':
        raise ValueError('Motor residuals can only be computed for optBay model')

    return pseudo_ids, output_dir


def check_config():
    """Load config yaml and perform some basic checks"""
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
    if config['estimator'] == sklm.LogisticRegression:
        config['estimator_kwargs'] = {**config['estimator_kwargs'], 'penalty': 'l1', 'solver': 'liblinear'}
    # Hyperparameter estimation
    config['use_native_sklearn_for_hyperparam_estimation'] = (config['estimator'] == sklm.Ridge)
    config['hparam_grid'] = ({"C": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
                             if config['estimator'] == sklm.LogisticRegression
                             else {"alpha": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])})

    return config


# Copied from from https://github.com/cskrasniak/wfield/blob/master/wfield/analyses.py
def downsample_atlas(atlas, pixelSize=20, mask=None):
    """
    Downsamples the atlas so that it can be matching to the downsampled images. if mask is not provided
    then just the atlas is used. pixelSize must be a common divisor of 540 and 640
    """
    if not mask:
        mask = atlas != 0
    downsampled_atlas = np.zeros((int(atlas.shape[0] / pixelSize), int(atlas.shape[1] / pixelSize)))
    for top in np.arange(0, 540, pixelSize):
        for left in np.arange(0, 640, pixelSize):
            useArea = (np.array([np.arange(top, top + pixelSize)] * pixelSize).flatten(),
                       np.array([[x] * pixelSize for x in range(left, left + pixelSize)]).flatten())
            u_areas, u_counts = np.unique(atlas[useArea], return_counts=True)
            if np.sum(mask[useArea] != 0) < .5:
                # if more than half of the pixels are outside of the brain, skip this group of pixels
                continue
            else:
                spot_label = u_areas[np.argmax(u_counts)]
                downsampled_atlas[int(top / pixelSize), int(left / pixelSize)] = spot_label
    return downsampled_atlas.astype(int)


def spatial_down_sample(stack, pixelSize=20):
    """
    Downsamples the whole df/f video for a session to a manageable size, best are to do a 10x or
    20x downsampling, this makes many tasks more manageable on a desktop.
    """
    mask = stack.U_warped != 0
    mask = mask.mean(axis=2)
    try:
        downsampled_im = np.zeros((stack.SVT.shape[1],
                                   int(stack.U_warped.shape[0] / pixelSize),
                                   int(stack.U_warped.shape[1] / pixelSize)))
    except:
        print('Choose a downsampling amount that is a common divisor of 540 and 640')
    for top in np.arange(0, 540, pixelSize):
        for left in np.arange(0, 640, pixelSize):
            useArea = (np.array([np.arange(top, top + pixelSize)] * pixelSize).flatten(),
                       np.array([[x] * pixelSize for x in range(left, left + pixelSize)]).flatten())
            if np.sum(mask[useArea] != 0) < .5:
                # if more than half of the pixels are outside of the brain, skip this group of pixels
                continue
            else:
                spot_activity = stack.get_timecourse(useArea).mean(axis=0)
                downsampled_im[:, int(top / pixelSize), int(left / pixelSize)] = spot_activity
    return downsampled_im


def subtract_motor_residuals(motor_signals, all_targets, trials_mask):
    """Subtract predictions based on motor signal from predictions as residuals from the behavioural targets"""
    # Update trials mask with possible nans from motor signal
    trials_mask = trials_mask & ~np.any(np.isnan(motor_signals), axis=1)
    # Compute motor predictions and subtract them from targets
    new_targets = []
    for target_data in all_targets:
        clf = sklm.RidgeCV(alphas=[1e-3, 1e-2, 1e-1]).fit(motor_signals[trials_mask], target_data[trials_mask])
        motor = np.full_like(trials_mask, np.nan)
        motor[trials_mask] = clf.predict(motor_signals[trials_mask])
        new_targets.append(target_data - motor)

    return new_targets, trials_mask
