import hashlib
import logging
import yaml
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
import sklearn.linear_model as sklm
from behavior_models.utils import build_path
from iblutil.numerical import bincount2D


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
                             else {"alpha": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])})

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
    for set_targets in all_targets:
        new_set = []
        for target_data in set_targets:
            clf = sklm.RidgeCV(alphas=[1e-3, 1e-2, 1e-1]).fit(motor_signals[trials_mask], target_data[trials_mask])
            motor = np.full_like(trials_mask, np.nan)
            motor[trials_mask] = clf.predict(motor_signals[trials_mask])
            new_set.append(target_data - motor)
        new_targets.append(new_set)

    return new_targets, trials_mask


def format_data_for_decoding(ys, Xs):
    """Transform target data into standard format: list of np.ndarrays.

    Parameters
    ----------
    ys : np.ndarray or list or pandas.Series
        targets
    Xs : np.ndarray or list
        regressors

    Returns
    -------
    tuple
        - (list of np.ndarray) formatted ys
        - (list of np.ndarray) formatted Xs

    """

    # tranform targets
    if isinstance(ys, np.ndarray):
        # input is single numpy array
        ys = [np.array([y]) for y in ys]
    elif isinstance(ys, list) and ys[0].shape == ():
        # input is list of float instead of list of np.ndarrays
        ys = [np.array([y]) for y in ys]
    elif isinstance(ys, pd.Series):
        # input is a pandas Series
        ys = ys.to_numpy()
        ys = [np.array([y]) for y in ys]

    # transform regressors
    if isinstance(Xs, np.ndarray):
        Xs = [x[None, :] for x in Xs]

    return ys, Xs


def logisticreg_criteria(array, min_unique_counts=3):
    """Check array contains two classes, and that a minimum number of examples exist per class.

    Parameters
    ----------
    array : array-like
        array of ints
    min_unique_counts : int, optional
        minimum number of examples required for each class

    Returns
    -------
    bool

    """
    array = format_data_for_decoding(array, [])[0]
    y_uniquecounts = np.unique(array, return_counts=True)[1]
    return len(y_uniquecounts) == 2 and np.min(y_uniquecounts) >= min_unique_counts


def get_spike_data_per_trial(times, clusters, intervals, binsize):
    """Select spiking data for specified interval on each trial.

    Parameters
    ----------
    times : array-like
        time in seconds for each spike
    clusters : array-like
        cluster id for each spike
    intervals : np.array
        shape (n_trials, 2) where columns indicate interval onset/offset (seconds)
    binsize : float
        width of each bin in seconds

    Returns
    -------
    tuple
        - (list): time in seconds for each trial; timepoints refer to the start/left edge of a bin
        - (list): data for each trial of shape (n_clusters, n_bins)

    """

    interval_begs = intervals[:, 0]
    interval_ends = intervals[:, 1]
    interval_len = interval_ends - interval_begs
    n_trials = intervals.shape[0]

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.median(np.ceil(interval_len / binsize).astype('int')))

    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    binned_spikes = np.zeros((n_trials, n_clusters_in_region, n_bins))
    spike_times_list = []
    for tr, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
        # just get spikes for this region/trial
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            binned_spikes_tmp = np.zeros((n_clusters_in_region, n_bins))
            if np.isnan(t_beg) or np.isnan(t_end):
                t_idxs = np.nan * np.ones(n_bins)
            else:
                t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            binned_spikes_tmp, t_idxs, cluster_idxs = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)

        # update data block
        binned_spikes[tr, idxs_tmp, :] += binned_spikes_tmp[:, :n_bins]
        spike_times_list.append(t_idxs[:n_bins])

    return binned_spikes, spike_times_list


def build_lagged_predictor_matrix(array, n_lags, return_valid=True):
    """Build predictor matrix with time-lagged datapoints.

    Parameters
    ----------
    array : np.ndarray
        shape (n_time, n_clusters)
    n_lags : int
        number of lagged timepoints (includes zero lag)
    return_valid : bool, optional
        True to crop first n_lags rows, False to leave all

    Returns
    -------
    np.ndarray
        shape (n_time - n_lags, n_clusters * (n_lags + 1)) if return_valid==True
        shape (n_time, n_clusters * (n_lags + 1)) if return_valid==False

    """
    if n_lags < 0:
        raise ValueError('`n_lags` must be >=0, not {}'.format(n_lags))
    mat = np.hstack([np.roll(array, i, axis=0) for i in range(n_lags + 1)])
    if return_valid:
        mat = mat[n_lags:]
    return mat


def str2int(string, digits=8):
    return int(hashlib.sha1(string.encode('utf-8')).hexdigest(), 16) % (10 ** digits)
