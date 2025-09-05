import numpy as np
from scipy.interpolate import interp1d
import torch

from iblutil.util import setup_logger
from behavior_models.utils import format_data, format_input
from behavior_models.models import ActionKernel, StimulusKernel

from prior_localization.functions.utils import check_bhv_fit_exists, check_config

logger = setup_logger()


def optimal_Bayesian(act, side):
    """
    Generates the optimal prior
    Params:
        act (array of shape [nb_sessions, nb_trials]): action performed by the mice of shape
        side (array of shape [nb_sessions, nb_trials]): stimulus side (-1 (right), 1 (left)) observed by the mice
    Output:
        prior (array of shape [nb_sessions, nb_chains, nb_trials]): prior for each chain and session
    """
    act = torch.from_numpy(act)
    side = torch.from_numpy(side)
    lb, tau, ub, gamma = 20, 60, 100, 0.8
    nb_blocklengths = 100
    nb_typeblocks = 3
    eps = torch.tensor(1e-15)

    alpha = torch.zeros([act.shape[-1], nb_blocklengths, nb_typeblocks])
    alpha[0, 0, 1] = 1
    alpha = alpha.reshape(-1, nb_blocklengths * nb_typeblocks)
    h = torch.zeros([nb_typeblocks * nb_blocklengths])

    # build transition matrix
    b = torch.zeros([nb_blocklengths, nb_typeblocks, nb_typeblocks])
    b[1:][:, 0, 0], b[1:][:, 1, 1], b[1:][:, 2, 2] = 1, 1, 1  # case when l_t > 0
    b[0][0][-1], b[0][-1][0], b[0][1][np.array([0, 2])] = 1, 1, 1. / 2  # case when l_t = 1
    n = torch.arange(1, nb_blocklengths + 1)
    ref = torch.exp(-n / tau) * (lb <= n) * (ub >= n)
    torch.flip(ref.double(), (0,))
    hazard = torch.cummax(
        ref / torch.flip(torch.cumsum(torch.flip(ref.double(), (0,)), 0) + eps, (0,)), 0)[0]
    l_mat = torch.cat(
        (torch.unsqueeze(hazard, -1),
         torch.cat((torch.diag(1 - hazard[:-1]), torch.zeros(nb_blocklengths - 1)[None]), axis=0)),
        axis=-1)  # l_{t-1}, l_t
    transition = eps + torch.transpose(l_mat[:, :, None, None] * b[None], 1, 2).reshape(
        nb_typeblocks * nb_blocklengths, -1)

    # likelihood
    lks = torch.hstack([
        gamma * (side[:, None] == -1) + (1 - gamma) * (side[:, None] == 1),
        torch.ones_like(act[:, None]) * 1. / 2,
        gamma * (side[:, None] == 1) + (1 - gamma) * (side[:, None] == -1)
    ])
    to_update = torch.unsqueeze(torch.unsqueeze(act.not_equal(0), -1), -1) * 1

    for i_trial in range(act.shape[-1]):
        # save priors
        if i_trial >= 0:
            if i_trial > 0:
                alpha[i_trial] = (torch.sum(torch.unsqueeze(h, -1) * transition, axis=0) * to_update[i_trial - 1]
                                  + alpha[i_trial - 1] * (1 - to_update[i_trial - 1]))
            # else:
            #    alpha = alpha.reshape(-1, nb_blocklengths, nb_typeblocks)
            #    alpha[i_trial, 0, 0] = 0.5
            #    alpha[i_trial, 0, -1] = 0.5
            #    alpha = alpha.reshape(-1, nb_blocklengths * nb_typeblocks)
            h = alpha[i_trial] * lks[i_trial].repeat(nb_blocklengths)
            h = h / torch.unsqueeze(torch.sum(h, axis=-1), -1)
        else:
            if i_trial > 0:
                alpha[i_trial, :] = alpha[i_trial - 1, :]

    predictive = torch.sum(alpha.reshape(-1, nb_blocklengths, nb_typeblocks), 1)
    Pis = predictive[:, 0] * gamma + predictive[:, 1] * 0.5 + predictive[:, 2] * (1 - gamma)

    return 1 - Pis


model_name2class = {
    "optBay": optimal_Bayesian,
    "actKernel": ActionKernel,
    "stimKernel": StimulusKernel,
    "oracle": None
}


def compute_beh_target(trials_df, session_id, subject, model, target, behavior_path, remove_old=False):
    """
    Computes regression target for use with regress_target, using subject, eid, and a string
    identifying the target parameter to output a vector of N_trials length containing the target

    Parameters
    ----------
    trials_df : pandas.DataFrame
        Pandas dataframe containing trial information
    session_id : str
        UUID of the session to compute the target for
    subject : str
        Subject identity in the IBL database, e.g. KS022
    model : str
        String in ['optBay', 'actKernel', 'stimKernel', 'oracle']
    target : str
        signcont (categorical): signed stimulus contrast (includes side information)
        strengthcont (categorical): strength of stimulus contrast (excludes side information)
        stimside (binary): stimulus side information (excludes contrast information)
        choice (binary): animal's binary choice (L/R)
        feedback (binary): reward or punishment
        pLeft (categorical): oracle prior (0.5, 0.8, 0.2)
        prior (continuous): model-based prior
        wheel-speed (continuous): speed of wheel used to indicate decision
        wheel-velocity (continuous): velocity of wheel used to indicate decision
    behavior_path : str
        Path to the behavior data
    remove_old : bool
            Whether to remove old fits

    Returns
    -------
    pandas.Series
        Pandas series in which index is trial number, and value is the target
    """

    '''
    load/fit a behavioral model to compute target on a single session
    Params:
        beh_data_test: if you have to launch the model on beh_data_test.
                       if beh_data_test is explicited, the eid_test will not be considered
        target can be pLeft or signcont. If target=pLeft, it will return the prior predicted by modeltype
                                         if modetype=None, then it will return the actual pLeft (.2, .5, .8)
    '''

    stim_targets = ['signcont', 'strengthcont', 'stimside']
    if target in stim_targets:
        if 'signedContrast' in trials_df.keys():
            out = trials_df['signedContrast'].values
        else:
            out = np.nan_to_num(trials_df.contrastLeft) - np.nan_to_num(trials_df.contrastRight)
        if target == 'signcont':
            return out
        elif target == 'stimside':
            # return vals in {-1, 0, 1}
            out_1 = out.copy()
            out_1[out < 0] = -1
            out_1[out > 0] = 1
            return out_1
        else:
            return np.abs(out)
    if target == 'choice':
        return trials_df.choice.values
    if target == 'feedback':
        return trials_df.feedbackType.values
    elif (target == 'pLeft') and (model == 'oracle'):
        return trials_df.probabilityLeft.values
    elif (target == 'pLeft') and (model == 'optBay'):  # bypass fitting and generate priors
        side, stim, act, _ = format_data(trials_df)
        signal = optimal_Bayesian(act, side)
        return signal.numpy().squeeze()
    elif target in ['wheel-speed', 'wheel-velocity']:
        return trials_df[target].tolist()

    istrained, fullpath = check_bhv_fit_exists(subject, model, session_id, behavior_path, single_zeta=True)

    # load behavior model
    if (not istrained) and (target not in stim_targets) and (model != 'oracle'):
        side, stim, act, _ = format_data(trials_df)
        stimuli, actions, stim_side = format_input([stim], [act], [side])
        model = model_name2class[model](
            path_to_results=behavior_path, session_uuids=session_id, mouse_name=subject, actions=actions,
            stimuli=stimuli, stim_side=stim_side, single_zeta=True,
        )
        model.load_or_train(remove_old=remove_old)
    elif (target not in stim_targets) and (model != 'oracle'):
        model = model_name2class[model](
            path_to_results=behavior_path, session_uuids=session_id, mouse_name=subject, actions=None,
            stimuli=None, stim_side=None, single_zeta=True,
        )
        model.load_or_train(loadpath=str(fullpath))

    # compute signal from behavior model
    stim_side, stimuli, actions, _ = format_data(trials_df)
    stimuli, actions, stim_side = format_input([stimuli], [actions], [stim_side])
    target_ = 'prior' if target == 'pLeft' else target
    signal = model.compute_signal(signal=target_, act=actions, stim=stimuli, side=stim_side)[target_]

    tvec = signal.squeeze()
    config = check_config()
    if config['binarization_value'] is not None:
        tvec = (tvec > config['binarization_value']) * 1

    return tvec


def add_target_to_trials(session_loader, target, intervals, binsize, interval_len=None, mask=None):
    """Add behavior signal to trials df.

    Parameters
    ----------
    session_loader : brainbox.io.one.SessionLoader object
    target : str
        - wheel-speed
        - wheel-velocity
    intervals : np.array
        shape (n_trials, 2) where columns indicate interval onset/offset (seconds)
    binsize : float
        temporal width (seconds) of bins that divide interval
    interval_len : float
    mask : array-like, optional
        existing trial mask, will be updated with mask from new behavior

    Returns
    -------
    tuple
        - updated trials dataframe
        - updated mask array

    """

    if session_loader.trials.shape[0] == 0:
        session_loader.load_trials()

    # load time/value arrays for the desired behavior
    times, values = load_target(session_loader, target)

    # split data into interval-based arrays
    _, target_vals_list, target_mask = split_behavior_data_by_trial(
        times=times, values=values, intervals=intervals, binsize=binsize, interval_len=interval_len)

    if len(target_vals_list) == 0:
        return None, None

    # add to trial df
    trials_df = session_loader.trials.copy()
    trials_df[target] = target_vals_list

    # update mask to exclude trials with bad behavior signal
    if mask is None:
        mask = target_mask
    else:
        mask = mask & target_mask

    return trials_df, mask


def load_target(session_loader, target):
    """Load wheel or DLC data using a SessionLoader and return timestamps and values.

    Parameters
    ----------
    session_loader : brainbox.io.one.SessionLoader object
    target : str
        'wheel-vel' | 'wheel-speed' | 'l-whisker-me' | 'r-whisker-me'

    Returns
    -------
    tuple
        - timestamps for signal
        - associated values
        - bool; True if there was an error loading data

    """

    if target in ['wheel-speed', 'wheel-velocity']:
        session_loader.load_wheel()
        times = session_loader.wheel['times'].to_numpy()
        values = session_loader.wheel['velocity'].to_numpy()
        if target == 'wheel-speed':
            values = np.abs(values)
    elif target == 'l-whisker-me':
        session_loader.load_motion_energy(views=['left'])
        times = session_loader.motion_energy['leftCamera']['times'].to_numpy(),
        values = session_loader.motion_energy['leftCamera']['whiskerMotionEnergy'].to_numpy()
    elif target == 'r-whisker-me':
        session_loader.load_motion_energy(views=['right'])
        times = session_loader.motion_energy['rightCamera']['times'].to_numpy(),
        values = session_loader.motion_energy['rightCamera']['whiskerMotionEnergy'].to_numpy()
    else:
        raise NotImplementedError

    return times, values


def split_behavior_data_by_trial(times, values, intervals, binsize, interval_len=None, allow_nans=False):
    """Format a single session-wide array of target data into a list of trial-based arrays.

    Note: the bin size of the returned data will only be equal to the input `binsize` if that value
    evenly divides `intervals`; for example if `intervals[0]=(0, 0.2)` and `binsize=0.10`,
    then the returned data will have the correct binsize. If `intervals[0]=(0, 0.2)` and
    `binsize=0.06` then the returned data will not have the correct binsize.

    Parameters
    ----------
    times : array-like
        time in seconds for each sample
    values : array-like
        data samples
    intervals : np.array
        shape (n_trials, 2) where columns indicate interval onset/offset (seconds)
    binsize : float
        size of individual bins in interval
    interval_len : float
    allow_nans : bool, optional
        False to skip trials with >0 NaN values in target data

    Returns
    -------
    tuple
        - (list): time in seconds for each trial
        - (list): data for each trial
        - (array-like): mask of good trials (True) and bad trials (False)

    """

    if interval_len is None:
        interval_len = np.nanmedian(intervals[:, 1] - intervals[:, 0])

    # split data by trial
    if np.all(np.isnan(intervals[:, 0])) or np.all(np.isnan(intervals[:, 1])):
        logger.warning('interval times all nan')
        good_trial = np.full((intervals.shape[0],), False)
        times_list = []
        values_list = []
        return times_list, values_list, good_trial

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    # split data into trials
    idxs_beg = np.searchsorted(times, intervals[:, 0], side='right')
    idxs_end = np.searchsorted(times, intervals[:, 1], side='left')
    target_times_og_list = [times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_vals_og_list = [values[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    # interpolate and store
    times_list = []
    values_list = []
    good_trial = [None for _ in range(len(target_times_og_list))]
    for i, (target_time, values) in enumerate(zip(target_times_og_list, target_vals_og_list)):

        if len(values) == 0:
            logger.debug(f'Target data not present on trial {i}; skipping')
            good_trial[i] = False
            times_list.append(None)
            values_list.append(None)
            continue
        if np.sum(np.isnan(values)) > 0 and not allow_nans:
            logger.debug(f'NaNs in target data on trial {i}; skipping')
            good_trial[i] = False
            times_list.append(None)
            values_list.append(None)
            continue
        if np.isnan(intervals[i, 0]) or np.isnan(intervals[i, 1]):
            logger.debug(f'Bad trial interval data on trial {i}; skipping')
            good_trial[i] = False
            times_list.append(None)
            values_list.append(None)
            continue
        if np.abs(intervals[i, 0] - target_time[0]) > binsize:
            logger.debug(f'Target data starts too late on trial {i}; skipping')
            good_trial[i] = False
            times_list.append(None)
            values_list.append(None)
            continue
        if np.abs(intervals[i, 1] - target_time[-1]) > binsize:
            logger.debug(f'Target data ends too early on trial {i}; skipping')
            good_trial[i] = False
            times_list.append(None)
            values_list.append(None)
            continue

        # resample signal in desired bins
        # using `interval_begs[i] + binsize` forces the interpolation to sample the continuous
        # signal at the *end* (or right side) of each bin; this way the spikes in a given bin will
        # fully precede the corresponding target sample for that same bin.
        x_interp = np.linspace(intervals[i, 0] + binsize, intervals[i, 1], n_bins)
        if len(values.shape) > 1 and values.shape[1] > 1:
            n_dims = values.shape[1]
            y_interp_tmps = []
            for n in range(n_dims):
                y_interp_tmps.append(
                    interp1d(target_time, values[:, n], kind='linear', fill_value='extrapolate')(x_interp)
                )
            y_interp = np.hstack([y[:, None] for y in y_interp_tmps])
        else:
            y_interp = interp1d(target_time, values, kind='linear', fill_value='extrapolate')(x_interp)

        times_list.append(x_interp)
        values_list.append(y_interp)
        good_trial[i] = True

    return times_list, values_list, np.array(good_trial)
