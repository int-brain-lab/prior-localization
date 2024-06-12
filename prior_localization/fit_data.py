import logging
import pickle
import numpy as np

from sklearn import linear_model as sklm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
from sklearn.utils.class_weight import compute_sample_weight

from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask

from prior_localization.prepare_data import (
    prepare_ephys,
    prepare_behavior,
    prepare_motor,
    prepare_pupil,
    prepare_widefield,
    prepare_widefield_old,
)
from prior_localization.functions.behavior_targets import add_target_to_trials
from prior_localization.functions.neurometric import get_neurometric_parameters
from prior_localization.functions.utils import (
    check_inputs,
    check_config,
    compute_mask,
    subtract_motor_residuals,
    format_data_for_decoding,
    logisticreg_criteria,
    str2int,
)

# Set up logger
logger = logging.getLogger('prior_localization')
# Load and check configuration file
config = check_config()


def fit_session_ephys(
        one, session_id, subject, probe_name, output_dir, pseudo_ids=None, target='pLeft', align_event='stimOn_times',
        min_rt=0.08, max_rt=None, time_window=(-0.6, -0.1), saturation_intervals=None,
        binsize=None, n_bins_lag=None, n_bins=None, model='optBay',
        n_runs=10, compute_neurometrics=False, motor_residuals=False, stage_only=False
):
    """
    Fits a single session for ephys data.
    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    probe_name: str or list of str
     Probe name(s), if list of probe names, the probes, the data of both probes will be merged for decoding
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. Note that the function will reuse
     previously computed behavioural models if you are using the same path as for a previous run. Decoding results
     will be overwritten though.
    pseudo_ids: None or list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont, 'wheel-speed', 'wheel-velocity'},
     default is pLeft, meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is None. If None, don't apply.
    time_window: tuple of float
     Time window in which neural activity is considered, relative to align_event, default is (-0.6, -0.1)
    saturation_intervals: str or list of str or None
         If str or list of str, the name of the interval(s) to be used to exclude trials if the ephys signal shows
         saturation in the interval(s). Default is None. Possible values are:
            saturation_stim_plus04
            saturation_feedback_plus04
            saturation_move_minus02
            saturation_stim_minus04_minus01
            saturation_stim_plus06
            saturation_stim_minus06_plus06
            saturation_stim_plus01
    binsize : float or None
     if None, sum spikes in time_window for decoding; if float, split time window into smaller bins
    n_bins_lag : int or None
     number of lagged timepoints (includes zero lag) for decoding wheel and DLC targets
    n_bins : int or None
     number of bins; should be computable from intervals and binsize, but there are occasional rounding errors
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    n_runs: int
     Number of times to repeat full nested cross validation with different folds
    compute_neurometrics: bool
     Whether to compute neurometric shift and slopes (cf. Fig 3 of the paper)
    motor_residuals: bool
     Whether ot compute the motor residual before performing neural decoding. This argument is used to study embodiment
     corresponding to figure 2f, default is False
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding

    Returns
    -------
    list
     List of paths to the results files
    """

    # Check some inputs
    pseudo_ids, output_dir = check_inputs(
        model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics, motor_residuals
    )

    np.random.seed(str2int(session_id) + np.sum(pseudo_ids))

    # Load trials data and compute mask
    sl = SessionLoader(one, eid=session_id)
    sl.load_trials()
    _, trials_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        saturation_intervals=saturation_intervals,
        exclude_nochoice=True, exclude_unbiased=False,
    )
    intervals = np.vstack([sl.trials[align_event] + time_window[0], sl.trials[align_event] + time_window[1]]).T
    if target in ['wheel-speed', 'wheel-velocity']:
        # add behavior signal to df and update trials mask to reflect trials with signal issues
        if binsize is None:
            raise ValueError(f"If target is wheel-speed or wheel-velocity, binsize cannot be None")
        sl.trials, trials_mask = add_target_to_trials(
            session_loader=sl, target=target, intervals=intervals, binsize=binsize,
            interval_len=time_window[1] - time_window[0], mask=trials_mask)

    if sum(trials_mask) <= config['min_trials']:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {config['min_trials']}.")

    # Prepare ephys data
    data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys(
        one, session_id, probe_name, config['regions'], intervals,
        binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
        qc=config['unit_qc'], min_units=config['min_units'], stage_only=stage_only,
    )
    n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)

    # Compute or load behavior targets
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, n_pseudo_sets=n_pseudo_sets,
        output_dir=output_dir, model=model, target=target, compute_neurometrics=compute_neurometrics)

    # Remove the motor residuals from the targets if indicated
    if motor_residuals:
        motor_signals = prepare_motor(one, session_id, time_window=time_window)
        all_targets, all_masks = subtract_motor_residuals(motor_signals, all_targets, all_masks)

    # If we are only staging data, we are done here
    if stage_only:
        return

    # Create strings for saving
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    probe_str = 'merged_probes' if (isinstance(probe_name, list) and len(probe_name) > 1) else probe_name

    # Fit per region
    filenames = []
    for i in range(len(data_epoch)):

        # Apply mask to targets
        if isinstance(all_targets[i][0], list):
            targets_masked = [
                [t[m_] for m_ in np.squeeze(np.where(m))] for t, m in zip(all_targets[i], all_masks[i])
            ]
        else:
            targets_masked = [t[m] for t, m in zip(all_targets[i], all_masks[i])]

        # Apply mask to ephys data
        if isinstance(data_epoch[0], list):
            data_masked = [
                [data_epoch[i][m_] for m_ in np.squeeze(np.where(m))] for m in all_masks[i]
            ]
        else:
            data_masked = [data_epoch[i][m] for m in all_masks[i]]

        # Fit
        fit_results = fit_target(
            all_data=data_masked,
            all_targets=targets_masked,
            all_trials=all_trials[i],
            n_runs=n_runs,
            all_neurometrics=all_neurometrics[i],
            pseudo_ids=pseudo_ids,
            base_rng_seed=str2int(session_id + '_'.join(actual_regions[i])),
        )

        # Add the mask to fit results
        save_predictions = True if pseudo_ids[0] == -1 else config['save_predictions']
        for fit_result in fit_results:
            fit_result['mask'] = all_masks[i] if save_predictions else None

        # Create output paths and save
        region_str = config['regions'] if (config['regions'] == 'all_regions') or (
                config['regions'] in config['region_defaults'].keys()) else '_'.join(actual_regions[i])
        filename = output_dir.joinpath(subject, session_id, f'{region_str}_{probe_str}_pseudo_ids_{pseudo_str}.pkl')
        filename.parent.mkdir(parents=True, exist_ok=True)

        outdict = {
            "fit": fit_results,
            "subject": subject,
            "eid": session_id,
            "probe": probe_str,
            "region": actual_regions[i],
            "N_units": n_units[i],
            "cluster_uuids": cluster_ids[i],
        }
        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)
        filenames.append(filename)
    return filenames


def fit_session_widefield(
        one, session_id, subject, output_dir, pseudo_ids=None, hemisphere=("left", "right"), target='pLeft',
        align_event='stimOn_times', min_rt=0.08, max_rt=None, frame_window=(-2, -2), model='optBay', n_runs=10,
        compute_neurometrics=False, stage_only=False, old_data=False
):

    """
    Fit a single session for widefield data.

    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. Note that the function will reuse
     previously computed behavioural models if you are using the same path as for a previous run. Decoding results
     will be overwritten though.
    pseudo_ids: None or list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    hemisphere: str or tuple of str
     Which hemisphere(s) to decode from {'left', 'right', ('left', 'right')}
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is None. If None, don't apply.
    frame_window: tuple of int
     Window in which neural activity is considered, in frames relative to align_event, default is (-2, -2) i.e. only a
     single frame is considered
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    n_runs: int
     Number of times to repeat full nested cross validation with different folds
    compute_neurometrics: bool
     Whether to compute neurometric shift and slopes (cf. Fig 3 of the paper)
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding
    old_data: False or str
     Only used for sanity check, if false, use updated way of loading data from ONE. If str it should be a path
     to local copies of the previously used version of the data.

    Returns
    -------
    list
     List of paths to the results files
    """

    # Check some inputs
    pseudo_ids, output_dir = check_inputs(model, pseudo_ids, target, output_dir, config, logger, compute_neurometrics)

    # Load trials data
    sl = SessionLoader(one, eid=session_id)
    sl.load_trials()
    trials_mask = compute_mask(sl.trials, align_event=align_event, min_rt=min_rt, max_rt=max_rt, n_trials_crop_end=1)
    # _, trials_mask = load_trials_and_mask(
    #     one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
    #     min_trial_len=None, max_trial_len=None,
    #     exclude_nochoice=True, exclude_unbiased=False,
    # )
    if sum(trials_mask) <= config['min_trials']:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {config['min_trials']}.")

    # Prepare widefield data
    if old_data is False:
        data_epoch, actual_regions = prepare_widefield(
            one, session_id, regions=config['regions'], align_times=sl.trials[align_event].values,
            frame_window=frame_window, hemisphere=hemisphere, stage_only=stage_only
        )
    else:
        data_epoch, actual_regions = prepare_widefield_old(old_data, hemisphere=hemisphere, regions=config['regions'],
                                                           align_event=align_event, frame_window=frame_window)

    n_pseudo_sets = 1 if actual_regions is None else len(actual_regions)

    # Compute or load behavior targets
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, n_pseudo_sets=n_pseudo_sets,
        output_dir=output_dir, model=model, target=target, compute_neurometrics=compute_neurometrics)

    # If we are only staging data, we are done here
    if stage_only:
        return

    # Strings for saving
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    hemi_str = 'both_hemispheres' if isinstance(hemisphere, tuple) or isinstance(hemisphere, list) else hemisphere

    # Fit data per region
    filenames = []
    for i in range(len(data_epoch)):

        # Apply mask to targets
        if isinstance(all_targets[i][0], list):
            targets_masked = [
                [t[m_] for m_ in np.squeeze(np.where(m))] for t, m in zip(all_targets[i], all_masks[i])
            ]
        else:
            targets_masked = [t[m] for t, m in zip(all_targets[i], all_masks[i])]

        # Apply mask to ephys data
        if isinstance(data_epoch[0], list):
            data_masked = [
                [data_epoch[i][m_] for m_ in np.squeeze(np.where(m))] for m in all_masks[i]
            ]
        else:
            data_masked = [data_epoch[i][m] for m in all_masks[i]]

        # Fit
        fit_results = fit_target(
            all_data=data_masked,
            all_targets=targets_masked,
            all_trials=all_trials[i],
            n_runs=n_runs,
            all_neurometrics=all_neurometrics[i],
            pseudo_ids=pseudo_ids,
            base_rng_seed=str2int(session_id + '_'.join(actual_regions[i])),
        )

        # Add the mask to fit results
        for fit_result in fit_results:
            fit_result['mask'] = all_masks[i] if config['save_predictions'] else None

        # Create output paths and save
        region_str = config['regions'] if (config['regions'] == 'all_regions') or (
                config['regions'] in config['region_defaults'].keys()) else '_'.join(actual_regions[i])
        filename = output_dir.joinpath(subject, session_id, f'{region_str}_{hemi_str}_pseudo_ids_{pseudo_str}.pkl')
        filename.parent.mkdir(parents=True, exist_ok=True)

        outdict = {
            "fit": fit_results,
            "subject": subject,
            "eid": session_id,
            "hemisphere": hemisphere,
            "region": actual_regions[0],
            "N_units": data_epoch[0].shape[1],
        }

        with open(filename, "wb") as fw:
            pickle.dump(outdict, fw)
        filenames.append(filename)
    return filenames


def fit_session_pupil(
        one, session_id, subject, output_dir, pseudo_ids=None, target='pLeft', align_event='stimOn_times',
        min_rt=0.08, max_rt=None, time_window=(-0.6, -0.1), model='optBay', n_runs=10, stage_only=False
):
    """
    Fit pupil tracking data to behavior (instead of neural activity)

    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. Note that the function will reuse
     previously computed behavioural models if you are using the same path as for a previous run. Decoding results
     will be overwritten though.
    pseudo_ids: None or list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is None. If None, don't apply.
    time_window: tuple of float
     Time window in which pupil movement is considered, relative to align_event, default is (-0.6, -0.1)
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    n_runs: int
     Number of times to repeat full nested cross validation with different folds
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding

    Returns
    -------
    list
     List of paths to the results files
    """
    # Check some inputs
    pseudo_ids, output_dir = check_inputs(model, pseudo_ids, target, output_dir, config, logger)

    # Load trials data
    sl = SessionLoader(one, eid=session_id)
    sl.load_trials()
    _, trials_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )
    if sum(trials_mask) <= config['min_trials']:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {config['min_trials']}.")

    # Compute or load behavior targets
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, n_pseudo_sets=1, output_dir=output_dir,
        model=model, target=target)

    # Load the pupil data
    pupil_data = prepare_pupil(one, session_id=session_id, time_window=time_window, align_event=align_event)

    if stage_only:
        return

    # For trials where there was no pupil data recording (start/end), add these to the trials_mask
    # `all_masks` is a list returned from prepare_behavior(), and generally has an entry for each region we are decoding
    # from. Here we're only decoding from the pupil, so len(all_masks) = 1.
    all_masks[0] = [a & ~np.any(np.isnan(pupil_data), axis=1) for a in all_masks[0]]

    # Apply mask to targets
    targets_masked = [t[m] for t, m in zip(all_targets[0], all_masks[0])]

    # Apply mask to ephys data
    pupil_masked = [pupil_data[m] for m in all_masks[0]]

    # Fit
    fit_results = fit_target(
        all_data=pupil_masked,
        all_targets=targets_masked,
        all_trials=all_trials[0],
        n_runs=n_runs,
        all_neurometrics=all_neurometrics[0],
        pseudo_ids=pseudo_ids,
        base_rng_seed=str2int(session_id),
    )

    # Create output paths and save
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    filename = output_dir.joinpath(subject, session_id, f'pupil_pseudo_ids_{pseudo_str}.pkl')
    filename.parent.mkdir(parents=True, exist_ok=True)

    outdict = {
        "fit": fit_results,
        "subject": subject,
        "eid": session_id,
    }
    with open(filename, "wb") as fw:
        pickle.dump(outdict, fw)

    return filename


def fit_session_motor(
        one, session_id, subject, output_dir, pseudo_ids=None, target='pLeft', align_event='stimOn_times',
        min_rt=0.08, max_rt=None, time_window=(-0.6, -0.1), model='optBay', n_runs=10, stage_only=False
):
    """
    Fit movement tracking data to behavior (instead of neural actvity)

    Parameters
    ----------
    one: one.api.ONE object
     ONE instance connected to database that data should be loaded from
    session_id: str
     Database UUID of the session uuid to run the decoding on
    subject: str
     Nickname of the mouse
    output_dir: str, pathlib.Path or None
     Directory in which the results are saved, will be created if it doesn't exist. Note that the function will reuse
     previously computed behavioural models if you are using the same path as for a previous run. Decoding results
     will be overwritten though.
    pseudo_ids: None or list of int
     List of sessions / pseudo sessions to decode, -1 represents decoding of the actual session, integers > 0 indicate
     pseudo_session ids.
    target: str
     Target to be decoded, options are {pLeft, prior, choice, feedback, signcont}, default is pLeft,
     meaning the prior probability that the stimulus  will appear on the left side
    align_event: str
     Event to which we align the time window, default is stimOn_times (stimulus onset). Options are
     {"firstMovement_times", "goCue_times", "stimOn_times", "feedback_times"}
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is None. If None, don't apply.
    time_window: tuple of float
     Time window in which movement is considered, relative to align_event, default is (-0.6, -0.1)
    model: str
     Model to be decoded, options are {optBay, actKernel, stimKernel, oracle}, default is optBay
    n_runs: int
     Number of times to repeat full nested cross validation with different folds
    stage_only: bool
     If true, only download all required data, don't perform the actual decoding

    Returns
    -------
    list
     List of paths to the results files
    """

    # Check some inputs
    pseudo_ids, output_dir = check_inputs(model, pseudo_ids, target, output_dir, config, logger)

    # Load trials data
    sl = SessionLoader(one, eid=session_id)
    sl.load_trials()
    _, trials_mask = load_trials_and_mask(
        one=one, eid=session_id, sess_loader=sl, min_rt=min_rt, max_rt=max_rt,
        min_trial_len=None, max_trial_len=None,
        exclude_nochoice=True, exclude_unbiased=False,
    )
    if sum(trials_mask) <= config['min_trials']:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {config['min_trials']}.")

    # Compute or load behavior targets
    all_trials, all_targets, all_masks, all_neurometrics = prepare_behavior(
        session_id, subject, sl.trials, trials_mask, pseudo_ids=pseudo_ids, n_pseudo_sets=1,
        output_dir=output_dir, model=model, target=target)

    # Load the motor data
    motor_data = prepare_motor(one, session_id=session_id, time_window=time_window, align_event=align_event)

    if stage_only:
        return

    # For trials where there was no pupil data recording (start/end), add these to the trials_mask
    # `all_masks` is a list returned from prepare_behavior(), and generally has an entry for each region we are decoding
    # from. Here we're only decoding from pose estimation traces, so len(all_masks) = 1.
    all_masks[0] = [a & ~np.any(np.isnan(motor_data), axis=1) for a in all_masks[0]]

    # Apply mask to targets
    targets_masked = [t[m] for t, m in zip(all_targets[0], all_masks[0])]

    # Apply mask to motor data
    motor_masked = [motor_data[m] for m in all_masks[0]]

    # Fit
    fit_results = fit_target(
        all_data=motor_masked,
        all_targets=targets_masked,
        all_trials=all_trials[0],
        n_runs=n_runs,
        all_neurometrics=all_neurometrics[0],
        pseudo_ids=pseudo_ids,
        base_rng_seed=str2int(session_id),
    )

    # Create output paths and save
    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if len(pseudo_ids) > 1 else str(pseudo_ids[0])
    filename = output_dir.joinpath(subject, session_id, f'motor_pseudo_ids_{pseudo_str}.pkl')
    filename.parent.mkdir(parents=True, exist_ok=True)

    outdict = {
        "fit": fit_results,
        "subject": subject,
        "eid": session_id,
    }
    with open(filename, "wb") as fw:
        pickle.dump(outdict, fw)

    return filename


def fit_target(
        all_data, all_targets, all_trials, n_runs, all_neurometrics=None, pseudo_ids=None,
        base_rng_seed=0
):
    """
    Fits data (neural, motor, etc) to behavior targets.

    Parameters
    ----------
    all_data : list of np.ndarray
        List of neural or other data, each element is a (n_trials, n_units) array with the averaged neural activity
    all_targets : list of np.ndarray
        List of behavior targets, each element is a (n_trials,) array with the behavior targets for one (pseudo)session
    all_trials : list of pd.DataFrames
        List of trial information, each element is a pd.DataFrame with the trial information for one (pseudo)session
    n_runs: int
        Number of times to repeat full nested cross validation with different folds
    all_neurometrics : list of pd.DataFrames or None
        List of neurometrics, each element is a pd.DataFrame with the neurometrics for one (pseudo)session.
        If None, don't compute neurometrics. Default is None
    pseudo_ids : list of int or None
        List of pseudo session ids, -1 indicates the actual session. If None, run only on actual session.
        Default is None.
    base_rng_seed : int
        seed that will be added to run- and pseudo_id-specific seeds
    """

    # Loop over (pseudo) sessions and then over runs
    if pseudo_ids is None:
        pseudo_ids = [-1]
    if not all_neurometrics:
        all_neurometrics = [None] * len(all_targets)
    fit_results = []
    for targets, data, trials, neurometrics, pseudo_id in zip(
            all_targets, all_data, all_trials, all_neurometrics, pseudo_ids):
        # run decoders
        for i_run in range(n_runs):
            # set seed for reproducibility
            if pseudo_id == -1:
                rng_seed = base_rng_seed + i_run
            else:
                rng_seed = base_rng_seed + pseudo_id * n_runs + i_run
            fit_result = decode_cv(
                ys=targets,
                Xs=data,
                estimator=config['estimator'],
                estimator_kwargs=config['estimator_kwargs'],
                hyperparam_grid=config['hparam_grid'],
                save_binned=False,
                save_predictions=True if pseudo_id == -1 else config['save_predictions'],
                shuffle=config['shuffle'],
                balanced_weight=config['balanced_weighting'],
                rng_seed=rng_seed,
                use_cv_sklearn_method=config['use_native_sklearn_for_hyperparam_estimation'],
            )

            fit_result["trials_df"] = trials
            fit_result["pseudo_id"] = pseudo_id
            fit_result["run_id"] = i_run

            if neurometrics:
                fit_result["full_neurometric"], fit_result["fold_neurometric"] = get_neurometric_parameters(
                    fit_result, trialsdf=neurometrics, compute_on_each_fold=config['compute_neuro_on_each_fold']
                )
            else:
                fit_result["full_neurometric"] = None
                fit_result["fold_neurometric"] = None

            fit_results.append(fit_result)

    return fit_results


def decode_cv(ys, Xs, estimator, estimator_kwargs, balanced_weight=False, hyperparam_grid=None, test_prop=0.2,
              n_folds=5, save_binned=False, save_predictions=True, verbose=False, shuffle=True, outer_cv=True,
              rng_seed=None, use_cv_sklearn_method=False):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator.

    Parameters
    ----------
    ys : list of arrays or np.ndarray or pandas.Series
        targets; if list, each entry is an array of targets for one trial. if 1D numpy array, each
        entry is treated as a single scalar for one trial. if pd.Series, trial number is the index
        and teh value is the target.
    Xs : list of arrays or np.ndarray
        predictors; if list, each entry is an array of neural activity for one trial. if 2D numpy
        array, each row is treated as a single vector of activity for one trial, i.e. the array is
        of shape (n_trials, n_neurons)
    estimator : sklearn.linear_model object
        estimator from sklearn which provides .fit, .score, and .predict methods. CV estimators
        are NOT SUPPORTED. Must be a normal estimator, which is internally wrapped with
        GridSearchCV
    estimator_kwargs : dict
        additional arguments for sklearn estimator
    balanced_weight : bool
        balanced weighting to target
    hyperparam_grid : dict
        key indicates hyperparameter to grid search over, and value is an array of nodes on the
        grid. See sklearn.model_selection.GridSearchCV : param_grid for more specs.
        Defaults to None, which means no hyperparameter estimation or GridSearchCV use.
    test_prop : float
        proportion of data to hold out as the test set after running hyperparameter tuning; only
        used if `outer_cv=False`
    n_folds : int
        Number of folds for cross-validation during hyperparameter tuning; only used if
        `outer_cv=True`
    save_binned : bool
        True to put the regressors Xs into the output dictionary.
        Can cause file bloat if saving outputs.
        Note: this function does not actually save any files!
    save_predictions : bool
        True to put the model predictions into the output dictionary.
        Can cause file bloat if saving outputs.
        Note: this function does not actually save any files!
    shuffle : bool
        True for interleaved cross-validation, False for contiguous blocks
    outer_cv: bool
        Perform outer cross validation such that the testing spans the entire dataset
    rng_seed : int
        control data splits
    verbose : bool
        Whether you want to hear about the function's life, how things are going, and what the
        neighbor down the street said to it the other day.

    Returns
    -------
    dict
        Dictionary of fitting outputs including:
            - Regression score (from estimator)
            - Decoding coefficients
            - Decoding intercept
            - Per-trial target values (copy of tvec)
            - Per-trial predictions from model
            - Input regressors (optional, see Xs argument)

    """

    # transform target data into standard format: list of np.ndarrays
    ys, Xs = format_data_for_decoding(ys, Xs)

    # initialize containers to save outputs
    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]  # different for logistic regression

    # split the dataset in two parts, train and test
    # when shuffle=False, the method will take the end of the dataset to create the test set
    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)
    if outer_cv:
        # create kfold function to loop over
        get_kfold = lambda: KFold(n_folds if not use_cv_sklearn_method else 50, shuffle=shuffle).split(indices)
        # define function to evaluate whether folds are satisfactory
        if estimator == sklm.LogisticRegression:
            # folds must be chosen such that 2 classes are present in each fold, with minimum 2 examples per class
            assert logisticreg_criteria(ys)
            isysat = lambda ys: logisticreg_criteria(ys, min_unique_counts=2)
        else:
            isysat = lambda ys: True
        sample_count, _, outer_kfold_iter = sample_folds(ys, get_kfold, isysat)
        if sample_count > 1:
            print(f'sampled outer folds {sample_count} times to ensure enough targets')
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])
        outer_kfold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]

    # scoring function; use R2 for linear regression, accuracy for logistic regression
    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # Select either the GridSearchCV estimator for a normal estimator, or use the native estimator
    # in the case of CV-type estimators
    if estimator == sklm.RidgeCV or estimator == sklm.LassoCV or estimator == sklm.LogisticRegressionCV:
        raise NotImplementedError("the code does not support a CV-type estimator.")
    else:
        # loop over outer folds
        for train_idxs_outer, test_idxs_outer in outer_kfold_iter:
            # outer fold data split
            # X_train = np.vstack([Xs[i] for i in train_idxs])
            # y_train = np.concatenate([ys[i] for i in train_idxs], axis=0)
            # X_test = np.vstack([Xs[i] for i in test_idxs])
            # y_test = np.concatenate([ys[i] for i in test_idxs], axis=0)
            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            key = list(hyperparam_grid.keys())[0]  # TODO: make this more robust

            if not use_cv_sklearn_method:

                """NOTE
                This section of the code implements a modified nested-cross validation procedure. When decoding signals
                with multiple samples per trial -- such as the wheel -- we need to create folds that do not put
                samples from the same trial into different folds.
                """

                # now loop over inner folds
                idx_inner = np.arange(len(X_train))

                get_kfold_inner = lambda: KFold(n_splits=n_folds, shuffle=shuffle).split(idx_inner)

                # produce inner_fold_iter
                if estimator == sklm.LogisticRegression:
                    # make sure data has at least 2 examples per class
                    assert logisticreg_criteria(y_train, min_unique_counts=2)
                    # folds must be chosen such that 2 classes are present in each fold, with min 1 example per class
                    isysat_inner = lambda ys: logisticreg_criteria(ys, min_unique_counts=1)
                else:
                    isysat_inner = lambda ys: True
                sample_count, _, inner_kfold_iter = sample_folds(y_train, get_kfold_inner, isysat_inner)
                if sample_count > 1:
                    print(f'sampled inner folds {sample_count} times to ensure enough targets')

                r2s = np.zeros([n_folds, len(hyperparam_grid[key])])
                inner_predictions = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                inner_targets = np.zeros([len(y_train), len(hyperparam_grid[key])]) + np.nan
                for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold_iter):

                    # inner fold data split
                    X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                    y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                    X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                    y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                    for i_alpha, alpha in enumerate(hyperparam_grid[key]):

                        # compute weight for each training sample if requested
                        sample_weight = compute_sample_weight("balanced", y=y_train_inner) if balanced_weight else None

                        # initialize model
                        model_inner = estimator(**{**estimator_kwargs, key: alpha})
                        # fit model
                        model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
                        # evaluate model
                        pred_test_inner = model_inner.predict(X_test_inner)
                        # record predictions and targets to check for nans later
                        inner_predictions[test_idxs_inner, i_alpha] = np.mean(pred_test_inner)
                        inner_targets[test_idxs_inner, i_alpha] = np.mean(y_test_inner)
                        # record score
                        r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

                assert np.all(~np.isnan(inner_predictions))
                assert np.all(~np.isnan(inner_targets))

                # select model with best hyperparameter value evaluated on inner-fold test data;
                # refit/evaluate on all inner-fold data
                r2s_avg = r2s.mean(axis=0)

                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)

                # compute weight for each training sample if requested
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                # initialize model
                best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
                model = estimator(**{**estimator_kwargs, key: best_alpha})
                # fit model
                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
            else:
                if estimator not in [Ridge, Lasso]:
                    raise NotImplementedError("This case is not implemented")
                model = (
                    RidgeCV(alphas=hyperparam_grid[key])
                    if estimator == Ridge
                    else LassoCV(alphas=hyperparam_grid[key])
                )
                X_train_array = np.vstack(X_train)
                y_train_array = np.concatenate(y_train, axis=0)
                sample_weight = compute_sample_weight("balanced", y=y_train_array) if balanced_weight else None

                model.fit(X_train_array, y_train_array, sample_weight=sample_weight)
                best_alpha = model.alpha_

            # evalute model on train data
            y_pred_train = model.predict(X_train_array)
            scores_train.append(scoring_f(y_train_array, y_pred_train))

            # evaluate model on test data
            y_true = np.concatenate(y_test, axis=0)
            y_pred = model.predict(np.vstack(X_test))
            if isinstance(model, sklm.LogisticRegression):
                y_pred_probs = model.predict_proba(np.vstack(X_test))[:, 1]  # probability of class 1
            else:
                y_pred_probs = None
            scores_test.append(scoring_f(y_true, y_pred))

            # save the raw prediction in the case of linear and the predicted probabilities when
            # working with logitistic regression
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    # we already computed these estimates, take from above
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    # we already computed these above, but after all trials were stacked; recompute per-trial
                    predictions[i_global] = model.predict(X_test[i_fold])
                    if isinstance(model, sklm.LogisticRegression):
                        predictions_to_save[i_global] = model.predict_proba(X_test[i_fold])[:, 1]
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            # save out other data of interest
            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            if model.fit_intercept:
                intercepts.append(model.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: best_alpha})

    ys_true_full = np.concatenate(ys, axis=0)
    ys_pred_full = np.concatenate(predictions, axis=0)
    outdict = dict()
    outdict["scores_test_full"] = scoring_f(ys_true_full, ys_pred_full)
    outdict["scores_train"] = scores_train
    outdict["scores_test"] = scores_test
    outdict["Rsquared_test_full"] = r2_score(ys_true_full, ys_pred_full)
    if estimator == sklm.LogisticRegression:
        outdict["acc_test_full"] = accuracy_score(ys_true_full, ys_pred_full)
        outdict["balanced_acc_test_full"] = balanced_accuracy_score(
            ys_true_full, ys_pred_full
        )
    outdict["weights"] = weights if save_predictions else None
    outdict["intercepts"] = intercepts if save_predictions else None
    outdict["target"] = ys
    outdict["predictions_test"] = predictions_to_save
    outdict["regressors"] = Xs if save_binned else None
    outdict["idxes_test"] = idxes_test if save_predictions else None
    outdict["idxes_train"] = idxes_train if save_predictions else None
    outdict["best_params"] = best_params if save_predictions else None
    outdict["n_folds"] = n_folds
    if hasattr(model, "classes_"):
        outdict["classes_"] = model.classes_

    # logging
    if verbose:
        # verbose output
        if outer_cv:
            print("Performance is only described for last outer fold \n")
        print(
            "Possible regularization parameters over {} validation sets:".format(
                n_folds
            )
        )
        print("{}: {}".format(list(hyperparam_grid.keys())[0], hyperparam_grid))
        print("\nBest parameters found over {} validation sets:".format(n_folds))
        print(model.best_params_)
        print("\nAverage scores over {} validation sets:".format(n_folds))
        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, model.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\n", "Detailed scores on {} validation sets:".format(n_folds))
        for i_fold in range(n_folds):
            tscore_fold = list(
                np.round(model.cv_results_["split{}_test_score".format(int(i_fold))], 3)
            )
            print("perf on fold {}: {}".format(int(i_fold), tscore_fold))

        print("\n", "Detailed classification report:", "\n")
        print("The model is trained on the full (train + validation) set.")

    return outdict


def sample_folds(ys, get_kfold, isfoldsat, max_iter=100):
    """Sample a set of folds and ensure each fold satisfies user-defined criteria.

    Parameters
    ----------
    ys : array-like
        array of indices to split into folds
    get_kfold : callable
        callable function that returns a generator object
    isfoldsat : callable
        callable function that takes an array as input and returns a bool denoting is criteria are satisfied
    max_iter : int, optional
        maximum number of attempts to split folds

    Returns
    -------
    tuple
        - (int) number of samples required to satisfy fold criteria
        - (generator) fold generator
        - (list) list of tuples (train_idxs, test_idxs), one tuple for each fold

    """
    sample_count = 0
    ysatisfy = [False]
    while not np.all(np.array(ysatisfy)):
        assert sample_count < max_iter
        sample_count += 1
        outer_kfold = get_kfold()
        fold_iter = [(train_idxs, test_idxs) for _, (train_idxs, test_idxs) in enumerate(outer_kfold)]
        ysatisfy = [isfoldsat(np.concatenate([ys[i] for i in t_idxs], axis=0)) for t_idxs, _ in fold_iter]

    return sample_count, outer_kfold, fold_iter
