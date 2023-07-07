import logging
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import RidgeCV

from ibllib.atlas import BrainRegions
from brainbox.io.one import SessionLoader
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.behavior.dlc import get_licks

from brainwidemap.bwm_loading import load_good_units, merge_probes
from behavior_models.utils import format_data, format_input
from behavior_models.models import ActionKernel, StimulusKernel

from prior_localization.functions.behavior_targets import optimal_Bayesian, compute_beh_target
from prior_localization.functions.utils import compute_mask, derivative, check_bhv_fit_exists, average_data_in_epoch
from prior_localization.functions.nulldistributions import generate_null_distribution_session
from prior_localization.functions.neurometric import compute_neurometric_prior

from prior_localization.params import REGION_DEFAULTS, COMPUTE_NEUROMETRIC, DECODE_DERIVATIVE, MOTOR_RESIDUAL, \
    MIN_BEHAV_TRIALS, MOTOR_BIN

logger = logging.getLogger('prior_localization')

model_name2class = {
    "optBay": optimal_Bayesian,
    "actKernel": ActionKernel,
    "stimKernel": StimulusKernel,
    "oracle": None
}


def prepare_ephys(one, session_id, probe_name, regions, intervals, qc=1, min_units=10, stage_only=False):

    # Load spikes and clusters and potentially merge probes
    if isinstance(probe_name, list) and len(probe_name) > 1:
        to_merge = [load_good_units(one, pid=None, eid=session_id, qc=qc, pname=probe_name)
                    for probe_name in probe_name]
        spikes, clusters = merge_probes([spikes for spikes, _ in to_merge], [clusters for _, clusters in to_merge])
    else:
        spikes, clusters = load_good_units(one, pid=None, eid=session_id, qc=qc, pname=probe_name)

    # This allows us to just stage the data without running the analysis, we can then switch ONE in local mode
    if stage_only:
        return None, None

    # Prepare list of brain regions
    brainreg = BrainRegions()
    beryl_regions = brainreg.acronym2acronym(clusters['acronym'], mapping="Beryl")
    if isinstance(regions, str):
        if regions in REGION_DEFAULTS.keys():
            regions = REGION_DEFAULTS[regions]
        elif regions == 'single_regions':
            regions = [[k] for k in np.unique(beryl_regions) if k not in ['root', 'void']]
        elif regions == 'all_regions':
            regions = [np.unique([r for r in beryl_regions if r not in ['root', 'void']])]
        else:
            regions = [regions]
    elif isinstance(regions, list):
        pass

    binned_spikes = []
    actual_regions = []
    for region in regions:
        # find all clusters in region (where region can be a list of regions)
        region_mask = np.isin(beryl_regions, region)
        if sum(region_mask) < min_units:
            print(f"{'_'.join(region)} below min units threshold ({min_units}) : {sum(region_mask)}, not decoding")
        else:
            # find all spikes in those clusters
            spike_mask = np.isin(spikes['clusters'], clusters[region_mask].index)
            binned, _ = get_spike_counts_in_bins(spikes['times'][spike_mask], spikes['clusters'][spike_mask], intervals)
            binned_spikes.append(binned.T)
            actual_regions.append(region)
    return binned_spikes, actual_regions


def prepare_motor(one, eid, align_event='stimOn_times', time_window=(-0.6, -0.1)):

    # Initiate session loader, load trials, wheel, motion energy and pose estimates
    sl = SessionLoader(one, eid)
    sl.load_trials()
    sl.load_wheel()
    sl.load_motion_energy(views=['left', 'right'])
    sl.load_pose(views=['left', 'right'], likelihood_thr=0.9)

    # Convert wheel velocity to acceleration and normalize to max
    wheel_acc = abs(sl.wheel['velocity'])
    wheel_acc = wheel_acc / max(wheel_acc)
    wheel_epochs = average_data_in_epoch(
        sl.wheel['times'], wheel_acc, sl.trials, align_event=align_event, epoch=time_window
    )

    # Get the licks from both cameras combined and bin
    lick_times = [get_licks(sl.pose[f'{side}Camera'], sl.pose[f'{side}Camera']['times']) for side in ['left', 'right']]
    lick_times = np.unique(np.concatenate(lick_times))
    lick_times.sort()
    # binned_licks[i] contains all licks greater/equal to binned_licks_times[i-1] and smaller than binned_licks_times[i]
    binned_licks_times = np.arange(lick_times[0], lick_times[-1] + MOTOR_BIN, MOTOR_BIN)
    binned_licks = np.bincount(np.digitize(lick_times, binned_licks_times))
    lick_epochs = average_data_in_epoch(
        binned_licks_times, binned_licks, sl.trials, align_event=align_event, epoch=time_window
    )

    # Get nose position
    nose_epochs = average_data_in_epoch(
        sl.pose['leftCamera']['times'], sl.pose['leftCamera']['nose_tip_x'], sl.trials,
        align_event=align_event, epoch=time_window
    )

    # Get whisking
    whisking_l_epochs = average_data_in_epoch(
        sl.motion_energy['leftCamera']['times'], sl.motion_energy['leftCamera']['whiskerMotionEnergy'], sl.trials,
        align_event=align_event, epoch=time_window
    )
    whisking_r_epochs = average_data_in_epoch(
        sl.motion_energy['rightCamera']['times'], sl.motion_energy['rightCamera']['whiskerMotionEnergy'], sl.trials,
        align_event=align_event, epoch=time_window
    )

    # get root of sum of squares for paw position
    paw_pos_r = sl.pose['rightCamera'][['paw_r_x', 'paw_r_y']].pow(2).sum(axis=1, skipna=False).apply(np.sqrt)
    paw_pos_l = sl.pose['leftCamera'][['paw_r_x', 'paw_r_y']].pow(2).sum(axis=1, skipna=False).apply(np.sqrt)

    paw_l_epochs = average_data_in_epoch(
        sl.pose['leftCamera']['times'], paw_pos_l, sl.trials, align_event=align_event, epoch=time_window
    )
    paw_r_epochs = average_data_in_epoch(
        sl.pose['rightCamera']['times'], paw_pos_r, sl.trials, align_event=align_event, epoch=time_window
    )

    # motor signals has shape (n_trials, n_regressors)
    motor_signals = np.c_[
        wheel_epochs, lick_epochs, nose_epochs, whisking_l_epochs, whisking_r_epochs, paw_l_epochs, paw_r_epochs
    ]
    # normalize the motor signals
    motor_signals = stats.zscore(motor_signals, axis=0, nan_policy='omit')
    return motor_signals


def compute_motor_prediction(one, eid, target, time_window):
    motor_signals = prepare_motor(one, eid, time_window=time_window)
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1]).fit(motor_signals, target)
    motor_prediction = clf.predict(motor_signals)

    return motor_prediction


def prepare_behavior(
        one, session_id, subject, pseudo_ids=None, output_dir=None, model='optBay', target='pLeft',
        align_event='stimOn_times', time_window=(-0.6, -0.1), stage_only=False, integration_test=False,
):
    if pseudo_ids is None:
        pseudo_ids = [-1]  # -1 is always the actual session

    # Load trials
    sl = SessionLoader(one, session_id)
    sl.load_trials()

    # Compute trials mask and intervals from original trials
    trials_mask = compute_mask(sl.trials, align_time=align_event, time_window=time_window)
    intervals = np.vstack([sl.trials[align_event] + time_window[0], sl.trials[align_event] + time_window[1]]).T
    if sum(trials_mask) <= MIN_BEHAV_TRIALS:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {MIN_BEHAV_TRIALS}.")
    if stage_only:
        return None, None, trials_mask, intervals, None

    behavior_path = output_dir.joinpath('behavior') if output_dir else Path.cwd().joinpath('behavior')
    # Train model if not trained already, optimal Bayesian and oracle (None) don't need to be trained
    if model not in ['oracle', 'optBay']:
        side, stim, act, _ = format_data(sl.trials)
        stimuli, actions, stim_side = format_input([stim], [act], [side])
        behavior_model = model_name2class[model](behavior_path, session_id, subject, actions, stimuli, stim_side,
                                                 single_zeta=True)
        istrained, _ = check_bhv_fit_exists(subject, model, session_id, behavior_path, single_zeta=True)
        if not istrained:
            behavior_model.load_or_train(remove_old=False)

    all_targets = []
    all_trials = []
    # For all sessions (pseudo and or actual session) compute the behavioral targets
    for pseudo_id in pseudo_ids:
        if pseudo_id == -1:  # this is the actual session
            all_trials.append(sl.trials)
            all_targets.append(compute_beh_target(sl.trials, session_id, subject, model, target, behavior_path))
        else:
            if integration_test:  # for reproducing the test results we need to fix a seed in this case
                np.random.seed(pseudo_id)
            control_trials = generate_null_distribution_session(sl.trials, session_id, subject, model, behavior_path)
            all_trials.append(control_trials)
            all_targets.append(compute_beh_target(control_trials, session_id, subject, model, target, behavior_path))

    # mask all targets
    all_targets = [target[trials_mask] for target in all_targets]
    # compute derivative if indicated
    if DECODE_DERIVATIVE:
        all_targets = [derivative(target) for target in all_targets]
    # add motor residual to regressors if indicated
    if MOTOR_RESIDUAL:
        motor_predictions = [compute_motor_prediction(session_id, target, time_window) for target in all_targets]
        all_targets = [target - motor for target, motor in zip(all_targets, motor_predictions)]

    # Compute neurometrics if indicated
    if COMPUTE_NEUROMETRIC:
        all_neurometrics = []
        for i in range(len(pseudo_ids)):
            neurometrics = compute_neurometric_prior(all_trials[i], session_id, subject, model, behavior_path)
            all_neurometrics.append(neurometrics[trials_mask].reset_index(drop=True))
    else:
        all_neurometrics = None

    return all_trials, all_targets, trials_mask, intervals, all_neurometrics



# def prepare_widefield():
#     from prior_localization.decoding.functions.process_inputs import get_bery_reg_wfi
#
#     beryl_reg = get_bery_reg_wfi(neural_dict, **kwargs)
#     reg_mask = select_widefield_imaging_regions(neural_dict, region, **kwargs)
#     msub_binned = preprocess_widefield_imaging(neural_dict, reg_mask, **kwargs)
#     n_units = np.sum(reg_mask)
#
#
#     return hemisphere

