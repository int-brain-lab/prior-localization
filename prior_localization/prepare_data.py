import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import RidgeCV

import wfield
from ibllib.atlas import BrainRegions
from brainbox.io.one import SessionLoader
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.behavior.dlc import get_licks

from brainwidemap.bwm_loading import load_good_units, merge_probes
from behavior_models.utils import format_data, format_input
from behavior_models.models import ActionKernel, StimulusKernel

from prior_localization.functions.behavior_targets import optimal_Bayesian, compute_beh_target
from prior_localization.functions.utils import compute_mask, check_bhv_fit_exists, average_data_in_epoch
from prior_localization.functions.nulldistributions import generate_null_distribution_session
from prior_localization.functions.neurometric import compute_neurometric_prior

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
        if regions == 'single_regions':
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


def prepare_widefield(one, eid, corrected=True):
    if corrected:
        SVT = one.load_dataset(eid, 'widefieldSVT.haemoCorrected.npy')
    else:
        SVT = one.load_dataset(eid, 'widefieldSVT.uncorrected.npy')
    U = one.load_dataset(eid, 'widefieldU.images.npy')
    times = one.load_dataset(eid, 'imaging.times.npy')
    channels = one.load_dataset(eid, 'imaging.imagingLightSource.npy')
    channel_info = one.load_dataset(eid, 'imagingLightSource.properties.htsv', download_only=True)
    channel_info = pd.read_csv(channel_info)
    lmark_file = one.load_dataset(eid, 'widefieldLandmarks.dorsalCortex.json', download_only=True)
    landmarks = wfield.load_allen_landmarks(lmark_file)
    # If haemocorrected need to take timestamps that correspond to functional channel
    functional_channel = 470
    functional_chn = channel_info.loc[channel_info['wavelength'] == functional_channel]['channel_id'].values[0]
    times = times[channels == functional_chn]

    # Align the image stack to Allen reference
    stack = wfield.SVDStack(U, SVT)
    stack.set_warped(True, M=landmarks['transform'])

    # Load in the Allen atlas
    atlas, area_names, mask = wfield.atlas_from_landmarks_file(lmark_file, do_transform=False)
    ccf_regions, _, _ = wfield.allen_load_reference('dorsal_cortex')
    ccf_regions = ccf_regions[['acronym', 'name', 'label']]

    # Create a 3d mask of the brain outline
    mask3d = wfield.mask_to_3d(mask, shape=np.roll(stack.U_warped.shape, 1))
    # Set pixels outside the brain outline to zero
    stack.U_warped[~mask3d.transpose([1, 2, 0])] = 0
    # Do the same to the Allen image
    atlas[~mask] = 0

    # Downsample the images
    downsampled_atlas = downsample_atlas(atlas, pixelSize=10)
    downsampled_stack = spatial_down_sample(stack, pixelSize=10)


    trials = one.load_object(eid, 'trials')
    trials = trials.to_df()
    frames = pd.DataFrame()
    for key in trials.keys():
        if 'times' in key:
            idx = np.searchsorted(times, trials[key].values).astype(np.float64)
            idx[np.isnan(trials[key].values)] = np.nan
            frames[key] = idx
        else:
            frames[key] = trials[key].values

    # remove last trial as this is detected wrong
    frames = frames[:-1]

    neural_activity = {}
    neural_activity['activity'] = downsampled_stack
    neural_activity['timings'] = frames
    neural_activity['regions'] = downsampled_atlas
    neural_activity['atlas'] = ccf_regions


    return neural_activity


# Taken from https://github.com/cskrasniak/wfield/blob/master/wfield/analyses.py
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


def prepare_motor(one, session_id, align_event='stimOn_times', time_window=(-0.6, -0.1), lick_bins=0.02):

    # Initiate session loader, load trials, wheel, motion energy and pose estimates
    sl = SessionLoader(one, session_id)
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
    binned_licks_times = np.arange(lick_times[0], lick_times[-1] + lick_bins, lick_bins)
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


def compute_motor_prediction(one, eid, target_data, trials_mask, time_window):
    motor_signals = prepare_motor(one, eid, time_window=time_window)
    trials_mask = trials_mask & ~np.any(np.isnan(motor_signals), axis=1)
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1]).fit(motor_signals[trials_mask], target_data[trials_mask])
    motor_prediction = np.full_like(trials_mask, np.nan)
    motor_prediction[trials_mask] = clf.predict(motor_signals[trials_mask])

    return motor_prediction


def prepare_behavior(
        one, session_id, subject, pseudo_ids=None, output_dir=None, model='optBay', target='pLeft',
        align_event='stimOn_times', time_window=(-0.6, -0.1), min_trials=150, motor_residual=False,
        compute_neurometrics=False, stage_only=False, integration_test=False,
):
    if pseudo_ids is None:
        pseudo_ids = [-1]  # -1 is always the actual session

    # Load trials
    sl = SessionLoader(one, session_id)
    sl.load_trials()

    # Compute trials mask and intervals from original trials
    trials_mask = compute_mask(sl.trials, align_time=align_event, time_window=time_window)
    intervals = np.vstack([sl.trials[align_event] + time_window[0], sl.trials[align_event] + time_window[1]]).T
    if sum(trials_mask) <= min_trials:
        raise ValueError(f"Session {session_id} has {sum(trials_mask)} good trials, less than {min_trials}.")
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

    # add motor residual to regressors if indicated
    if motor_residual:
        motor_predictions = [compute_motor_prediction(one, session_id, t, trials_mask, time_window) for t in all_targets]
        all_targets = [t - motor for t, motor in zip(all_targets, motor_predictions)]
        # update trials mask with possible nans from motor prediction
        trials_mask = trials_mask & ~np.isnan(motor_predictions[0])

    # Compute neurometrics if indicated
    if compute_neurometrics:
        all_neurometrics = []
        for i in range(len(pseudo_ids)):
            neurometrics = compute_neurometric_prior(all_trials[i], session_id, subject, model, behavior_path)
            all_neurometrics.append(neurometrics[trials_mask].reset_index(drop=True))
    else:
        all_neurometrics = None

    return all_trials, all_targets, trials_mask, intervals, all_neurometrics


def prepare_pupil(one, session_id, time_window=(-0.6, -0.1), align_event='stimOn_times', camera='left'):
    # Load the trials data
    sl = SessionLoader(one, session_id)
    sl.load_trials()
    # TODO: replace this with SessionLoader ones that loads lightning pose
    pupil_data = one.load_object(session_id, f'{camera}Camera', attribute=['lightningPose', 'times'])

    # Calculate the average x position of the pupil in the time windows
    epochs_x = average_data_in_epoch(
        pupil_data['times'], pupil_data['lightningPose']['pupil_top_r_x'].values, sl.trials, align_event=align_event,
        epoch=time_window
    )
    # Calculate the average y position (first average between bottom and top) in each time window
    epochs_y = average_data_in_epoch(
        pupil_data['times'], (pupil_data['lightningPose'][['pupil_bottom_r_y', 'pupil_top_r_y']].sum(axis=1) / 2),
        sl.trials, align_event='stimOn_times', epoch=time_window
    )
    # Return concatenated x and y
    return np.c_[epochs_x, epochs_y]


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

