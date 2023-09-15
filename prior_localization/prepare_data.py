import logging
import numpy as np
from pathlib import Path
from scipy import stats
import pandas as pd

import wfield
import sklearn.linear_model as sklm

from ibllib.atlas import BrainRegions
from brainbox.io.one import SessionLoader
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.behavior.dlc import get_licks

from brainwidemap.bwm_loading import load_good_units, merge_probes
from behavior_models.utils import format_data, format_input
from behavior_models.models import ActionKernel, StimulusKernel

from prior_localization.functions.behavior_targets import optimal_Bayesian, compute_beh_target
from prior_localization.functions.utils import (
    check_bhv_fit_exists, average_data_in_epoch, check_config, downsample_atlas, spatial_down_sample
)
from prior_localization.functions.nulldistributions import generate_null_distribution_session
from prior_localization.functions.neurometric import compute_neurometric_prior


logger = logging.getLogger('prior_localization')

model_name2class = {
    "optBay": optimal_Bayesian,
    "actKernel": ActionKernel,
    "stimKernel": StimulusKernel,
    "oracle": None
}

config = check_config()


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
        if regions in config['region_defaults'].keys():  # if regions is a key into the regions_default config dict
            regions = config['region_defaults'][regions]
        elif regions == 'single_regions':  # if single_regions, make a list of lists to decode every region separately
            regions = [[b] for b in np.unique(beryl_regions) if b not in ['root', 'void']]
        elif regions == 'all_regions':  # if all regions make a list of a single list to decode all regions together
            regions = [[b for b in np.unique(beryl_regions) if b not in ['root', 'void']]]
        else:
            regions = [regions]  # if one region is given, put it into a list
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


def prepare_behavior(
        session_id, subject, trials_df, trials_mask, pseudo_ids=None, output_dir=None, model='optBay',
        target='pLeft', compute_neurometrics=False, integration_test=False,
):
    if pseudo_ids is None:
        pseudo_ids = [-1]  # -1 is always the actual session

    behavior_path = output_dir.joinpath('behavior') if output_dir else Path.cwd().joinpath('behavior')
    # Train model if not trained already, optimal Bayesian and oracle (None) don't need to be trained
    if model not in ['oracle', 'optBay']:
        side, stim, act, _ = format_data(trials_df)
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
            all_trials.append(trials_df)
            all_targets.append(compute_beh_target(trials_df, session_id, subject, model, target, behavior_path))
        else:
            if integration_test:  # for reproducing the test results we need to fix a seed in this case
                np.random.seed(pseudo_id)
            control_trials = generate_null_distribution_session(trials_df, session_id, subject, model, behavior_path)
            all_trials.append(control_trials)
            all_targets.append(compute_beh_target(control_trials, session_id, subject, model, target, behavior_path))

    # Compute neurometrics if indicated
    if compute_neurometrics:
        all_neurometrics = []
        for i in range(len(pseudo_ids)):
            neurometrics = compute_neurometric_prior(all_trials[i], session_id, subject, model, behavior_path)
            all_neurometrics.append(neurometrics[trials_mask].reset_index(drop=True))
    else:
        all_neurometrics = None

    return all_trials, all_targets, trials_mask, all_neurometrics


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


def prepare_widefield(
        one, session_id, hemisphere, regions, align_times, frame_window, functional_channel=470, stage_only=False
):
    # Load the haemocorrected temporal components and projected spatial components
    SVT = one.load_dataset(session_id, 'widefieldSVT.haemoCorrected.npy')
    U = one.load_dataset(session_id, 'widefieldU.images.npy')

    # Load the channel information
    channels = one.load_dataset(session_id, 'imaging.imagingLightSource.npy')
    channel_info = one.load_dataset(session_id, 'imagingLightSource.properties.htsv', download_only=True)
    channel_info = pd.read_csv(channel_info)
    lmark_file = one.load_dataset(session_id, 'widefieldLandmarks.dorsalCortex.json', download_only=True)
    landmarks = wfield.load_allen_landmarks(lmark_file)

    # Load the timestamps and get those that correspond to functional channel
    times = one.load_dataset(session_id, 'imaging.times.npy')
    functional_chn = channel_info.loc[channel_info['wavelength'] == functional_channel]['channel_id'].values[0]
    times = times[channels == functional_chn]

    # If we are only staging data, end here
    if stage_only:
        return None, None

    # Align the image stack to Allen reference
    stack = wfield.SVDStack(U, SVT)
    stack.set_warped(True, M=landmarks['transform'])

    # Load in the Allen atlas, mask and downsample
    atlas, area_names, mask = wfield.atlas_from_landmarks_file(lmark_file, do_transform=False)
    ccf_regions, _, _ = wfield.allen_load_reference('dorsal_cortex')
    ccf_regions = ccf_regions[['acronym', 'name', 'label']]
    atlas[~mask] = 0
    downsampled_atlas = downsample_atlas(atlas, pixelSize=10)

    # Create a 3d mask to mask image stack, also downsample
    mask3d = wfield.mask_to_3d(mask, shape=np.roll(stack.U_warped.shape, 1))
    stack.U_warped[~mask3d.transpose([1, 2, 0])] = 0
    downsampled_stack = spatial_down_sample(stack, pixelSize=10)

    # Find the frames corresponding to the trials times of the align event
    frames = np.searchsorted(times, align_times).astype(np.float64)
    frames[np.isnan(align_times)] = np.nan
    # Get the frame window intervals around the align event frame, from which to draw data
    intervals = np.sort(frames[:, None] + np.arange(frame_window[0], frame_window[1] + 1), axis=1).astype(int).squeeze()

    # Get the brain regions for now disregarding hemisphere
    atlas_regions = ccf_regions.acronym[ccf_regions.label.isin(np.unique(downsampled_atlas))].values
    if isinstance(regions, str):
        if regions in config['region_defaults'].keys():  # if regions is a key into the regions_default config dict
            regions = config['region_defaults'][regions]
        elif regions == 'single_regions':  # if single_regions, make a list of lists to decode every region separately
            regions = [[r] for r in atlas_regions]
        elif regions == 'all_regions':  # if all regions make a list of a single list to decode all regions together
            regions = [atlas_regions]
        else:
            regions = [regions]  # if one region is given, put it into a list
    elif isinstance(regions, list):
        pass

    data_epoch = []
    actual_regions = []
    for region in regions:
        # Translate back into label for masking the atlas
        region_labels = ccf_regions[ccf_regions.acronym.isin(region)].label.values
        # If left hemisphere, these are the correct labels, for right hemisphere need negative labels (both for both)
        if hemisphere == 'right':
            region_labels = -region_labels
        elif sorted(hemisphere) == ['left', 'right']:
            region_labels = np.concatenate((region_labels, -region_labels))
        # Make a region mask, apply it to the actual data and get data in the respective time intervals
        region_mask = np.isin(downsampled_atlas, region_labels)
        masked_data = downsampled_stack[:, region_mask]
        if np.sum(masked_data) == 0:
            print(f"{'_'.join(region)} no pixels in mask, not decoding")
        else:
            binned = np.take(masked_data, intervals, axis=0)
            data_epoch.append(binned)
            actual_regions.append(region)

    return data_epoch, actual_regions


def prepare_widefield_old(old_data, hemisphere, regions, align_event, frame_window):
    """
    Function to load previous version of data (directly given by Chris) for sanity check
    Assumes three files called timings.pqt, activity.npy and regions.npy in old_data
    """

    # Load old data from disk, path given in old_data input, also load allen atlas
    downsampled_stack = np.load(old_data.joinpath('activity.npy'))  # used to be neural_activity['activity']
    frames = pd.read_parquet(old_data.joinpath('timings.pqt'))  # used to be neural_activity['timings']
    downsampled_atlas = np.load(old_data.joinpath('regions.npy'))  # used to be neural_activity['regions']
    ccf_regions, _, _ = wfield.allen_load_reference('dorsal_cortex')
    ccf_regions = ccf_regions[['acronym', 'name', 'label']]  # used to be neural_activity['atlas']

    # From here it is pretty much equivalent with the new prepare_widefield function

    # Get the frame window intervals around the align event frame, from which to draw data
    intervals = np.sort(
        frames[align_event].values[:, None] + np.arange(frame_window[0], frame_window[1] + 1), axis=1
    ).astype(int).squeeze()

    # Get the brain regions for now disregarding hemisphere
    atlas_regions = ccf_regions.acronym[ccf_regions.label.isin(np.unique(downsampled_atlas))].values
    if isinstance(regions, str):
        if regions in config['region_defaults'].keys():  # if regions is a key into the regions_default config dict
            regions = config['region_defaults'][regions]
        elif regions == 'single_regions':  # if single_regions, make a list of lists to decode every region separately
            regions = [[r] for r in atlas_regions]
        elif regions == 'all_regions':  # if all regions make a list of a single list to decode all regions together
            regions = [atlas_regions]
        else:
            regions = [regions]  # if one region is given, put it into a list
    elif isinstance(regions, list):
        pass

    data_epoch = []
    actual_regions = []
    for region in regions:
        # Translate back into label for masking the atlas
        region_labels = ccf_regions[ccf_regions.acronym.isin(region)].label.values
        # If left hemisphere, these are the correct labels, for right hemisphere need negative labels (both for both)
        if hemisphere == 'right':
            region_labels = -region_labels
        elif sorted(hemisphere) == ['left', 'right']:
            region_labels = np.concatenate((region_labels, -region_labels))
        # Make a region mask, apply it to the actual data and get data in the respective time intervals
        region_mask = np.isin(downsampled_atlas, region_labels)
        masked_data = downsampled_stack[:, region_mask]
        if np.sum(masked_data) == 0:
            print(f"{'_'.join(region)} no pixels in mask, not decoding")
        else:
            binned = np.take(masked_data, intervals, axis=0)
            data_epoch.append(binned)
            actual_regions.append(region)

    return data_epoch, actual_regions