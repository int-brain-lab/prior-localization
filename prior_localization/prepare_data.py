import numpy as np
import pandas as pd

from ibllib.atlas import BrainRegions
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_good_units, merge_probes
from behavior_models.utils import format_data, format_input

from prior_localization.settings import region_defaults, modeldispatcher
from prior_localization.functions.process_targets import optimal_Bayesian, compute_beh_target
from prior_localization.functions.process_motors import compute_motor_prediction
from prior_localization.functions.utils import compute_mask, derivative
from prior_localization.utils import check_bhv_fit_exists
from prior_localization.functions.nulldistributions import generate_null_distribution_session
from prior_localization.settings import COMPUTE_NEUROMETRIC, DECODE_DERIVATIVE, MOTOR_RESIDUAL, MIN_LEN, MAX_LEN, MIN_RT, MAX_RT, NO_UNBIAS
from prior_localization.functions.neurometric import compute_neurometric_prior


def prepare_behavior(
        one, session_id, subject, output_path, model=None, pseudo_ids=-1,
        target='pLeft', align_event='stimOn_times', time_window=(-0.6, -0.1), stage_only=False
):
    # Checks on pseudo ids
    if 0 in pseudo_ids:
        raise ValueError("pseudo id can only be -1 (actual session) or strictly greater than 0 (pseudo session)")
    if not np.all(np.sort(pseudo_ids) == pseudo_ids):
        raise ValueError("pseudo_ids must be sorted")

    # Load trials
    sl = SessionLoader(one, session_id)
    sl.load_trials()
    if stage_only:
        return

    behavior_path = output_path.joinpath('behavior')
    # Train model if not trained already, optimal Bayesian and oracle (None) don't need to be trained
    if model and model != optimal_Bayesian:
        side, stim, act, _ = format_data(sl.trials)
        stimuli, actions, stim_side = format_input([stim], [act], [side])
        behavior_model = model(behavior_path. session_id, subject, actions, stimuli, stim_side, single_zeta=True)
        istrained, _ = check_bhv_fit_exists(
            subject, model, session_id, behavior_path, modeldispatcher, single_zeta=True,
        )
        if not istrained:
            behavior_model.load_or_train(remove_old=False)

    behavior_targets = compute_beh_target(sl.trials, session_id, subject, model, target, behavior_path)
    mask_target = np.ones(len(behavior_targets), dtype=bool)
    mask = compute_mask(sl.trials, align_event, time_window, min_len=MIN_LEN,
                        max_len=MAX_LEN, no_unbias=NO_UNBIAS, min_rt=MIN_RT, max_rt=MAX_RT) & mask_target

    # get intervals
    intervals = np.vstack([
        sl.trials[align_event] + time_window[0],
        sl.trials[align_event] + time_window[1]
    ]).T

    all_trials = []
    all_targets = []
    # Prepare pseudo sessions if necessary
    for pseudo_id in pseudo_ids:
        # This is the actual session
        if pseudo_id == -1:
            all_trials.append(sl.trials)
            all_targets.append(behavior_targets)
        else:
            control_trials = generate_null_distribution_session(sl.trials, session_id, subject, model, behavior_path)
            control_targets = compute_beh_target(control_trials, session_id, subject, model, target, behavior_path)
            all_trials.append(control_trials)
            all_targets.append(control_targets)

    # Compute prior for neurometric if indicated
    if COMPUTE_NEUROMETRIC:
        all_neurometrics = []
        for trials_df in all_trials:
            all_neurometrics.append(
                compute_neurometric_prior(trials_df.reset_index(), session_id, subject, model, behavior_path)
            )
    else:
        all_neurometrics = [None] * len(all_trials)

    # Derivative of target signal if indicated
    if DECODE_DERIVATIVE:
        all_targets = [derivative(behavior_targets) for behavior_targets in all_targets]

    # Replace target signal by residual of motor prediction if indicated
    if MOTOR_RESIDUAL:
        motor_predictions = [compute_motor_prediction(session_id, behavior_targets, time_window)
                             for behavior_targets in all_targets]
        all_targets = [b - m for b, m in zip(all_targets, motor_predictions)]

    return all_trials, all_targets, all_neurometrics, mask, intervals


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
        return

    # Prepare list of brain regions
    brainreg = BrainRegions()
    beryl_regions = brainreg.acronym2acronym(clusters['acronym'], mapping="Beryl")
    if isinstance(regions, str):
        if regions in region_defaults.keys():
            regions = region_defaults[regions]
        elif regions == 'single_regions':
            regions = [[k] for k in np.unique(beryl_regions) if k not in ['root', 'void']]
        elif regions == 'all_regions':
            regions = [np.unique([r for r in beryl_regions if r not in ['root', 'void']])]
        else:
            regions = [regions]
    elif isinstance(regions, list):
        pass

    binned_spikes = []
    n_units = []
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
            binned = binned.T  # binned is a 2D array
            binned_spikes.append([x[None, :] for x in binned])
            actual_regions.append(region)
            n_units.append(sum(region_mask))
    return binned_spikes, actual_regions, n_units


def prepare_motor():
    # account for sessions without motor

    motor_regressors = cut_behavior(eid,duration =2, lag = -1,query_type='auto') # very large interval
    motor_signals_of_interest = ['licking', 'whisking_l', 'whisking_r', 'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']
    regressors = dict(filter(lambda i:i[0] in motor_signals_of_interest, motor_regressors.items()))
    return regressors

#
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

