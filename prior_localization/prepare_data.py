import numpy as np
from pathlib import Path

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
from prior_localization.settings import COMPUTE_NEUROMETRIC, DECODE_DERIVATIVE, MOTOR_RESIDUAL, MIN_BEHAV_TRIALS
from prior_localization.functions.neurometric import compute_neurometric_prior


def prepare_behavior(
        one, session_id, subject, pseudo_ids=None, output_dir=None, model=None, target='pLeft',
        align_event='stimOn_times', time_window=(-0.6, -0.1), stage_only=False, integration_test=False,
):
    if pseudo_ids is None:
        pseudo_ids = [-1]  # -1 is always the actual session

    # Load trials
    sl = SessionLoader(one, session_id)
    sl.load_trials()
    if stage_only:
        return

    behavior_path = output_dir.joinpath('behavior') if output_dir else Path.cwd().joinpath('behavior')
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

    # Compute trials mask and intervals from original trials
    trials_mask = compute_mask(sl.trials, align_time=align_event, time_window=time_window)
    intervals = np.vstack([sl.trials[align_event] + time_window[0], sl.trials[align_event] + time_window[1]]).T
    if sum(trials_mask) <= MIN_BEHAV_TRIALS:
        return None, None, None, None, None

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
        all_neurometrics = [None] * len(pseudo_ids)

    return all_trials, all_targets, all_neurometrics, trials_mask, intervals


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

