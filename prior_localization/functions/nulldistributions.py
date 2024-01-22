import numpy as np
from behavior_models.utils import format_input as mut_format_input
from brainbox.task.closed_loop import generate_pseudo_session
from behavior_models.models import ActionKernel, StimulusKernel
from prior_localization.functions.behavior_targets import optimal_Bayesian, check_bhv_fit_exists


model_name2class = {
    "optBay": optimal_Bayesian,
    "actKernel": ActionKernel,
    "stimKernel": StimulusKernel,
    "oracle": None
}


def generate_null_distribution_session(trials_df, session_id, subject, model, behavior_path):
    if "signedContrast" in trials_df.columns:
        out = np.nan_to_num(trials_df.contrastLeft.values) - np.nan_to_num(
            trials_df.contrastRight.values
        )
        assert np.all(np.nan_to_num(trials_df.signedContrast.values) == out)
    pseudosess = generate_pseudo_session(trials_df, generate_choices=False)
    if model == "actKernel":
        subjModel = {"eid": session_id, "subject": subject, "modeltype": model, "behfit_path": behavior_path}
        pseudosess["choice"] = generate_choices(pseudosess, trials_df, subjModel,)
        pseudosess["feedbackType"] = np.where(pseudosess["choice"] == pseudosess["stim_side"], 1, -1)
    else:
        pseudosess["choice"] = trials_df.choice
    return pseudosess


def generate_null_distribution_session_imposter(trials_df, session_id, imposter_df):
    # copy since we will modify below
    df = imposter_df.copy()
    # remove current eid from imposter sessions
    df_clean = df[df.eid != session_id].reset_index()
    # randomly select imposter trial to start sequence
    n_trials = trials_df.index.size
    total_imposter_trials = df_clean.shape[0]
    idx_beg = np.random.choice(total_imposter_trials - n_trials)
    control_trials = df_clean.iloc[idx_beg:idx_beg + n_trials]
    return control_trials


def generate_choices(pseudosess, trials_df, subjModel):

    istrained, fullpath = check_bhv_fit_exists(subjModel["subject"], subjModel["modeltype"], subjModel["eid"],
                                               subjModel["behfit_path"], single_zeta=True)

    if not istrained:
        raise ValueError("Something is wrong. The model should be trained by this line")

    modelclass = model_name2class[subjModel["modeltype"]]
    model = modelclass(subjModel["behfit_path"], subjModel["eid"], subjModel["subject"],
                       actions=None, stimuli=None, stim_side=None, single_zeta=True)
    model.load_or_train(loadpath=str(fullpath))
    arr_params = model.get_parameters(parameter_type="posterior_mean")[None]

    stim, _, side = mut_format_input(
        [pseudosess.signed_contrast.values],
        [trials_df.choice.values],
        [pseudosess.stim_side.values],
    )
    act_sim, stim, side = model.simulate(arr_params, stim.squeeze(), side.squeeze(), nb_simul=1, only_perf=False)
    act_sim = np.array(act_sim.squeeze().T, dtype=np.int64)

    return act_sim
