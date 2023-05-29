import numpy as np
import torch
from behavior_models.models.utils import format_input as mut_format_input
from brainbox.task.closed_loop import generate_pseudo_session
from prior_pipelines.decoding.functions.process_targets import (
    optimal_Bayesian,
    check_bhv_fit_exists,
)
from sklearn.metrics import mutual_info_score


def generate_null_distribution_session(trials_df, metadata, **kwargs):
    sess_abs_contrast = trials_df.contrastLeft.abs().fillna(
        value=0
    ) + trials_df.contrastRight.abs().fillna(value=0)
    sess_abs_contrast = (
        sess_abs_contrast.replace(1, 4)
        .replace(0.0625, 1)
        .replace(0.125, 2)
        .replace(0.25, 3)
        .values
    )
    if "signedContrast" in trials_df.columns:
        out = np.nan_to_num(trials_df.contrastLeft.values) - np.nan_to_num(
            trials_df.contrastRight.values
        )
        assert np.all(np.nan_to_num(trials_df.signedContrast.values) == out)
    if kwargs["filter_pseudosessions_on_mutualInformation"]:
        while True:
            pseudosess = generate_pseudo_session(trials_df, generate_choices=False)
            pseudo_abs_contrast = (
                pseudosess.signed_contrast.abs()
                .replace(1, 4)
                .replace(0.0625, 1)
                .replace(0.125, 2)
                .replace(0.25, 3)
                .values
            )
            valid_pseudoSess = mutual_info_score(
                pseudo_abs_contrast[1:], pseudo_abs_contrast[:-1]
            ) > mutual_info_score(sess_abs_contrast[1:], sess_abs_contrast[:-1])
            if valid_pseudoSess:
                break
    else:
        pseudosess = generate_pseudo_session(trials_df, generate_choices=False)
    if (
        kwargs["model"] is not None
        and kwargs["model"] != optimal_Bayesian
        and kwargs["model"].name == "actKernel"
    ):
        subjModel = {
            **metadata,
            "modeltype": kwargs["model"],
            "behfit_path": kwargs["behfit_path"],
        }
        pseudosess["choice"] = generate_choices(
            pseudosess,
            trials_df,
            subjModel,
            kwargs["modeldispatcher"],
            kwargs["constrain_null_session_with_beh"],
            kwargs["model_parameters"],
        )
        pseudosess["feedbackType"] = np.where(
            pseudosess["choice"] == pseudosess["stim_side"], 1, -1
        )
    else:
        pseudosess["choice"] = trials_df.choice
    return pseudosess


def generate_choices(
    pseudosess,
    trials_df,
    subjModel,
    modeldispatcher,
    constrain_null_session_with_beh,
    model_parameters=None,
):

    if model_parameters is None:
        istrained, fullpath = check_bhv_fit_exists(
            subjModel["subject"],
            subjModel["modeltype"],
            subjModel["eids_train"],
            subjModel["behfit_path"].as_posix() + "/",
            modeldispatcher,
            single_zeta=True,
        )
    else:
        istrained, fullpath = True, ""

    if not istrained:
        raise ValueError("Something is wrong. The model should be trained by this line")
    model = subjModel["modeltype"](
        subjModel["behfit_path"],
        subjModel["eids_train"],
        subjModel["subject"],
        actions=None,
        stimuli=None,
        stim_side=None,
        single_zeta=True,
    )

    if model_parameters is None:
        model.load_or_train(loadpath=str(fullpath))
        arr_params = model.get_parameters(parameter_type="posterior_mean")[None]
    else:
        arr_params = np.array(list(model_parameters.values()))[None]
    valid = np.ones([1, pseudosess.index.size], dtype=bool)
    stim, _, side = mut_format_input(
        [pseudosess.signed_contrast.values],
        [trials_df.choice.values],
        [pseudosess.stim_side.values],
    )
    act_sim, stim, side = model.simulate(
        arr_params,
        stim.squeeze(),
        side.squeeze(),
        nb_simul=10000 if constrain_null_session_with_beh else 1,
        only_perf=False,
    )
    act_sim = np.array(act_sim.squeeze().T, dtype=np.int64)

    if constrain_null_session_with_beh:
        # behavior on simulations
        perf_1contrast_sims = np.array(
            [
                torch.mean(
                    (torch.from_numpy(a_s) == side.squeeze())[stim.abs().squeeze() == 1]
                    * 1.0
                ).numpy()
                for a_s in act_sim
            ]
        )
        perf_0contrast_sims = np.array(
            [
                torch.mean(
                    (torch.from_numpy(a_s) == side.squeeze())[stim.squeeze() == 0] * 1.0
                ).numpy()
                for a_s in act_sim
            ]
        )
        repBias_sims = np.array([np.mean(a_s[1:] == a_s[:-1]) for a_s in act_sim])

        # behavior of animal
        perf_1contrast = (trials_df.feedbackType.values > 0)[
            (trials_df.contrastRight == 1) + (trials_df.contrastLeft == 1)
        ].mean()
        perf_0contrast = (trials_df.feedbackType.values > 0)[
            (trials_df.contrastRight == 0) + (trials_df.contrastLeft == 0)
        ].mean()
        repBias = np.mean(trials_df.choice.values[1:] == trials_df.choice.values[:-1])

        # distance between both
        distance = (
            (repBias_sims - repBias) ** 2
            + (perf_0contrast_sims - perf_0contrast) ** 2
            + (perf_1contrast_sims - perf_1contrast) ** 2
        )

        act_sim = act_sim[np.argmin(distance)]

    return act_sim

