import numpy as np
import torch

from behavior_models.utils import format_data, format_input
from behavior_models.models import ActionKernel, StimulusKernel

from prior_localization.functions.utils import check_bhv_fit_exists
from prior_localization.params import BINARIZATION_VALUE


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
        String in ['optBay', 'actKernel', 'stimKernel', 'oracle'], indication model-based prior, prediction error,
    target : str
        String in ['prior', 'prederr', 'signcont'], indication model-based prior, prediction error,
        or simple signed contrast per trial
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

    istrained, fullpath = check_bhv_fit_exists(subject, model, session_id, behavior_path, single_zeta=True)

    if target in ['signcont', 'strengthcont']:
        if 'signedContrast' in trials_df.keys():
            out = trials_df['signedContrast'].values
        else:
            out = np.nan_to_num(trials_df.contrastLeft) - np.nan_to_num(trials_df.contrastRight)
        if target == 'signcont':
            return out
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

    if (not istrained) and (target != 'signcont') and (model != 'oracle'):
        side, stim, act, _ = format_data(trials_df)
        stimuli, actions, stim_side = format_input(stim, act, side)
        model = model_name2class[model](behavior_path, session_id, subject, actions, stimuli, stim_side,
                                        single_zeta=True)
        model.load_or_train(remove_old=remove_old)
    elif (target != 'signcont') and (model != 'oracle'):
        model = model_name2class[model](behavior_path, session_id, subject, actions=None, stimuli=None,
                                        stim_side=None, single_zeta=True)
        model.load_or_train(loadpath=str(fullpath))

    # compute signal
    stim_side, stimuli, actions, _ = format_data(trials_df)
    stimuli, actions, stim_side = format_input([stimuli], [actions], [stim_side])
    signal = model.compute_signal(signal='prior' if target == 'pLeft' else target, act=actions, stim=stimuli,
                                  side=stim_side)['prior' if target == 'pLeft' else target]
    tvec = signal.squeeze()
    if BINARIZATION_VALUE is not None:
        tvec = (tvec > BINARIZATION_VALUE) * 1

    return tvec
