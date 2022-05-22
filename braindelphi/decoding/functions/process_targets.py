from pathlib import Path
from behavior_models.models import utils as mut
import os
import numpy as np


def check_bhv_fit_exists(subject, model, eids, resultpath):
    '''
    subject: subject_name
    eids: sessions on which the model was fitted
    check if the behavioral fits exists
    return Bool and filename
    '''
    if model not in modeldispatcher.keys():
        raise KeyError('Model is not an instance of a model from behavior_models')
    path_results_mouse = 'model_%s_' % modeldispatcher[model]
    trunc_eids = [eid.split('-')[0] for eid in eids]
    filen = mut.build_path(path_results_mouse, trunc_eids)
    subjmodpath = Path(resultpath).joinpath(Path(subject))
    fullpath = subjmodpath.joinpath(filen)
    return os.path.exists(fullpath), fullpath

def fit_load_bhvmod(target,
                    subject,
                    savepath,
                    eids_train,
                    behavior_data_train,
                    modeltype,
                    remove_old=False,
                    beh_data_test=None):
    '''
    load/fit a behavioral model to compute target on a single session
    Params:
        eids_train: list of eids on which we train the network
        eid_test: eid on which we want to compute the target signals, only one string
        beh_data_test: if you have to launch the model on beh_data_test.
                       if beh_data_test is explicited, the eid_test will not be considered
        target can be pLeft or signcont. If target=pLeft, it will return the prior predicted by modeltype
                                         if modetype=None, then it will return the actual pLeft (.2, .5, .8)
    '''

    # check if is trained
    istrained, fullpath = check_bhv_fit_exists(subject, modeltype, eids_train, savepath)

    if target == 'signcont':
        if 'signedContrast' in beh_data_test.keys():
            out = beh_data_test['signedContrast']
        else:
            out = np.nan_to_num(beh_data_test['contrastLeft']) - np.nan_to_num(
                beh_data_test['contrastRight'])
        return out
    if target == 'choice':
        return np.array(beh_data_test['choice'])
    if target == 'feedback':
        return np.array(beh_data_test['feedbackType'])
    elif (target == 'pLeft') and (modeltype is None):
        return np.array(beh_data_test['probabilityLeft'])
    elif (target
          == 'pLeft') and (modeltype is optimal_Bayesian):  # bypass fitting and generate priors
        side, stim, act, _ = mut.format_data(beh_data_test)
        if isinstance(side, np.ndarray) and isinstance(act, np.ndarray):
            signal = optimal_Bayesian(act, stim, side)
        else:
            signal = optimal_Bayesian(act.values, stim, side.values)
        return signal.numpy().squeeze()

    if (not istrained) and (target != 'signcont') and (modeltype is not None):
        datadict = {'stim_side': [], 'actions': [], 'stimuli': []}
        for eid in eids_train:
            subdf = behavior_data_train[behavior_data_train.eid == eid]
            stim_side, stimuli, actions = subdf.stim_side.values, subdf.signedContrast.values, subdf.choice.values
            datadict['stim_side'].append(stim_side)
            datadict['stimuli'].append(stimuli)
            datadict['actions'].append(actions)
        stimuli, actions, stim_side = mut.format_input(datadict['stimuli'], datadict['actions'],
                                                       datadict['stim_side'])
        eids = np.array(eids_train)
        model = modeltype(savepath, eids, subject, actions, stimuli, stim_side)
        model.load_or_train(remove_old=remove_old)
    elif (target != 'signcont') and (modeltype is not None):
        model = modeltype(savepath,
                          eids_train,
                          subject,
                          actions=None,
                          stimuli=None,
                          stim_side=None)
        model.load_or_train(loadpath=str(fullpath))

    # compute signal
    stim_side, stimuli, actions, _ = mut.format_data(beh_data_test)
    stimuli, actions, stim_side = mut.format_input([stimuli], [actions], [stim_side])
    signal = model.compute_signal(signal='prior' if target == 'pLeft' else target,
                                  act=actions,
                                  stim=stimuli,
                                  side=stim_side)['prior' if target == 'pLeft' else target]

    return signal.squeeze()

possible_targets = ['choice', 'feedback', 'signcont', 'pLeft']

def compute_beh_target(target,
                   subject,
                   eids_train,
                   eid_test,
                   savepath,
                   binarization_value,
                   modeltype,
                   one=None,
                   behavior_data_train=None,
                   beh_data_test=None):
    """
    Computes regression target for use with regress_target, using subject, eid, and a string
    identifying the target parameter to output a vector of N_trials length containing the target

    Parameters
    ----------
    target : str
        String in ['prior', 'prederr', 'signcont'], indication model-based prior, prediction error,
        or simple signed contrast per trial
    subject : str
        Subject identity in the IBL database, e.g. KS022
    eids_train : list of str
        list of UUID identifying sessions on which the model is trained.
    eids_test : str
        UUID identifying sessions on which the target signal is computed
    savepath : str
        where the beh model outputs are saved
    behmodel : str
        behmodel to use
    pseudo : bool
        Whether or not to compute a pseudosession result, rather than a real result.
    modeltype : behavior_models model object
        Instantiated object of behavior models. Needs to be instantiated for pseudosession target
        generation in the case of a 'prior' or 'prederr' target.
    beh_data : behavioral data feed to the model when using pseudo-sessions

    Returns
    -------
    pandas.Series
        Pandas series in which index is trial number, and value is the target
    """
    if target not in possible_targets:
        raise ValueError('target should be in {}'.format(possible_targets))

    tvec = fit_load_bhvmod(target,
                           subject,
                           savepath.as_posix() + '/',
                           eids_train,
                           eid_test,
                           remove_old=False,
                           modeltype=modeltype,
                           one=one,
                           behavior_data_train=behavior_data_train,
                           beh_data_test=beh_data_test)

    if binarization_value is not None:
        tvec = (tvec > binarization_value) * 1

    return tvec


def optimal_Bayesian(act, stim, side):
    '''
    Generates the optimal prior
    Params:
        act (array of shape [nb_sessions, nb_trials]): action performed by the mice of shape
        side (array of shape [nb_sessions, nb_trials]): stimulus side (-1 (right), 1 (left)) observed by the mice
    Output:
        prior (array of shape [nb_sessions, nb_chains, nb_trials]): prior for each chain and session
    '''
    act = torch.from_numpy(act)
    side = torch.from_numpy(side)
    lb, tau, ub, gamma = 20, 60, 100, 0.8
    nb_blocklengths = 100
    nb_typeblocks = 3
    eps = torch.tensor(1e-15)

    alpha = torch.zeros([act.shape[-1], nb_blocklengths, nb_typeblocks])
    alpha[0, 0, 1] = 1
    alpha = alpha.reshape(-1, nb_typeblocks * nb_blocklengths)
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
    l = torch.cat(
        (torch.unsqueeze(hazard, -1),
         torch.cat((torch.diag(1 - hazard[:-1]), torch.zeros(nb_blocklengths - 1)[None]), axis=0)),
        axis=-1)  # l_{t-1}, l_t
    transition = eps + torch.transpose(l[:, :, None, None] * b[None], 1, 2).reshape(
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
        if i_trial > 0:
            alpha[i_trial] = torch.sum(torch.unsqueeze(h, -1) * transition, axis=0) * to_update[i_trial - 1] \
                             + alpha[i_trial - 1] * (1 - to_update[i_trial - 1])
        h = alpha[i_trial] * lks[i_trial].repeat(nb_blocklengths)
        h = h / torch.unsqueeze(torch.sum(h, axis=-1), -1)

    predictive = torch.sum(alpha.reshape(-1, nb_blocklengths, nb_typeblocks), 1)
    Pis = predictive[:, 0] * gamma + predictive[:, 1] * 0.5 + predictive[:, 2] * (1 - gamma)

    return 1 - Pis

modeldispatcher = {
    expSmoothing_prevAction: expSmoothing_prevAction.name,
    expSmoothing_stimside: expSmoothing_stimside.name,
    optimal_Bayesian: 'optBay',
    None: 'oracle'
}


def get_target_pLeft(nb_trials,
                     nb_sessions,
                     take_out_unbiased,
                     bin_size_kde,
                     subjModel=None,
                     antithetic=True):
    # if subjModel is empty, compute the optimal Bayesian prior
    if subjModel is not None:
        istrained, fullpath = check_bhv_fit_exists(subjModel['subject'], subjModel['modeltype'],
                                                   subjModel['subjeids'],
                                                   subjModel['modelfit_path'].as_posix() + '/')
        if not istrained:
            raise ValueError('Something is wrong. The model should be trained by this line')
        model = subjModel['modeltype'](subjModel['modelfit_path'].as_posix() + '/',
                                       subjModel['subjeids'],
                                       subjModel['subject'],
                                       actions=None,
                                       stimuli=None,
                                       stim_side=None)
        model.load_or_train(loadpath=str(fullpath))
    else:
        model = None
    contrast_set = np.array([0., 0.0625, 0.125, 0.25, 1])
    target_pLeft = []
    for _ in np.arange(nb_sessions):
        if model is None or not subjModel['use_imposter_session_for_balancing']:
            pseudo_trials = pd.DataFrame()
            pseudo_trials['probabilityLeft'] = generate_pseudo_blocks(nb_trials)
            for i in range(pseudo_trials.shape[0]):
                position = _draw_position([-1, 1], pseudo_trials['probabilityLeft'][i])
                contrast = _draw_contrast(contrast_set, 'uniform')
                if position == -1:
                    pseudo_trials.loc[i, 'contrastLeft'] = contrast
                elif position == 1:
                    pseudo_trials.loc[i, 'contrastRight'] = contrast
                pseudo_trials.loc[i, 'stim_side'] = position
            pseudo_trials['signed_contrast'] = pseudo_trials['contrastRight']
            pseudo_trials.loc[pseudo_trials['signed_contrast'].isnull(),
                              'signed_contrast'] = -pseudo_trials['contrastLeft']
            pseudo_trials['choice'] = np.NaN  # choice padding
        else:
            pseudo_trials = generate_imposter_session(subjModel['imposterdf'],
                                                      subjModel['eid'],
                                                      nb_trials,
                                                      nbSampledSess=10)
        side, stim, act, _ = mut.format_data(pseudo_trials)
        if model is None:
            msub_pseudo_tvec = optimal_Bayesian(act.values, stim, side.values)
        elif not subjModel['use_imposter_session_for_balancing']:
            arr_params = model.get_parameters(parameter_type='posterior_mean')[None]
            valid = np.ones([1, pseudo_trials.index.size], dtype=bool)
            stim, act, side = mut.format_input([stim], [act.values], [side.values])
            act_sim, stim, side = model.simulate(arr_params,
                                                 stim,
                                                 side,
                                                 torch.from_numpy(valid),
                                                 nb_simul=10,
                                                 only_perf=False)
            act_sim = act_sim.squeeze().T
            stim = torch.tile(stim.squeeze()[None], (act_sim.shape[0], 1))
            side = torch.tile(side.squeeze()[None], (act_sim.shape[0], 1))
            msub_pseudo_tvec = model.compute_signal(
                signal=('prior' if subjModel['target'] == 'pLeft' else subjModel['target']),
                act=act_sim,
                stim=stim,
                side=side)
            msub_pseudo_tvec = msub_pseudo_tvec['prior'].T
        else:
            stim, act, side = mut.format_input([stim], [act.values], [side.values])
            msub_pseudo_tvec = model.compute_signal(
                signal=('prior' if subjModel['target'] == 'pLeft' else subjModel['target']),
                act=act,
                stim=stim,
                side=side)
            msub_pseudo_tvec = msub_pseudo_tvec['prior' if subjModel['target'] ==
                                                'pLeft' else subjModel['target']]
        if take_out_unbiased:
            target_pLeft.append(
                msub_pseudo_tvec[(pseudo_trials.probabilityLeft != 0.5).values].ravel())
        else:
            target_pLeft.append(msub_pseudo_tvec.ravel())
    target_pLeft = np.concatenate(target_pLeft)
    if antithetic:
        target_pLeft = np.concatenate([target_pLeft, 1 - target_pLeft])
    out = np.histogram(target_pLeft,
                       bins=(np.arange(-bin_size_kde, 1 + bin_size_kde / 2., bin_size_kde) +
                             bin_size_kde / 2.),
                       density=True)
    return out, target_pLeft

