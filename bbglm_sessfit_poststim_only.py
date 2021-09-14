"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from one.api import ONE
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
import brainbox.modeling.design_matrix as dm
import brainbox.modeling.linear as lm
import brainbox.modeling.utils as mut
import brainbox.io.one as bbone
from tqdm import tqdm
from copy import deepcopy
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prevAct
from models import utils
from brainbox.task.closed_loop import generate_pseudo_session

one = ONE()
subjects, ins, ins_id, sess_ids, _ = utils.get_bwm_ins_alyx(one)

trials_cache = {}


def fit_session(session_id, kernlen, nbases,
                t_before=0., t_after=0., prior_estimate='charles', max_len=2., probe='probe00',
                contnorm=5., binwidth=0.02, abswheel=False, num_pseudosess=100, progress=True,
                one=one):
    if not abswheel:
        signwheel = True
    else:
        signwheel = False
    trialsdf = bbone.load_trials_df(session_id, maxlen=max_len, t_before=t_before, t_after=t_after,
                                    wheel_binsize=binwidth, ret_abswheel=abswheel,
                                    ret_wheel=signwheel, one=one)
    if prior_estimate == 'charles':
        print('Fitting behavioral esimates...')
        mouse_name = one.get_details(session_id)['subject']
        stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
        mcounter = 0
        for i in range(len(sess_ids)):
            if subjects[i] == mouse_name:
                data = utils.load_session(sess_ids[i])
                if data['choice'] is not None and data['probabilityLeft'][0] == 0.5:
                    stim_side, stimuli, actions, pLeft_oracle = utils.format_data(data)
                    stimuli_arr.append(stimuli)
                    actions_arr.append(actions)
                    stim_sides_arr.append(stim_side)
                    session_uuids.append(sess_ids[i])
                if sess_ids[i] == session_id:
                    j = mcounter
                mcounter += 1
        # format data
        stimuli, actions, stim_side = utils.format_input(
            stimuli_arr, actions_arr, stim_sides_arr)
        session_uuids = np.array(session_uuids)
        model = exp_prevAct('./results/inference/', session_uuids,
                            mouse_name, actions, stimuli, stim_side)
        model.load_or_train(remove_old=False)
        # compute signals of interest
        signals = model.compute_signal(signal=['prior', 'prediction_error', 'score'],
                                       verbose=False)
        if len(signals['prior'].shape) == 1:
            trialsdf['prior'] = signals['prior'][trialsdf.index]
        else:
            trialsdf['prior'] = signals['prior'][j, trialsdf.index]
        trialsdf['prior_last'] = pd.Series(np.roll(trialsdf['prior'], 1), index=trialsdf.index)
        fitinfo = trialsdf.copy()
    elif prior_estimate is None:
        fitinfo = trialsdf.copy()
    else:
        raise NotImplementedError('\'prior_estimate\' must be either None or charles')

    if kernlen > 0.1:
        raise ValueError('Model is only fit on a 100ms window. Kernel length can\'t be longer')

    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id, one=one,
                                                                aligned=True)

    spk_times = spikes[probe].times
    spk_clu = spikes[probe].clusters
    clu_regions = clusters[probe].acronym
    fitinfo = fitinfo.iloc[1:-1]
    fitinfo['trial_end'] = fitinfo['trial_start'] + 0.1
    fitinfo['adj_contrastLeft'] = np.tanh(contnorm * fitinfo['contrastLeft']) / np.tanh(contnorm)
    fitinfo['adj_contrastRight'] = np.tanh(contnorm * fitinfo['contrastRight']) / np.tanh(contnorm)

    vartypes = {'choice': 'value',
                'response_times': 'timing',
                'probabilityLeft': 'value',
                'pLeft_last': 'value',
                'feedbackType': 'value',
                'feedback_times': 'timing',
                'contrastLeft': 'value',
                'adj_contrastLeft': 'value',
                'contrastRight': 'value',
                'adj_contrastRight': 'value',
                'goCue_times': 'timing',
                'stimOn_times': 'timing',
                'trial_start': 'timing',
                'trial_end': 'timing',
                'prior': 'value',
                'prior_last': 'value',
                'wheel_velocity': 'continuous'}

    design = dm.DesignMatrix(fitinfo, vartypes, binwidth=binwidth)

    cosbases = mut.full_rcos(kernlen, nbases, design.binf)
    cosbases_whl = mut.full_rcos(kernlen, nbases, design.binf)
    design.add_covariate_timing('stimonL', 'stimOn_times', cosbases,
                                cond=lambda tr: np.isfinite(tr.contrastLeft),
                                deltaval='adj_contrastLeft',
                                desc='Kernel conditioned on L stimulus onset')
    design.add_covariate_timing('stimonR', 'stimOn_times', cosbases,
                                cond=lambda tr: np.isfinite(tr.contrastRight),
                                deltaval='adj_contrastRight',
                                desc='Kernel conditioned on R stimulus onset')
    design.add_covariate_raw('prior_tr',
                             lambda row: np.ones(design.binf(row.duration)) * row.prior_last,
                             desc='Step function on post-stimulus prior')
    # design.add_covariate('wheel', fitinfo['wheel_velocity'], cosbases_whl, -kernlen)
    design.compile_design_matrix()

    # _, s, v = np.linalg.svd(design[:, design.covar['wheel']['dmcol_idx']], full_matrices=False)
    # variances = s**2 / (s**2).sum()
    # n_keep = np.argwhere(np.cumsum(variances) >= 0.9999)[0, 0]
    # wheelcols = design[:, design.covar['wheel']['dmcol_idx']]
    # reduced = wheelcols @ v[:n_keep].T
    # bases_reduced = cosbases_whl @ v[:n_keep].T
    # keepcols = ~np.isin(np.arange(design.dm.shape[1]), design.covar['wheel']['dmcol_idx'])
    # basedm = design[:, keepcols]
    # design.dm = np.hstack([basedm, reduced])
    # design.covar['wheel']['dmcol_idx'] = design.covar['wheel']['dmcol_idx'][:n_keep]
    # design.covar['wheel']['bases'] = bases_reduced

    print(np.linalg.cond(design.dm))
    trialinds = np.array([(tr, np.searchsorted(design.trlabels.flat, tr))
                          for tr in design.trialsdf.index])
    tmparr = np.roll(trialinds[:, 1], -1)
    tmparr[-1] = design.dm.shape[0]
    trialinds = np.hstack((trialinds, tmparr.reshape(-1, 1)))

    nglm = lm.LinearGLM(design, spk_times, spk_clu, estimator=RidgeCV(cv=3))
    sfs = mut.SequentialSelector(nglm)
    sfs.fit()
    sequences, scores = sfs.sequences_, sfs.scores_
    nglm.clu_regions = clu_regions
    return nglm, sequences, scores
    # glm_template = deepcopy(nglm)
    # nglm.fit(printcond=False)
    # realscores = nglm.score()
    # nglm.clu_regions = clu_regions

    # scoreslist = []
    # weightslist = []
    # for _ in tqdm(range(num_pseudosess), desc='Pseudo block iteration num', leave=False,
    #               disable=not progress):
    #     newsess = generate_pseudo_session(fitinfo)
    #     stim_side, stimuli, actions, _ = utils.format_data(newsess)

    #     tmpdf = design.trialsdf.copy()
    #     tmpdf['contrastRight'] = newsess['contrastRight']
    #     tmpdf['contrastLeft'] = newsess['contrastLeft']
    #     tmpdf['adj_contrastLeft'] = np.tanh(contnorm * tmpdf['contrastLeft']) / np.tanh(contnorm)
    #     tmpdf['adj_contrastRight'] = np.tanh(contnorm * tmpdf['contrastRight']) / np.tanh(contnorm)

    #     tmpdm = dm.DesignMatrix(tmpdf, vartypes, binwidth=binwidth)
    #     tmpdm.add_covariate_timing('stimonL', 'stimOn_times', cosbases,
    #                                cond=lambda tr: np.isfinite(tr.contrastLeft),
    #                                deltaval='adj_contrastLeft',
    #                                desc='Kernel conditioned on L stimulus onset')
    #     tmpdm.add_covariate_timing('stimonR', 'stimOn_times', cosbases,
    #                                cond=lambda tr: np.isfinite(tr.contrastRight),
    #                                deltaval='adj_contrastRight',
    #                                desc='Kernel conditioned on R stimulus onset')
    #     tmpdm.compile_design_matrix()

    #     tmpglm = deepcopy(glm_template)
    #     stimlind = tmpglm.design.covar['stimonL']['dmcol_idx']
    #     stimrind = tmpglm.design.covar['stimonR']['dmcol_idx']

    #     tmpglm.design.dm[:, stimlind] = tmpdm.dm[:, :nbases]
    #     tmpglm.design.dm[:, stimrind] = tmpdm.dm[:, nbases:]

    #     tmpglm.fit(printcond=False)
    #     weightslist.append((tmpglm.coefs, tmpglm.intercepts))
    #     with np.errstate(all='ignore'):
    #         scoreslist.append(tmpglm.score())
    # return nglm, realscores, scoreslist, weightslist

