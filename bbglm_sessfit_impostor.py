"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from oneibl import one
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

if __name__ != "__main__":
    offline = True
else:
    offline = False
one = one.ONE(offline=offline)
subjects, ins, ins_id, sess_ids, _ = utils.get_bwm_ins_alyx(one)

ephys_cache = {}


def fit_session(session_id, kernlen, nbases,
                t_before=1., t_after=0.6, prior_estimate='charles', max_len=2., probe_idx=0,
                contnorm=5., binwidth=0.02, abswheel=False, num_pseudosess=100, progress=True,
                target_regressor='prior', one=one):
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
        for i in range(len(sess_ids)):
            if subjects[i] == mouse_name:  # take only sessions of first mice
                data = utils.load_session(sess_ids[i])
                if data['choice'] is not None and data['probabilityLeft'][0] == 0.5:
                    stim_side, stimuli, actions, pLeft_oracle = utils.format_data(data)
                    stimuli_arr.append(stimuli)
                    actions_arr.append(actions)
                    stim_sides_arr.append(stim_side)
                    session_uuids.append(sess_ids[i])

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
        trialsdf['prior'] = signals['prior'][trialsdf.index]
        trialsdf['prior_last'] = pd.Series(np.roll(trialsdf['prior'], 1), index=trialsdf.index)
        fitinfo = trialsdf.copy()
    elif prior_estimate is None:
        fitinfo = trialsdf.copy()
    else:
        raise NotImplementedError('Only psytrack currently available')
    # spk_times = one.load(session_id, dataset_types=['spikes.times'], offline=offline)[probe_idx]
    # spk_clu = one.load(session_id, dataset_types=['spikes.clusters'], offline=offline)[probe_idx]

    # A bit of messy loading to get spike times, clusters, and cluster brain regions.
    # This is the way it is because loading with regions takes forever. The weird for loop
    # ensures that we don't waste memory storing unnecessary and large arrays.
    probestr = 'probe0' + str(probe_idx)
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id, one=one,
                                                                aligned=True)

    spk_times = spikes[probestr].times
    spk_clu = spikes[probestr].clusters
    clu_regions = clusters[probestr].acronym
    fitinfo = fitinfo.iloc[1:-1]
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
    if t_before < 0.6:
        raise ValueError('t_before needs to be 0.7 or greater in order to do -0.1 to -0.7 step'
                         ' function on pLeft')

    def stepfunc_prestim(row):
        stepvec = np.zeros(design.binf(row.duration))
        stepvec[stepbounds[0]:stepbounds[1]] = row.prior_last
        return stepvec

    def stepfunc_poststim(row):
        vec = np.zeros(design.binf(row.duration))
        currtr_start = design.binf(row.stimOn_times + 0.1)
        currtr_end = design.binf(row.feedback_times)
        vec[currtr_start:currtr_end] = row.prior_last
        vec[currtr_end:] = row.prior
        return vec

    design = dm.DesignMatrix(fitinfo, vartypes, binwidth=binwidth)
    stepbounds = [design.binf(t_before - 0.6), design.binf(t_before - 0.1)]

    cosbases_long = mut.full_rcos(kernlen, nbases, design.binf)
    cosbases_short = mut.full_rcos(0.4, nbases, design.binf)
    design.add_covariate_timing('stimonL', 'stimOn_times', cosbases_long,
                                cond=lambda tr: np.isfinite(tr.contrastLeft),
                                deltaval='adj_contrastLeft',
                                desc='Kernel conditioned on L stimulus onset')
    design.add_covariate_timing('stimonR', 'stimOn_times', cosbases_long,
                                cond=lambda tr: np.isfinite(tr.contrastRight),
                                deltaval='adj_contrastRight',
                                desc='Kernel conditioned on R stimulus onset')
    design.add_covariate_timing('correct', 'feedback_times', cosbases_long,
                                cond=lambda tr: tr.feedbackType == 1,
                                desc='Kernel conditioned on correct feedback')
    design.add_covariate_timing('incorrect', 'feedback_times', cosbases_long,
                                cond=lambda tr: tr.feedbackType == -1,
                                desc='Kernel conditioned on incorrect feedback')
    design.add_covariate_raw('prior', stepfunc_prestim,
                             desc='Step function on prior estimate')
    design.add_covariate_raw('prior_tr', stepfunc_poststim,
                             desc='Step function on post-stimulus prior')
    design.add_covariate('wheel', fitinfo['wheel_velocity'], cosbases_short, -0.4)
    design.compile_design_matrix()

    _, s, v = np.linalg.svd(design[:, design.covar['wheel']['dmcol_idx']], full_matrices=False)
    variances = s**2 / (s**2).sum()
    n_keep = np.argwhere(np.cumsum(variances) >= 0.9999)[0, 0]
    wheelcols = design[:, design.covar['wheel']['dmcol_idx']]
    reduced = wheelcols @ v[:n_keep].T
    bases_reduced = cosbases_short @ v[:n_keep].T
    keepcols = ~np.isin(np.arange(design.dm.shape[1]), design.covar['wheel']['dmcol_idx'])
    basedm = design[:, keepcols]
    design.dm = np.hstack([basedm, reduced])
    design.covar['wheel']['dmcol_idx'] = design.covar['wheel']['dmcol_idx'][:n_keep]
    design.covar['wheel']['bases'] = bases_reduced

    print(np.linalg.cond(design.dm))
    trialinds = np.array([(tr, np.searchsorted(design.trlabels.flat, tr))
                          for tr in design.trialsdf.index])
    tmparr = np.roll(trialinds[:, 1], -1)
    tmparr[-1] = design.dm.shape[0]
    trialinds = np.hstack((trialinds, tmparr.reshape(-1, 1)))

    nglm = lm.LinearGLM(design, spk_times, spk_clu, estimator=RidgeCV(cv=3))
    glm_template = deepcopy(nglm)
    nglm.fit(printcond=False)
    realscores = nglm.score()
    nglm.clu_regions = clu_regions

    scoreslist = []
    weightslist = []
    for _ in tqdm(range(num_pseudosess), desc='Pseudo block iteration num', leave=False,
                  disable=not progress):
        newsess = generate_pseudo_session(fitinfo)
        stim_side, stimuli, actions, _ = utils.format_data(newsess)
        signals = model.compute_signal(signal=['prior', 'prediction_error'],
                                       act=actions.values, stim=stimuli,
                                       side=stim_side.values)

        tmp_df = design.trialsdf.copy()
        tmp_df['prior'] = signals['prior']
        tmp_df['prior_last'] = pd.Series(np.roll(tmp_df['prior'], 1),
                                         index=tmp_df.index)
        tmpglm = deepcopy(glm_template)
        pl_idx = design.covar['prior']['dmcol_idx']
        plt_idx = design.covar['prior_tr']['dmcol_idx']
        for tr, start, end in trialinds:
            if target_regressor == 'prior':
                tmpglm.design.dm[start:end, pl_idx][tmpglm.design.dm[start:end, pl_idx]
                                                    > 0] = tmp_df.prior_last[tr]
            else:
                p_old = fitinfo.prior[tr]
                pl_old = fitinfo.prior_last[tr]
                tmpglm.design.dm[start:end, plt_idx][tmpglm.design.dm[start:end,
                                                     plt_idx] == p_old] = tmp_df.prior[tr]
                tmpglm.design.dm[start:end, plt_idx][tmpglm.design.dm[start:end,
                                                     plt_idx] == pl_old] = tmp_df.prior_last[tr]
        tmpglm.fit(printcond=False)
        weightslist.append((tmpglm.coefs, tmpglm.intercepts))
        with np.errstate(all='ignore'):
            scoreslist.append(tmpglm.score())
    return nglm, realscores, scoreslist, weightslist


if __name__ == "__main__":
    nickname = 'CSH_ZAD_001'
    sessdate = '2020-01-15'
    probe_idx = 0
    method = 'ridge'
    ids = one.search(subject=nickname, date_range=[sessdate, sessdate],
                     dataset_types=['spikes.clusters'])

    abswheel = True
    kernlen = 0.6
    nbases = 10
    alpha = 0
    stepwise = True
    wholetrial_step = True
    no_50perc = True
    method = 'ridge'
    blocking = False
    prior_estimate = None
    fit_intercept = True
    binwidth = 0.02
    num_pseudosess = 100

    nglm, realscores, scoreslist, weightlist = fit_session(ids[0], kernlen, nbases,
                                                           prior_estimate=prior_estimate,
                                                           probe_idx=probe_idx, method=method,
                                                           alpha=alpha,
                                                           binwidth=binwidth, blocktrain=blocking,
                                                           wholetrial_step=wholetrial_step,
                                                           abswheel=abswheel, no_50perc=no_50perc,
                                                           fit_intercept=fit_intercept,
                                                           num_pseudosess=num_pseudosess)
