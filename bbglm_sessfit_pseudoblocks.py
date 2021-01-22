"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from oneibl import one
import numpy as np
import pandas as pd
from brainbox.modeling import glm
from brainbox.population.population import _generate_pseudo_blocks
import brainbox.io.one as bbone
from export_funs import trialinfo_to_df
from prior_funcs import fit_sess_psytrack
from datetime import date
import pickle
import os
from tqdm import tqdm

offline = True
one = one.ONE()

ephys_cache = {}


def fit_session(session_id, kernlen, nbases,
                t_before=1., t_after=0.6, prior_estimate='psytrack', max_len=2., probe_idx=0,
                method='minimize', alpha=0, contnorm=5., binwidth=0.02, wholetrial_step=False,
                blocktrain=False, abswheel=False, no_50perc=False, num_pseudosess=100,
                fit_intercept=True):
    if not abswheel:
        signwheel = True
    else:
        signwheel = False
    trialsdf = trialinfo_to_df(session_id, maxlen=max_len, t_before=t_before, t_after=t_after,
                               glm_binsize=binwidth, ret_abswheel=abswheel, ret_wheel=signwheel)
    if prior_estimate == 'psytrack':
        print('Fitting psytrack esimates...')
        wts, stds = fit_sess_psytrack(session_id, maxlength=max_len, as_df=True)
        wts['bias'] = wts['bias'] - np.mean(wts['bias'])
        fitinfo = pd.concat((trialsdf, wts['bias']), axis=1)
        bias_next = np.roll(fitinfo['bias'], -1)
        bias_next = pd.Series(bias_next, index=fitinfo['bias'].index)[:-1]
        fitinfo['bias_next'] = bias_next
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
    if session_id not in ephys_cache:
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id, one=one,
                                                                    aligned=True)
        for probe in spikes:
            null_keys_spk = []
            for key in spikes[probe]:
                if key not in ('times', 'clusters'):
                    null_keys_spk.append(key)
            for key in null_keys_spk:
                _ = spikes[probe].pop(key)
            null_keys_clu = []
            for key in clusters[probe]:
                if key not in ('acronym', 'atlas_id'):
                    null_keys_clu.append(key)
            for key in null_keys_clu:
                _ = clusters[probe].pop(key)
        ephys_cache[session_id] = (spikes, clusters)
    else:
        spikes, clusters = ephys_cache[session_id]
    spk_times = spikes[probestr].times
    spk_clu = spikes[probestr].clusters
    clu_regions = clusters[probestr].acronym
    fitinfo['pLeft_last'] = pd.Series(np.roll(fitinfo['probabilityLeft'], 1),
                                      index=fitinfo.index)[:-1]
    fitinfo = fitinfo.iloc[1:-1]
    fitinfo['adj_contrastLeft'] = np.tanh(contnorm * fitinfo['contrastLeft']) / np.tanh(contnorm)
    fitinfo['adj_contrastRight'] = np.tanh(contnorm * fitinfo['contrastRight']) / np.tanh(contnorm)

    if no_50perc:
        fitinfo = fitinfo[fitinfo.probabilityLeft != 0.5]

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
                'bias': 'value',
                'bias_next': 'value',
                'wheel_velocity': 'continuous'}
    if t_before < 0.7:
        raise ValueError('t_before needs to be 0.7 or greater in order to do -0.1 to -0.7 step'
                         ' function on pLeft')
 
    def stepfunc(row):
        currvec = np.ones(nglm.binf(row.feedback_times)) * row.pLeft_last
        nextvec = np.ones(nglm.binf(row.duration) - nglm.binf(row.feedback_times)) *\
            row.probabilityLeft
        return np.hstack((currvec, nextvec))

    def stepfunc_prestim(row):
        stepvec = np.zeros(nglm.binf(row.duration))
        stepvec[stepbounds[0]:stepbounds[1]] = row.pLeft_last
        return stepvec

    def stepfunc_bias(row):
        currvec = np.ones(nglm.binf(row.feedback_times)) * row.bias
        nextvec = np.ones(nglm.binf(row.duration) - nglm.binf(row.feedback_times)) *\
            row.bias_next
        return np.hstack((currvec, nextvec))

    nglm = glm.NeuralGLM(fitinfo, spk_times, spk_clu, vartypes, binwidth=binwidth,
                         blocktrain=blocktrain)
    nglm.clu_regions = clu_regions
    stepbounds = [nglm.binf(t_before - 0.7), nglm.binf(t_before - 0.1)]

    # cosbases_long = glm.full_rcos(kernlen, nbases, nglm.binf)
    # cosbases_short = glm.full_rcos(0.4, nbases, nglm.binf)
    # nglm.add_covariate_timing('stimonL', 'stimOn_times', cosbases_long,
    #                           cond=lambda tr: np.isfinite(tr.contrastLeft),
    #                           deltaval='adj_contrastLeft',
    #                           desc='Kernel conditioned on L stimulus onset')
    # nglm.add_covariate_timing('stimonR', 'stimOn_times', cosbases_long,
    #                           cond=lambda tr: np.isfinite(tr.contrastRight),
    #                           deltaval='adj_contrastRight',
    #                           desc='Kernel conditioned on R stimulus onset')
    # nglm.add_covariate_timing('correct', 'feedback_times', cosbases_long,
    #                           cond=lambda tr: tr.feedbackType == 1,
    #                           desc='Kernel conditioned on correct feedback')
    # nglm.add_covariate_timing('incorrect', 'feedback_times', cosbases_long,
    #                           cond=lambda tr: tr.feedbackType == -1,
    #                           desc='Kernel conditioned on incorrect feedback')
    if prior_estimate is None and wholetrial_step:
        nglm.add_covariate_raw('pLeft', stepfunc, desc='Step function on prior estimate')
    elif prior_estimate is None and not wholetrial_step:
        nglm.add_covariate_raw('pLeft', stepfunc_prestim,
                               desc='Step function on prior estimate')
    elif prior_estimate == 'psytrack':
        nglm.add_covariate_raw('pLeft', stepfunc_bias, desc='Step function on prior estimate')
    # nglm.add_covariate('wheel', fitinfo['wheel_velocity'], cosbases_short, -0.4)
    nglm.compile_design_matrix()
    nglm.fit(method=method, alpha=alpha, printcond=False, epochs=5000,
             fit_intercept=fit_intercept)
    realscores = nglm.score()

    scoreslist = []
    weightslist = []
    for i in tqdm(range(num_pseudosess), desc='Pseudo block iteration num', leave=False):
        newblocks = _generate_pseudo_blocks(len(fitinfo))
        probmap = {1: 0.8, 0: 0.2}
        newprobs = [probmap[i] for i in newblocks]
        tmp_df = fitinfo.copy()
        tmp_df['probabilityLeft'] = newprobs
        tmpglm = glm.NeuralGLM(tmp_df, spk_times, spk_clu, vartypes, binwidth=binwidth,
                               blocktrain=blocktrain)
        tmpglm.clu_regions = clu_regions
        stepbounds = [tmpglm.binf(t_before - 0.7), tmpglm.binf(t_before - 0.1)]

        # cosbases_long = glm.full_rcos(kernlen, nbases, tmpglm.binf)
        # cosbases_short = glm.full_rcos(0.4, nbases, tmpglm.binf)
        # tmpglm.add_covariate_timing('stimonL', 'stimOn_times', cosbases_long,
        #                           cond=lambda tr: np.isfinite(tr.contrastLeft),
        #                           deltaval='adj_contrastLeft',
        #                           desc='Kernel conditioned on L stimulus onset')
        # tmpglm.add_covariate_timing('stimonR', 'stimOn_times', cosbases_long,
        #                           cond=lambda tr: np.isfinite(tr.contrastRight),
        #                           deltaval='adj_contrastRight',
        #                           desc='Kernel conditioned on R stimulus onset')
        # tmpglm.add_covariate_timing('correct', 'feedback_times', cosbases_long,
        #                           cond=lambda tr: tr.feedbackType == 1,
        #                           desc='Kernel conditioned on correct feedback')
        # tmpglm.add_covariate_timing('incorrect', 'feedback_times', cosbases_long,
        #                           cond=lambda tr: tr.feedbackType == -1,
        #                           desc='Kernel conditioned on incorrect feedback')
        if prior_estimate is None and wholetrial_step:
            tmpglm.add_covariate_raw('pLeft', stepfunc, desc='Step function on prior estimate')
        elif prior_estimate is None and not wholetrial_step:
            tmpglm.add_covariate_raw('pLeft', stepfunc_prestim,
                                     desc='Step function on prior estimate')
        elif prior_estimate == 'psytrack':
            tmpglm.add_covariate_raw('pLeft', stepfunc_bias, desc='Step function on prior estimate')
        # tmpglm.add_covariate('wheel', fitinfo['wheel_velocity'], cosbases_short, -0.4)
        tmpglm.compile_design_matrix()
        tmpglm.fit(method=method, alpha=alpha, printcond=False, epochs=5000,
                   fit_intercept=fit_intercept)
        weightslist.append(tmpglm.coefs)
        with np.errstate(all='ignore'):
            scoreslist.append(tmpglm.score())
    return nglm, realscores, scoreslist, weightslist


if __name__ == "__main__":
    from scipy.stats import percentileofscore
    from brainbox.population.population import _generate_pseudo_blocks
    import matplotlib.pyplot as plt
    nickname = 'CSH_ZAD_001'
    sessdate = '2020-01-15'
    probe_idx = 0
    kernlen = 0.6
    nbases = 10
    method = 'pytorch'
    ids = one.search(subject=nickname, date_range=[sessdate, sessdate],
                     dataset_types=['spikes.clusters'])

    abswheel = True
    kernlen = 0.6
    nbases = 10
    alpha = 0
    stepwise = True
    wholetrial_step = False
    no_50perc = True
    method = 'sklearn'
    blocking = False
    prior_estimate = None
    fit_intercept = False
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
    
    nullscores = np.vstack(scoreslist)
    nullweights = np.vstack([[w[0] for w in weights] for weights in weightlist])
    msub_nullweights = nullweights - np.mean(nullweights, axis=0)
    msub_wts = np.array([w[0] for w in nglm.coefs]) - np.mean(nullweights, axis=0)
    score_percentiles = [(100 - percentileofscore(nullscores[:, i], sc)) / 100
                         for i, sc in enumerate(realscores)]
    wt_percentiles = [(100 - percentileofscore(np.abs(msub_nullweights)[:, i], np.abs(w))) / 100
                      for i, w in enumerate(msub_wts)]
    plt.hist(score_percentiles)

    trdf = nglm.trialsdf
    lowblock_rates = []
    highblock_rates = []
    for i, idx in enumerate(trdf.index):
        blockprob = trdf.loc[idx].probabilityLeft
        trialmask = np.isin(nglm.trlabels, idx).flatten()
        trialcounts = nglm.binnedspikes[trialmask]
        trialrates = np.sum(trialcounts, axis=0) / (trialcounts.shape[0] * binwidth)
        if blockprob == 0.5:
            continue
        if blockprob == 0.2:
            lowblock_rates.append(trialrates)
        elif blockprob == 0.8:
            highblock_rates.append(trialrates)
    meandiff = np.mean(np.vstack(lowblock_rates), axis=0) -\
        np.mean(np.vstack(highblock_rates), axis=0)
    mediandiff = np.median(np.vstack(lowblock_rates), axis=0) -\
        np.median(np.vstack(highblock_rates), axis=0)

    null_meandiffs = []
    null_mediandiffs = []
    pblock_mapper = {0: 0.2, 1: 0.8}
    for i in tqdm(range(1000)):
        trdf['pseudoblock'] = [pblock_mapper[i] for i in _generate_pseudo_blocks(len(trdf))]
        null_lowblock_rates = []
        null_highblock_rates = []
        for i, idx in enumerate(trdf.index):
            blockprob = trdf.loc[idx].pseudoblock
            trialmask = np.isin(nglm.trlabels, idx).flatten()
            trialcounts = nglm.binnedspikes[trialmask]
            trialrates = np.sum(trialcounts, axis=0) / (trialcounts.shape[0] * binwidth)
            if blockprob == 0.5:
                continue
            if blockprob == 0.2:
                null_lowblock_rates.append(trialrates)
            elif blockprob == 0.8:
                null_highblock_rates.append(trialrates)
        null_meandiff = np.mean(np.vstack(null_lowblock_rates), axis=0) -\
            np.mean(np.vstack(null_highblock_rates), axis=0)
        null_mediandiff = np.median(np.vstack(null_lowblock_rates), axis=0) -\
            np.median(np.vstack(null_highblock_rates), axis=0)
        null_meandiffs.append(null_meandiff)
        null_mediandiffs.append(null_mediandiff)
    null_meandiffsarr = np.vstack(null_meandiffs)
    null_mediandiffsarr = np.vstack(null_mediandiffs)
    
    mean_percentiles = [(100 - percentileofscore(null_meandiffsarr[:, i], diff)) / 100
                        for i, diff in enumerate(meandiff)]
    median_percentiles = [(100 - percentileofscore(null_mediandiffsarr[:, i], diff)) / 100
                          for i, diff in enumerate(mediandiff)]
    
