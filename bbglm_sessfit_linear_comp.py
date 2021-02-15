"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from oneibl import one
import numpy as np
import pandas as pd
from brainbox.modeling import glm
from brainbox.modeling import glm_linear
from brainbox.population.population import _generate_pseudo_blocks
import brainbox.io.one as bbone
from export_funs import trialinfo_to_df
from prior_funcs import fit_sess_psytrack

offline = True
one = one.ONE()

ephys_cache = {}


def fit_session(session_id, kernlen, nbases,
                t_before=1., t_after=0.6, prior_estimate='psytrack', max_len=2., probe_idx=0,
                method='minimize', alpha=0, contnorm=5., binwidth=0.02, wholetrial_step=False,
                blocktrain=False, abswheel=False, no_50perc=False, num_pseudosess=100,
                fit_intercept=True, subset=False):
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
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id, one=one,
                                                                aligned=True)

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
        currvec = np.ones(nglm.binf(row.duration)) * row.pLeft_last
        # nextvec = np.ones(nglm.binf(row.duration) - nglm.binf(row.feedback_times)) *\
        #     row.probabilityLeft
        return currvec

    def stepfunc_prestim(row):
        stepvec = np.zeros(nglm.binf(row.duration))
        stepvec[stepbounds[0]:stepbounds[1]] = row.pLeft_last
        return stepvec

    def stepfunc_bias(row):
        currvec = np.ones(nglm.binf(row.feedback_times)) * row.bias
        nextvec = np.ones(nglm.binf(row.duration) - nglm.binf(row.feedback_times)) *\
            row.bias_next
        return np.hstack((currvec, nextvec))

    poissglm = glm.NeuralGLM(fitinfo.copy(), spk_times, spk_clu, vartypes, binwidth=binwidth,
                             blocktrain=blocktrain, subset=subset)
    linglm = glm_linear.LinearGLM(fitinfo.copy(), spk_times, spk_clu, vartypes.copy(),
                                  binwidth=binwidth, blocktrain=blocktrain, subset=subset)

    for nglm in (poissglm, linglm):
        nglm.clu_regions = clu_regions
        stepbounds = [nglm.binf(0.1), nglm.binf(0.6)]

        cosbases_long = glm.full_rcos(kernlen, nbases, nglm.binf)
        cosbases_short = glm.full_rcos(0.4, 3, nglm.binf)
        nglm.add_covariate_timing('stimonL', 'stimOn_times', cosbases_long,
                                  cond=lambda tr: np.isfinite(tr.contrastLeft),
                                  deltaval='adj_contrastLeft',
                                  desc='Kernel conditioned on L stimulus onset')
        nglm.add_covariate_timing('stimonR', 'stimOn_times', cosbases_long,
                                  cond=lambda tr: np.isfinite(tr.contrastRight),
                                  deltaval='adj_contrastRight',
                                  desc='Kernel conditioned on R stimulus onset')
        nglm.add_covariate_timing('correct', 'feedback_times', cosbases_long,
                                  cond=lambda tr: tr.feedbackType == 1,
                                  desc='Kernel conditioned on correct feedback')
        nglm.add_covariate_timing('incorrect', 'feedback_times', cosbases_long,
                                  cond=lambda tr: tr.feedbackType == -1,
                                  desc='Kernel conditioned on incorrect feedback')
        if prior_estimate is None and wholetrial_step:
            nglm.add_covariate_raw('pLeft', stepfunc, desc='Step function on prior estimate')
        elif prior_estimate is None and not wholetrial_step:
            nglm.add_covariate_raw('pLeft', stepfunc_prestim,
                                   desc='Step function on prior estimate')
        elif prior_estimate == 'psytrack':
            nglm.add_covariate_raw('pLeft', stepfunc_bias, desc='Step function on prior estimate')
        nglm.add_covariate('wheel', fitinfo['wheel_velocity'], cosbases_short, -0.4)
        nglm.compile_design_matrix()
        if type(nglm) is glm.NeuralGLM:
            nglm.fit(method=method, alpha=0, rsq=True)
        else:
            nglm.fit(method='pure')

    return linglm, poissglm


def get_bwm_ins_alyx(one):
    """
    Return insertions that match criteria :
    - project code
    - session QC not critical (TODO may need to add probe insertion QC)
    - at least 1 alignment
    - behavior pass
    :return:
    ins: dict containing the full details on insertion as per the alyx rest query
    ins_id: list of insertions eids
    sess_id: list of (unique) sessions eids
    """
    ins = one.alyx.rest('insertions', 'list',
                        provenance='Ephys aligned histology track',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1,'
                               'json__extended_qc__alignment_resolved,True')
    ins_id = [item['id'] for item in ins]
    sess_id = [item['session_info']['id'] for item in ins]
    sess_id = np.unique(sess_id)
    return ins, ins_id, sess_id


if __name__ == "__main__":
    import pickle
    from datetime import date
    ins, ins_ids, sess_ids = get_bwm_ins_alyx(one=one)

    abswheel = True
    kernlen = 0.6
    nbases = 5
    alpha = 0
    stepwise = True
    wholetrial_step = False
    no_50perc = True
    method = 'pytorch'
    blocking = False
    prior_estimate = None
    fit_intercept = True
    binwidth = 0.08
    probe_idx = 0

    rscores = {}
    dscores = {}
    meanrates = {}
    for eid in sess_ids[::2]:
        try:
            linglm, poissglm = fit_session(eid, kernlen, nbases,
                                           prior_estimate=prior_estimate,
                                           probe_idx=probe_idx, method=method,
                                           alpha=alpha,
                                           binwidth=binwidth, blocktrain=blocking,
                                           wholetrial_step=wholetrial_step,
                                           abswheel=abswheel, no_50perc=no_50perc,
                                           fit_intercept=fit_intercept, subset=stepwise)
        except Exception as e:
            print(e)
            continue
        if not stepwise:
            rscores[eid] = (linglm.score(), poissglm.score(rsq=True))
            dscores[eid] = (linglm.score(dsq=True), poissglm.score(dsq=True))
        else:
            rscores[eid] = (linglm.submodel_scores, poissglm.submodel_scores)
        meanrates[eid] = poissglm.binnedspikes.mean(axis=0) / binwidth
    datestr = str(date.today())

    if stepwise:
        fw = open(f'/home/berk/Documents/lin_vs_poisson_modelcomp{datestr}.p', 'wb')
        pickle.dump({'rscores': rscores, 'meanrates': meanrates, 'linglm': linglm}, fw)
        fw.close()
    
    else:
        fw = open(f'/home/berk/Documents/lin_vs_poisson_modelcomp{datestr}.p', 'wb')
        pickle.dump({'rscores': rscores, 'dscores': dscores, 'meanrates': meanrates,
                     'linglm': linglm}, fw)
        fw.close()
