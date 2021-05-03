"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from oneibl import one
import numpy as np
import pandas as pd
import brainbox.modeling.design_matrix as dm
import brainbox.modeling.linear as lm
import brainbox.modeling.utils as gut
import brainbox.io.one as bbone

offline = True
one = one.ONE()

ephys_cache = {}


def fit_session(session_id, kernlen, nbases,
                t_before=1., t_after=0.6, prior_estimate='psytrack', max_len=2., probe_idx=0,
                method='minimize', alpha=0, contnorm=5., binwidth=0.02, wholetrial_step=False,
                blocktrain=False, abswheel=False, no_50perc=False, num_pseudosess=100,
                fit_intercept=True, subset=False, var_thresh=0.9999):
    if not abswheel:
        signwheel = True
    else:
        signwheel = False
    trialsdf = bbone.load_trials_df(session_id, one=one,
                                    maxlen=max_len, t_before=t_before, t_after=t_after,
                                    wheel_binsize=binwidth, ret_abswheel=abswheel,
                                    ret_wheel=signwheel)
    if prior_estimate is None:
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
        currvec = np.ones(design.binf(row.duration)) * row.pLeft_last
        # nextvec = np.ones(design.binf(row.duration) - design.binf(row.feedback_times)) *\
        #     row.probabilityLeft
        return currvec

    def stepfunc_prestim(row):
        stepvec = np.zeros(design.binf(row.duration))
        stepvec[stepbounds[0]:stepbounds[1]] = row.pLeft_last
        return stepvec

    def stepfunc_poststim(row):
        zerovec = np.ones(design.binf(row.duration))
        currtr_start = design.binf(row.stimOn_times + 0.1)
        currtr_end = design.binf(row.feedback_times)
        zerovec[currtr_start:currtr_end] = row.pLeft_last
        zerovec[currtr_end:] = row.probabilityLeft
        return zerovec

    def stepfunc_bias(row):
        currvec = np.ones(design.binf(row.feedback_times)) * row.bias
        nextvec = np.ones(design.binf(row.duration) - design.binf(row.feedback_times)) *\
            row.bias_next
        return np.hstack((currvec, nextvec))

    design = dm.DesignMatrix(fitinfo, vartypes)
    stepbounds = [design.binf(0.1), design.binf(0.6)]

    cosbases_long = gut.full_rcos(kernlen, nbases, design.binf)
    cosbases_short = gut.full_rcos(0.4, 10, design.binf)
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
    design.add_covariate_raw('pLeft', stepfunc_prestim,
                             desc='Step function on prior estimate')
    design.add_covariate_raw('pLeft_tr', stepfunc_poststim,
                             desc='Step function on post-stimulus prior')
    design.add_covariate('wheel', fitinfo['wheel_velocity'], cosbases_short, -0.4)
    design.compile_design_matrix()
    # Reduce dimension of the wheel covariates, to reduce the condition of the design matrix
    _, s, v = np.linalg.svd(design.dm[:, design.covar['wheel']['dmcol_idx']], full_matrices=False)
    variances = s**2 / (s**2).sum()
    n_keep = np.argwhere(np.cumsum(variances) >= var_thresh)[0, 0]
    wheelcols = design.dm[:, design.covar['wheel']['dmcol_idx']]
    reduced = wheelcols @ v[:n_keep].T
    bases_reduced = cosbases_short @ v[:n_keep].T
    keepcols = ~np.isin(np.arange(design.dm.shape[1]), design.covar['wheel']['dmcol_idx'])
    basedm = design.dm[:, keepcols]
    design.dm = np.hstack([basedm, reduced])
    design.covar['wheel']['dmcol_idx'] = design.covar['wheel']['dmcol_idx'][:n_keep]
    design.covar['wheel']['bases'] = bases_reduced

    linglm = lm.LinearGLM(design, spk_times, spk_clu, stepwise=False)
    linglm.fit()
    linglm.clu_regions = clu_regions
    return linglm


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
                               '~json__qc,CRITICAL,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1')
    sessions = {}
    for item in ins:
        s_eid = item['session_info']['id']
        if s_eid not in sessions:
            sessions[s_eid] = []
        sessions[s_eid].append(item['id'])
    return sessions


if __name__ == "__main__":
    import pickle
    from datetime import date
    sessions = get_bwm_ins_alyx(one=one)

    abswheel = True
    kernlen = 0.5
    nbases = 10
    alpha = 0
    stepwise = True
    wholetrial_step = False
    no_50perc = True
    method = 'minimize'
    blocking = False
    prior_estimate = None
    fit_intercept = True
    binwidth = 0.02

    rscores = {}
    msescores = {}
    meanrates = {}
    regions = {}
    weights = {}
    for eid in sessions.keys():
        for i in range(len(sessions[eid])):
            dkey = (eid, i)
            try:
                linglm = fit_session(eid, kernlen, nbases,
                                     prior_estimate=prior_estimate,
                                     probe_idx=i, method=method,
                                     alpha=alpha,
                                     binwidth=binwidth, blocktrain=blocking,
                                     wholetrial_step=wholetrial_step,
                                     abswheel=abswheel, no_50perc=no_50perc,
                                     fit_intercept=fit_intercept, subset=stepwise)
            except Exception as e:
                print(e)
                continue
            rscores[dkey] = linglm.submodel_scores
            msescores[dkey] = linglm.altsubmodel_scores
            meanrates[dkey] = linglm.binnedspikes.mean(axis=0) / binwidth
            regions[dkey] = linglm.clu_regions
            weights[dkey] = linglm.combine_weights()
    datestr = str(date.today())

    fw = open(f'/home/berk/Documents/lin_modelfit{datestr}.p', 'wb')
    pickle.dump({'rscores': rscores, 'msescores': msescores, 'meanrates': meanrates,
                 'regions': regions, 'weights': weights, 'linglm': linglm}, fw)
    fw.close()
