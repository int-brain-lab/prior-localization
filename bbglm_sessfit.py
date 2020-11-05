"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from oneibl import one
import numpy as np
import pandas as pd
from brainbox.modeling import glm
import brainbox.io.one as bbone
from export_funs import trialinfo_to_df
from prior_funcs import fit_sess_psytrack

one = one.ONE()


def fit_session(session_id, kernlen, nbases,
                t_before=0.4, t_after=0.6, prior_estimate=None, max_len=2., probe_idx=0,
                method='minimize', alpha=0, contnorm=5., binwidth=0.02):
    trialsdf = trialinfo_to_df(session_id, maxlen=max_len, t_before=t_before, t_after=t_after,
                               glm_binsize=binwidth, ret_abswheel=True)
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
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id)
    spk_times = spikes[probestr].times
    spk_clu = spikes[probestr].clusters
    clu_regions = clusters[probestr].acronym
    fitinfo['pLeft_last'] = pd.Series(np.roll(fitinfo['probabilityLeft'], 1),
                                      index=fitinfo.index)[:-1]
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
                'bias': 'value',
                'bias_next': 'value',
                'wheel_velocity': 'continuous'}
    nglm = glm.NeuralGLM(fitinfo, spk_times, spk_clu, vartypes, binwidth=binwidth,)  # subset=True)
    nglm.clu_regions = clu_regions

    def stepfunc(row):
        currvec = np.ones(nglm.binf(row.feedback_times)) * row.pLeft_last
        nextvec = np.ones(nglm.binf(row.duration) - nglm.binf(row.feedback_times)) *\
            row.probabilityLeft
        return np.hstack((currvec, nextvec))

    def stepfunc_bias(row):
        currvec = np.ones(nglm.binf(row.feedback_times)) * row.bias
        nextvec = np.ones(nglm.binf(row.duration) - nglm.binf(row.feedback_times)) *\
            row.bias_next
        return np.hstack((currvec, nextvec))

    cosbases_long = glm.full_rcos(kernlen, nbases, nglm.binf)
    cosbases_short = glm.full_rcos(0.4, nbases, nglm.binf)
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
    if prior_estimate is None:
        nglm.add_covariate_raw('pLeft', stepfunc, desc='Step function on prior estimate')
    elif prior_estimate == 'psytrack':
        nglm.add_covariate_raw('pLeft', stepfunc_bias, desc='Step function on prior estimate')
    nglm.add_covariate('wheel', fitinfo['wheel_velocity'], cosbases_short, -0.4)
    nglm.compile_design_matrix()
    nglm.fit(method=method, alpha=alpha)
    combined_weights = nglm.combine_weights()
    return nglm, combined_weights


if __name__ == "__main__":
    nickname = 'CSH_ZAD_001'
    sessdate = '2020-01-15'
    probe_idx = 0
    kernlen = 0.6
    nbases = 10
    method = 'pytorch'
    ids = one.search(subject=nickname, date_range=[sessdate, sessdate],
                     dataset_types=['spikes.clusters'])
    nglm, sessweights = fit_session(ids[0], kernlen, nbases,
                                    probe_idx=probe_idx, method=method)
    sknglm, _ = fit_session(ids[0], kernlen, nbases,
                            probe_idx=probe_idx, method='sklearn')

    # def bias_nll(weights, intercept, dm, y):
    #     biasdm = np.pad(dm, ((0, 0), (1, 0)), mode='constant', constant_values=1)
    #     biaswts = np.hstack((intercept, weights))
    #     return glm.neglog(biaswts, biasdm, y)[0]

    # sklearn_nll = pd.Series([bias_nll(wt, sknglm.intercepts.loc[i],
    #                                   sknglm.dm, sknglm.binnedspikes[:, nglm.clu_ids.flat == i])
    #                          for i, wt in sknglm.coefs.iteritems()])

    # minimize_nll = pd.Series([bias_nll(wt, nglm.intercepts.loc[i],
    #                                    nglm.dm, nglm.binnedspikes[:, nglm.clu_ids.flat == i])
    #                           for i, wt in nglm.coefs.iteritems()])
    # ll_diffs = -sklearn_nll - (-minimize_nll)

    # outdict = {'kernlen': kernlen, 'nbases': nbases, 'weights': sessweights, 'fitobj': nglm}
    # today = str(date.today())
    # subjfilepath = os.path.abspath(f'./fits/{nickname}/'
    #                                f'{sessdate}_session_{today}_probe{probe_idx}_pyfit_{method}.p')
    # fw = open(subjfilepath, 'wb')
    # pickle.dump(outdict, fw)
    # fw.close()
