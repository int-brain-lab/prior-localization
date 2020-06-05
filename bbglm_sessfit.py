"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from oneibl import one
import numpy as np
import pandas as pd
import numba as nb
from brainbox.modeling import glm
from export_funs import trialinfo_to_df
from prior_funcs import fit_sess_psytrack

one = one.ONE()


def neglog(weights, x, y):
    xproj = x @ weights
    f = np.exp(xproj)
    nzidx = f != 0
    if np.any(y[~nzidx] != 0):
        return np.inf
    return -y[nzidx].reshape(1, -1) @ xproj[nzidx] + np.sum(f)


def d_neglog(weights, x, y):
    xproj = x @ weights
    f = np.exp(xproj)
    df = f
    nzidx = (f != 0).reshape(-1)
    if np.any(y[~nzidx] != 0):
        return np.inf
    return x[nzidx, :].T @ ((1 - y[nzidx] / f[nzidx]) * df[nzidx])


def dd_neglog(weights, x, y):
    xproj = x @ weights
    f = np.exp(xproj)
    df = f
    ddf = df
    nzidx = (f != 0).reshape(-1)
    if np.any(y[~nzidx] != 0):
        return np.inf
    yf = y[nzidx] / f[nzidx]
    p1 = ddf[nzidx] * (1 - yf) + (y[nzidx] * (df[nzidx] / f[nzidx])**2)
    p2 = x[nzidx, :]
    return (p1.reshape(-1, 1) * p2).T @ x[nzidx, :]


def fit_session(session_id, subject_name, sessdate, kernlen, nbases,
                t_before=0.4, t_after=0.6, prior_estimate='psytrack', max_len=2., probe_idx=0):
    trialsdf = trialinfo_to_df(session_id, maxlen=max_len)
    trialsdf['trial_start'] = trialsdf['stimOn_times'] - t_before
    trialsdf['trial_end'] = trialsdf['feedback_times'] + t_after
    if prior_estimate == 'psytrack':
        print('Fitting psytrack esimates...')
        wts, stds = fit_sess_psytrack(session_id, maxlength=max_len, as_df=True)
    else:
        raise NotImplementedError('Only psytrack currently available')
    spk_times = one.load(session_id, ['spikes.times'])[0]
    spk_clu = one.load(session_id, ['spikes.clusters'])[0]
    fitinfo = pd.concat((trialsdf, wts['bias']), axis=1)
    vartypes = {'choice': 'value',
                'response_times': 'timing',
                'probabilityLeft': 'value',
                'feedbackType': 'value',
                'feedback_times': 'timing',
                'contrastLeft': 'value',
                'contrastRight': 'value',
                'goCue_times': 'timing',
                'stimOn_times': 'timing',
                'trial_start': 'timing',
                'trial_end': 'timing',
                'bias': 'value'}
    nglm = glm.NeuralGLM(fitinfo, spk_times, spk_clu, vartypes)
    cosbases_long = glm.raised_cosine(kernlen, nbases, nglm.binf)
    nglm.add_covariate_timing('stimonL', 'stimOn_times', cosbases_long,
                              deltaval=trialsdf.contrastLeft,
                              cond=lambda tr: np.isfinite(tr.contrastLeft),
                              desc='Kernel conditioned on L stimulus onset')
    nglm.add_covariate_timing('stimonR', 'stimOn_times', cosbases_long,
                              deltaval=trialsdf.contrastRight,
                              cond=lambda tr: np.isfinite(tr.contrastRight),
                              desc='Kernel conditioned on R stimulus onset')
    nglm.add_covariate_timing('correct', 'feedback_times', cosbases_long,
                              cond=lambda tr: tr.feedbackType == 1,
                              desc='Kernel conditioned on correct feedback')
    nglm.add_covariate_timing('incorrect', 'feedback_times', cosbases_long,
                              cond=lambda tr: tr.feedbackType == -1,
                              desc='Kernel conditioned on incorrect feedback')
    nglm.compile_design_matrix()
    nglm.bin_spike_trains()
    nglm.fit(method='minimize')
    combined_weights = nglm.combine_weights()
    return combined_weights


if __name__ == "__main__":
    nickname = 'ZM_2240'
    sessdate = '2020-01-22'
    ids = one.search(subject=nickname, date_range=[sessdate, sessdate],
                     dataset_types=['spikes.clusters'])
    fit_session(ids[0], nickname, sessdate, 0.6, 10)
