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

offline = True
one = one.ONE()

ephys_cache = {}


def fit_session(session_id, kernlen, nbases,
                t_before=0.6, t_after=0.6, max_len=2., probe_idx=0, contnorm=5., binwidth=0.02,
                abswheel=False, no_50perc=False, one=one):
    if not abswheel:
        signwheel = True
    else:
        signwheel = False
    trdf = bbone.load_trials_df(session_id, maxlen=max_len, t_before=t_before, t_after=t_after,
                                wheel_binsize=binwidth, ret_abswheel=abswheel,
                                ret_wheel=signwheel, one=one)
    probestr = 'probe0' + str(probe_idx)
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id, one=one,
                                                                aligned=True)
    spk_times = spikes[probestr].times
    spk_clu = spikes[probestr].clusters
    clu_regions = clusters[probestr].acronym
    try:
        clu_qc = clusters[probestr]['metrics'].loc[:, 'label':'ks2_label']
    except Exception:
        clu_qc = None

    trdf['pLeft_last'] = pd.Series(np.roll(trdf['probabilityLeft'], 1),
                                   index=trdf.index)[:-1]
    trdf = trdf.iloc[1:-1]
    trdf['adj_contrastLeft'] = np.tanh(
        contnorm * trdf['contrastLeft']) / np.tanh(contnorm)
    trdf['adj_contrastRight'] = np.tanh(
        contnorm * trdf['contrastRight']) / np.tanh(contnorm)

    if no_50perc:
        trdf = trdf[trdf.probabilityLeft != 0.5]

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

    def stepfunc_prestim(row):
        stepvec = np.zeros(design.binf(row.duration))
        stepvec[stepbounds[0]:stepbounds[1]] = row.pLeft_last
        return stepvec

    def stepfunc_poststim(row):
        zerovec = np.zeros(design.binf(row.duration))
        currtr_start = design.binf(row.stimOn_times + 0.1)
        currtr_end = design.binf(row.feedback_times)
        zerovec[currtr_start:currtr_end] = row.pLeft_last
        zerovec[currtr_end:] = row.probabilityLeft
        return zerovec

    design = dm.DesignMatrix(trdf, vartypes, binwidth=binwidth)
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
    design.add_covariate_raw('pLeft', stepfunc_prestim,
                             desc='Step function on prior estimate')
    design.add_covariate_raw('pLeft_tr', stepfunc_poststim,
                             desc='Step function on post-stimulus prior')

    design.add_covariate('wheel', trdf['wheel_velocity'], cosbases_short, -0.4)
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
    nglm.clu_regions = clu_regions
    nglm.clu_qc = clu_qc
    nglm.clu_ids = nglm.clu_ids.flatten()
    sfs = mut.SequentialSelector(nglm)
    sfs.fit(progress=True)
    return nglm, sfs.sequences_, sfs.scores_


if __name__ == "__main__":
    eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
    test = fit_session(eid, 0.6, 10, no_50perc=True)
