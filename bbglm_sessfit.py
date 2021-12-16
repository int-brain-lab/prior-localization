"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import numpy as np
import pandas as pd
import brainbox.modeling.design_matrix as dm
import brainbox.modeling.linear as lm
import brainbox.modeling.utils as mut
import brainbox.io.one as bbone
import brainbox.metrics.single_units as bbqc
from sklearn.linear_model import LinearRegression
from one.api import ONE


def load_regressors(session_id, probe,
                    max_len=2., t_before=0., t_after=0., binwidth=0.02, abswheel=False,
                    resolved_alignment=False, ret_qc=False, one=None):
    one = ONE() if one is None else one
    dataset_types = None if not ret_qc else ['spikes.times',
                                             'spikes.clusters',
                                             'spikes.amps',
                                             'spikes.depths']

    trialsdf = bbone.load_trials_df(session_id,
                                    maxlen=max_len, t_before=t_before, t_after=t_after,
                                    wheel_binsize=binwidth, ret_abswheel=abswheel,
                                    ret_wheel=~abswheel, one=one)

    if resolved_alignment:
        spikes, clusters, _ = bbone.load_spike_sorting_fast(session_id,
                                                            probe=probe,
                                                            dataset_types=dataset_types,
                                                            one=one)
    else:
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id,
                                                                    dataset_types=dataset_types,
                                                                    one=one,
                                                                    aligned=True)
    spikes, clusters = spikes[probe], clusters[probe]
    spk_times = spikes.times
    spk_clu = spikes.clusters
    clu_regions = clusters.acronym
    if not ret_qc:
        return trialsdf, spk_times, spk_clu, clu_regions

    try:
        clu_qc = clusters['metrics'].loc[:, 'label':'ks2_label']
    except Exception:
        clu_qc = bbqc.quick_unit_metrics(spikes.clusters, spikes.times, spikes.amps, spikes.depths)
    return trialsdf, spk_times, spk_clu, clu_regions, clu_qc


def generate_design(trialsdf, prior, t_before, bases,
                    iti_prior=[-0.4, -0.1], fmove_offset=-0.4, wheel_offset=-0.4,
                    contnorm=5., binwidth=0.02, reduce_wheel_dim=True):
    """
    Generate GLM design matrix object

    Parameters
    ----------
    trialsdf : pd.DataFrame
        Trials dataframe with trial timings in absolute (since session start) time
    prior : array-like
        Vector containing the prior estimate or true prior for each trial. Must be same length as
        trialsdf.
    t_before : float
        Time, in seconds, before stimulus onset that was used to define trial_start in trialsdf
    bases : dict
        Dictionary of basis functions for each regressor. See function internals for which keys
        are necessary
    iti_prior : list, optional
        Two element list defining bounds on which step function for ITI prior is
        applied, by default [-0.4, -0.1]
    contnorm : float, optional
        Normalization factor for contrast, by default 5.
    binwidth : float, optional
        Size of bins to use for design matrix, in seconds, by default 0.02
    """
    trialsdf = trialsdf.iloc[1:-1]
    trialsdf['adj_contrastL'] = np.tanh(contnorm * trialsdf['contrastLeft']) / np.tanh(contnorm)
    trialsdf['adj_contrastR'] = np.tanh(contnorm * trialsdf['contrastRight']) / np.tanh(contnorm)
    trialsdf['prior'] = prior
    trialsdf['prior_last'] = pd.Series(np.roll(trialsdf['prior'], 1), index=trialsdf.index)

    vartypes = {'choice': 'value',
                'response_times': 'timing',
                'probabilityLeft': 'value',
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
                'wheel_velocity': 'continuous',
                'firstMovement_times': 'timing'}

    def stepfunc_prestim(row):
        stepvec = np.zeros(design.binf(row.duration))
        stepvec[stepbounds[0]:stepbounds[1]] = row.prior_last
        return stepvec

    def stepfunc_poststim(row):
        zerovec = np.zeros(design.binf(row.duration))
        currtr_start = design.binf(row.stimOn_times + 0.1)
        currtr_end = design.binf(row.feedback_times)
        zerovec[currtr_start:currtr_end] = row.prior_last
        zerovec[currtr_end:] = row.prior
        return zerovec

    design = dm.DesignMatrix(trialsdf, vartypes, binwidth=binwidth)
    stepbounds = [design.binf(t_before + iti_prior[0]), design.binf(t_before + iti_prior[1])]

    design.add_covariate_timing('stimonL', 'stimOn_times', bases['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastLeft),
                                deltaval='adj_contrastLeft',
                                desc='Kernel conditioned on L stimulus onset')
    design.add_covariate_timing('stimonR', 'stimOn_times', bases['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastRight),
                                deltaval='adj_contrastRight',
                                desc='Kernel conditioned on R stimulus onset')
    design.add_covariate_timing('correct', 'feedback_times', bases['feedback'],
                                cond=lambda tr: tr.feedbackType == 1,
                                desc='Kernel conditioned on correct feedback')
    design.add_covariate_timing('incorrect', 'feedback_times', bases['feedback'],
                                cond=lambda tr: tr.feedbackType == -1,
                                desc='Kernel conditioned on incorrect feedback')
    design.add_covariate_timing('fmove', 'firstMovement_times', bases['fmove'],
                                offset=fmove_offset,
                                desc='Lead up to first movement')
    design.add_covariate_raw('pLeft', stepfunc_prestim,
                             desc='Step function on prior estimate')
    design.add_covariate_raw('pLeft_tr', stepfunc_poststim,
                             desc='Step function on post-stimulus prior')

    design.add_covariate('wheel', trialsdf['wheel_velocity'], bases['wheel'], wheel_offset)
    design.compile_design_matrix()

    if reduce_wheel_dim:
        _, s, v = np.linalg.svd(design[:, design.covar['wheel']['dmcol_idx']],
                                full_matrices=False)
        variances = s**2 / (s**2).sum()
        n_keep = np.argwhere(np.cumsum(variances) >= 0.9999)[0, 0]
        wheelcols = design[:, design.covar['wheel']['dmcol_idx']]
        reduced = wheelcols @ v[:n_keep].T
        bases_reduced = bases['wheel'] @ v[:n_keep].T
        keepcols = ~np.isin(np.arange(design.dm.shape[1]), design.covar['wheel']['dmcol_idx'])
        basedm = design[:, keepcols]
        design.dm = np.hstack([basedm, reduced])
        design.covar['wheel']['dmcol_idx'] = design.covar['wheel']['dmcol_idx'][:n_keep]
        design.covar['wheel']['bases'] = bases_reduced

    print('Condition of design matrix:', np.linalg.cond(design.dm))
    return design
