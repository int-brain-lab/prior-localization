"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

from .params import GLM_CACHE
import os
import hashlib
import logging
import numpy as np
import pandas as pd
import brainbox.modeling.design_matrix as dm
import brainbox.io.one as bbone
import brainbox.metrics.single_units as bbqc
from one.api import ONE
from pathlib import Path
from datetime import datetime as dt

_logger = logging.getLogger('enc-dec')


def load_regressors(session_id, probes,
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
                                    ret_wheel=~abswheel, addtl_types=['firstMovement_times'],
                                    one=one)

    if resolved_alignment:
        spikes, clusters, _ = bbone.load_spike_sorting_fast(session_id,
                                                            dataset_types=dataset_types,
                                                            one=one)
    else:
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(session_id,
                                                                    dataset_types=dataset_types,
                                                                    one=one,
                                                                    aligned=True)
    allspikes, allclu, allreg, allamps, alldepths = [], [], [], [], []
    clumax = 0
    for probe in probes:
        allspikes.append(spikes[probe].times)
        allclu.append(spikes[probe].clusters + clumax)
        allreg.append(clusters[probe].acronym)
        allamps.append(spikes[probe].amps)
        alldepths.append(spikes[probe].depths)
        clumax += np.max(spikes[probe].clusters) + 1
    
    allspikes, allclu, allamps, alldepths = [np.hstack(x)
                                             for x in (allspikes, allclu, allamps, alldepths)]
    sortinds = np.argsort(allspikes)
    spk_times = allspikes[sortinds]
    spk_clu = allclu[sortinds]
    spk_amps = allamps[sortinds]
    spk_depths = alldepths[sortinds]
    clu_regions = np.hstack(allreg)
    if not ret_qc:
        return trialsdf, spk_times, spk_clu, clu_regions

    # TODO: add cluster_ids=np.arange(beryl_reg.size) to quick_unit_metrics
    clu_qc = bbqc.quick_unit_metrics(spk_clu, spk_times, spk_amps, spk_depths,
                                     cluster_ids=np.arange(clu_regions.size))
    return trialsdf, spk_times, spk_clu, clu_regions, clu_qc


def cache_regressors(subject, session_id, probes, regressor_params,
                     trialsdf, spk_times, spk_clu, clu_regions, clu_qc):
    subpath = Path(GLM_CACHE.joinpath(subject))
    if not subpath.exists():
        os.mkdir(subpath)
    sesspath = subpath.join(session_id)
    if not sesspath.exists():
        os.mkdir(sesspath)
    curr_t = dt.now()
    fnbase = str(curr_t.date())
    metadata_fn = fnbase + '_metadata.pkl'
    data_fn = fnbase + '_regressors.pkl'
    regressors = {'trialsdf': trialsdf, 'spk_times': spk_times, 'spk_clu': spk_clu,
                  'clu_regions': clu_regions, 'clu_qc': clu_qc}
    reghash = _hash_dict(regressors)
    metadata = {'subject': subject, 'session_id': session_id, 'probes': probes,
                'regressor_hash': reghash, **regressor_params}
    return


def _hash_dict(d):
    hasher = hashlib.md5()
    sortkeys = sorted(d.keys())
    for k in sortkeys:
        v = d[k]
        if type(v) == np.ndarray:
            hasher.update(v)
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            hasher.update(v.to_string().encode())
        else:
            try:
                hasher.update(v)
            except Exception:
                _logger.warning(f'Key {k} was not able to be hashed. May lead to failure to update'
                                'in cached files if something was changed.')
    return hasher.hexdigest()


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
        Dictionary of basis functions for each regressor. Needs keys 'stim', 'feedback', 'fmove',
        (first movement) and 'wheel'.
    iti_prior : list, optional
        Two element list defining bounds on which step function for ITI prior is
        applied, by default [-0.4, -0.1]
    contnorm : float, optional
        Normalization factor for contrast, by default 5.
    binwidth : float, optional
        Size of bins to use for design matrix, in seconds, by default 0.02
    """
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
                'adj_contrastL': 'value',
                'contrastRight': 'value',
                'adj_contrastR': 'value',
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
                                deltaval='adj_contrastL',
                                desc='Kernel conditioned on L stimulus onset')
    design.add_covariate_timing('stimonR', 'stimOn_times', bases['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastRight),
                                deltaval='adj_contrastR',
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


if __name__ == "__main__":
    import sys
    import brainbox.modeling.utils as mut
    sys.path.append(Path(__file__).parent.joinpath('decoding'))
    from decoding.decoding_utils import compute_target, query_sessions

    eid = '5157810e-0fff-4bcf-b19d-32d4e39c7dfc'
    probe = 'probe00'
    binwidth = 0.02
    modelfit_path = '/home/berk/Documents/Projects/prior-localization/results/'
    one = ONE()

    def tmp_binf(t):
        return np.ceil(t / binwidth).astype(int)
    bases = {
        'stim': mut.raised_cosine(0.4, 5, tmp_binf),
        'feedback': mut.raised_cosine(0.4, 5, tmp_binf),
        'wheel': mut.raised_cosine(0.3, 3, tmp_binf),
        'fmove': mut.raised_cosine(0.2, 3, tmp_binf),
    }

    sessdf = query_sessions('aligned-behavior').set_index(['subject', 'eid'])
    subject = sessdf.xs(eid, level='eid').index[0]
    trialsdf, spk_times, spk_clu, clu_regions, clu_qc = load_regressors(eid, probe,
                                                                        t_after=0.4,
                                                                        t_before=0.4,
                                                                        ret_qc=True)
    train_eids = sessdf.xs(subject, level='subject').index.unique()
    prior = compute_target('prior', subject, train_eids, eid, modelfit_path, one=one)
    nadf = trialsdf.notna()
    nanmask = nadf.loc[:, ['firstMovement_times', 'stimOn_times', 'feedback_times']].all(axis=1)
    design = generate_design(trialsdf[nanmask].copy(), prior[trialsdf.index[nanmask]], 0.4, bases)
