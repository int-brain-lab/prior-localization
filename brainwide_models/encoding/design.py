"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import brainbox.modeling.design_matrix as dm
import brainbox.modeling.utils as mut
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm


def generate_design(trialsdf,
                    prior,
                    t_before,
                    bases,
                    iti_prior=[-0.4, -0.1],
                    fmove_offset=-0.4,
                    wheel_offset=-0.4,
                    contnorm=5.,
                    binwidth=0.02,
                    reduce_wheel_dim=True,
                    **kwargs):
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
    trialsdf['pLeft_last'] = pd.Series(np.roll(trialsdf['probabilityLeft'], 1),
                                       index=trialsdf.index)

    vartypes = {
        'choice': 'value',
        'response_times': 'timing',
        'probabilityLeft': 'value',
        'pLeft_last': 'value',
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
        'firstMovement_times': 'timing'
    }

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

    design.add_covariate_timing('stimonL',
                                'stimOn_times',
                                bases['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastLeft),
                                deltaval='adj_contrastL',
                                desc='Kernel conditioned on L stimulus onset')
    design.add_covariate_timing('stimonR',
                                'stimOn_times',
                                bases['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastRight),
                                deltaval='adj_contrastR',
                                desc='Kernel conditioned on R stimulus onset')
    design.add_covariate_timing('correct',
                                'feedback_times',
                                bases['feedback'],
                                cond=lambda tr: tr.feedbackType == 1,
                                desc='Kernel conditioned on correct feedback')
    design.add_covariate_timing('incorrect',
                                'feedback_times',
                                bases['feedback'],
                                cond=lambda tr: tr.feedbackType == -1,
                                desc='Kernel conditioned on incorrect feedback')
    design.add_covariate_timing('fmove',
                                'firstMovement_times',
                                bases['fmove'],
                                offset=fmove_offset,
                                desc='Lead up to first movement')
    design.add_covariate_raw('pLeft', stepfunc_prestim, desc='Step function on prior estimate')
    design.add_covariate_raw('pLeft_tr',
                             stepfunc_poststim,
                             desc='Step function on post-stimulus prior estimate')

    design.add_covariate('wheel', trialsdf['wheel_velocity'], bases['wheel'], wheel_offset)
    design.compile_design_matrix()

    if reduce_wheel_dim:
        _, s, v = np.linalg.svd(design[:, design.covar['wheel']['dmcol_idx']], full_matrices=False)
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


def fit(design, spk_t, spk_clu, binwidth, model, estimator, n_folds=5, contiguous=False, **kwargs):
    trials_idx = design.trialsdf.index
    nglm = model(design, spk_t, spk_clu, binwidth=binwidth, estimator=estimator)
    splitter = KFold(n_folds, shuffle=not contiguous)
    scores, weights, intercepts, alphas, splits = [], [], [], [], []
    for test, train in splitter.split(trials_idx):
        nglm.fit(train_idx=train, printcond=False)
        if isinstance(estimator, GridSearchCV):
            alphas.append(estimator.best_params_['alpha'])
        elif isinstance(estimator, RegressorMixin):
            alphas.append(estimator.get_params()['alpha'])
        else:
            raise TypeError('Estimator must be a sklearn linear regression instance')
        intercepts.append(nglm.intercepts)
        weights.append(nglm.combine_weights())
        scores.append(nglm.score(testinds=test))
        splits.append({'test': test, 'train': train})
    outdict = {
        'scores': scores,
        'weights': weights,
        'intercepts': intercepts,
        'alphas': alphas,
        'splits': splits
    }
    return outdict


def fit_stepwise(design,
                 spk_t,
                 spk_clu,
                 binwidth,
                 model,
                 estimator,
                 n_folds=5,
                 contiguous=False,
                 **kwargs):
    trials_idx = design.trialsdf.index
    nglm = model(design, spk_t, spk_clu, binwidth=binwidth, estimator=estimator)
    splitter = KFold(n_folds, shuffle=not contiguous)
    sequences, scores, splits = [], [], []
    for test, train in tqdm(splitter.split(trials_idx), desc='Fold', leave=False):
        nglm.traininds = train
        sfs = mut.SequentialSelector(nglm)
        sfs.fit()
        sequences.append(sfs.sequences_)
        scores.append(sfs.scores_)
        # TODO: Extract per-submodel alpha values
        splits.append({'test': test, 'train': train})
    outdict = {'scores': scores, 'sequences': sequences, 'splits': splits}
    return outdict
