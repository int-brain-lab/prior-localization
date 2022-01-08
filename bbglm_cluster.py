"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import pickle
import os
from datetime import date
from glob import glob
from pathlib import Path
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold, GridSearchCV
from one.api import ONE
from brainbox.task.closed_loop import generate_pseudo_session
from bbglm_sessfit import load_regressors, generate_design
from .decoding.decoding_utils import compute_target, query_sessions
from params import BEH_MOD_PATH, GLM_FIT_PATH


def get_cached_regressors(fpath):
    with open(fpath, 'rb') as fo:
        data = pickle.load(fo)
    return data['trialsdf'], data['spk_times'], data['spk_clu'], data['clu_reg'], data['clu_qc']


def fit(design, spk_t, spk_clu, binwidth, model, estimator, n_folds=5, contiguous=False):
    trials_idx = design.trialsdf.index
    nglm = model(design, spk_t, spk_clu, binwidth=binwidth, estimator=estimator)
    splitter = KFold(n_folds, shuffle=~contiguous)
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


def save(subject, session_id, fitout, params):
    subpath = Path(GLM_FIT_PATH).joinpath(subject)
    if not subpath.exists():
        os.mkdir(subpath)
    sesspath = subpath.joinpath(session_id)
    if not sesspath.exists():
        os.mkdir(sesspath)
    fn = sesspath.joinpath()
    subjfilepath = os.path.abspath(filename)
    fw = open(subjfilepath, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return True


def check_fit_exists(filename):
    if not os.path.exists(savepath + nickname):
        os.mkdir(savepath + nickname)
    subpaths = [n for n in glob(os.path.abspath(filename)) if os.path.isfile(n)]
    if len(subpaths) != 0:
        return True
    else:
        return False


if __name__ == "__main__":
    import dask
    import numpy as np
    import brainbox.modeling.utils as mut
    import brainbox.modeling.linear as lm
    import sklearn.linear_model as skl
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    # Model parameters
    def tmp_binf(t):
        return np.ceil(t / params['binwidth']).astype(int)

    params = {
        'binwidth': 0.02,
        'bases': {
            'stim': mut.raised_cosine(0.4, 5, tmp_binf),
            'feedback': mut.raised_cosine(0.4, 5, tmp_binf),
            'wheel': mut.raised_cosine(0.3, 3, tmp_binf),
            'fmove': mut.raised_cosine(0.2, 3, tmp_binf),
        },
        'iti_prior': [-0.4, -0.1],
        'fmove_offset': -0.4,
        'wheel_offset': -0.4,
        'contnorm': 5.,
        'reduce_wheel_dim': True,
        'dataset_fn': '2022-01-03_dataset_metadata.pkl',
        'model': lm.LinearGLM,
        'estimator': skl.Ridge
    }

    currdate = str(date.today())
    # currdate = '2021-05-04'

    savepath = '/home/gercek/scratch/fits/'

    sessions = query_sessions('resolved-behavior')
    with open(params['dataset_fn'], 'rb') as fo:
        dataset = pickle.load(fo)
    dataset_params = dataset['params']
    dataset_fns = dataset['dataset_filenames'].set_index(['subject', 'eid'])

    # Define delayed versions of the fit functions for use in dask
    dload = dask.delayed(get_cached_regressors, nout=5)
    dprior = dask.delayed(compute_target)
    ddesign = dask.delayed(generate_design)
    dfit = dask.delayed(fit)
    dsave = dask.delayed(save)
    
    for i, (subject, eid, probe) in sessions.iterrows():
        subjeids = sessions.xs(subject, level='subject').eid.to_list()
        probes, eidfn = dataset_fns.loc[subject, eid]
        stdf, sspkt, sspkclu, sclureg, scluqc = dload(eidfn)
        sessprior = dprior('prior', subjeids, subjeids, BEH_MOD_PATH)
        sessdesign = ddesign(stdf, sessprior, dataset_params['t_before'], **params)
