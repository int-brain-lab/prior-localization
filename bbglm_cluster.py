"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import pickle
import os
from datetime import date
from glob import glob
from pathlib import Path
from one.api import ONE
from brainbox.task.closed_loop import generate_pseudo_session
from bbglm_sessfit import fit, fit_stepwise, generate_design
from .decoding.decoding_utils import compute_target, query_sessions
from params import BEH_MOD_PATH, GLM_FIT_PATH, GLM_CACHE


def get_cached_regressors(fpath):
    with open(fpath, 'rb') as fo:
        d = pickle.load(fo)
    return d['trialsdf'], d['spk_times'], d['spk_clu'], d['clu_regions'], d['clu_qc']


def save_pseudo(subject, session_id, fitout, params):
    # TODO: Make this work
    raise NotImplementedError
    sesspath = _create_sub_sess_path(GLM_FIT_PATH, subject, session_id)
    fn = sesspath.joinpath()
    subjfilepath = os.path.abspath(filename)
    fw = open(subjfilepath, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return True


def save_stepwise(subject, session_id, fitout, params, probes, input_fn, clu_reg, clu_qc, fitdate):
    sesspath = _create_sub_sess_path(GLM_FIT_PATH, subject, session_id)
    fn = sesspath.joinpath(f'{fitdate}_stepwise_regression.pkl')
    outdict = {
        'params': params,
        'probes': probes,
        'model_input_fn': input_fn,
        'clu_regions': clu_reg,
        'clu_qc': clu_qc,
    }
    outdict.update(fitout)
    with open(fn, 'wb') as fw:
        pickle.dump(outdict, fw)
    return fn


def _create_sub_sess_path(parent, subject, session):
    subpath = Path(parent).joinpath(subject)
    if not subpath.exists():
        os.mkdir(subpath)
    sesspath = subpath.joinpath(session)
    if not sesspath.exists():
        os.mkdir(sesspath)
    return sesspath


if __name__ == "__main__":
    import dask
    import numpy as np
    import brainbox.modeling.utils as mut
    import brainbox.modeling.linear as lm
    import sklearn.linear_model as skl
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster
    from sklearn.model_selection import GridSearchCV

    # Model parameters
    def tmp_binf(t):
        return np.ceil(t / params['binwidth']).astype(int)

    params = {
        'binwidth': 0.02,
        'iti_prior': [-0.4, -0.1],
        'fmove_offset': -0.4,
        'wheel_offset': -0.4,
        'contnorm': 5.,
        'reduce_wheel_dim': True,
        'dataset_fn': '2022-01-03_dataset_metadata.pkl',
        'model': lm.LinearGLM,
        'alpha_grid': {'alpha': np.logspace(-2, 1.5, 100)},
        'contiguous': False,
        # 'n_pseudo': 100,
    }

    params['bases'] = {
            'stim': mut.raised_cosine(0.4, 5, tmp_binf),
            'feedback': mut.raised_cosine(0.4, 5, tmp_binf),
            'wheel': mut.raised_cosine(0.3, 3, tmp_binf),
            'fmove': mut.raised_cosine(0.2, 3, tmp_binf),
        }
    params['estimator'] = GridSearchCV(skl.Ridge(), params['alpha_grid'])


    currdate = str(date.today())
    # currdate = '2021-05-04'

    savepath = '/home/gercek/scratch/fits/'

    sessions = query_sessions('resolved-behavior').set_index(['subject', 'eid'])
    with open(Path(GLM_CACHE).joinpath(params['dataset_fn']), 'rb') as fo:
        dataset = pickle.load(fo)
    dataset_params = dataset['params']
    dataset_fns = dataset['dataset_filenames']

    # Define delayed versions of the fit functions for use in dask
    dload = dask.delayed(get_cached_regressors, nout=5)
    dprior = dask.delayed(compute_target)
    dselect_prior = dask.delayed(lambda arr, idx: arr[idx])
    ddesign = dask.delayed(generate_design)
    dpseudo = dask.delayed(generate_pseudo_session)
    dfit = dask.delayed(fit_stepwise)
    dsave = dask.delayed(save_stepwise)
    
    data_fns = []
    for i, (subject, eid, probes, metafn, eidfn) in dataset_fns.iterrows():
        subjeids = sessions.xs(subject, level='subject').index.unique().to_list()
        stdf, sspkt, sspkclu, sclureg, scluqc = dload(eidfn)
        sessfullprior = dprior('prior', subject, subjeids, eid, BEH_MOD_PATH)
        sessprior = dselect_prior(sessfullprior, stdf.index)
        sessdesign = ddesign(stdf, sessprior, dataset_params['t_before'],
                             **params)
        sessfit = dfit(sessdesign, sspkt, sspkclu, **params)
        outputfn = dsave(subject, eid, sessfit, params, probes, eidfn, sclureg, scluqc, currdate)
        data_fns.append(outputfn)

    N_CORES = 4
    cluster = SLURMCluster(cores=N_CORES, memory='32GB', processes=1, queue="shared-cpu",
                           walltime="01:15:00",
                           log_directory='/home/gercek/dask-worker-logs',
                           interface='ib0',
                           extra=["--lifetime", "60m", "--lifetime-stagger", "10m"],
                           job_cpu=N_CORES, env_extra=[f'export OMP_NUM_THREADS={N_CORES}',
                                                       f'export MKL_NUM_THREADS={N_CORES}',
                                                       f'export OPENBLAS_NUM_THREADS={N_CORES}'])
    cluster.adapt(minimum_jobs=0, maximum_jobs=400)
    client = Client(cluster)
    futures = client.compute(data_fns)
    
