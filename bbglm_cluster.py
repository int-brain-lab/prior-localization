"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import pickle
import os
from datetime import date
from glob import glob
from one.api import ONE
from bbglm_sessfit import load_regressors, generate_design
from .decoding.decoding_utils import compute_target, query_sessions


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
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    currdate = str(date.today())
    # currdate = '2021-05-04'

    savepath = '/home/gercek/scratch/fits/'

    sessions = query_sessions('resolved-behavior')

    # Define delayed versions of the fit functions for use in dask
    dload = dask.delayed(load_regressors, nout=5)
    dprior = dask.delayed(compute_target)
    ddesign = dask.delayed(generate_design)
    dfit = dask.delayed(fit)
    dsave = dask.delayed(save)
    
    for i, (subject, eid, probe) in sessions.iterrows():
