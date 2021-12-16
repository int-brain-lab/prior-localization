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
from .decoding.decoding_utils import query_sessions


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

    allsess = []
    for region in target_regions:
        reid, rsess, rprobe = sessions_with_region(region)
        for rs, rp in zip(rsess, rprobe):
            rs['probe'] = [rp]
        allsess.extend(zip(reid, rsess))

    sessdict = {}
    for sess in allsess:
        if sess[0] not in sessdict.keys():
            sessdict[sess[0]] = sess[1]
        elif sessdict[sess[0]]['probe'] != sess[1]['probe']:
            sessdict[sess[0]]['probe'].extend(sess[1]['probe'])

    abswheel = True
    kernlen = 0.6
    nbases = 10
    no_50perc = True
    prior_estimate = None
    binwidth = 0.02

    fit_kwargs = {'binwidth': binwidth, 'abswheel': abswheel,
                  'no_50perc': no_50perc, 'one': one}

    argtuples = []
    for eid in sessdict:
        nickname = sessdict[eid]['subject']
        sessdate = sessdict[eid]['start_time'][:10]
        for probe in sessdict[eid]['probe']:
            filename = savepath +\
                f'{nickname}/{sessdate}_session_{currdate}_{probe}_stepwise_fit.p'
            if not check_fit_exists(filename):
                argtuples.append(
                    (eid, kernlen, nbases, nickname, sessdate,
                     filename, probe)
                )
