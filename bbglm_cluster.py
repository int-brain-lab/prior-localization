"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import pickle
from datetime import date
import os
from oneibl import one
from bbglm_bwmfit import fit_session
from utils import sessions_with_region

one = one.ONE()


def fit_and_save(session_id, kernlen, nbases, nickname, sessdate, filename, probe_idx,
                 t_before=1., t_after=0.6, max_len=2., contnorm=5., binwidth=0.02,
                 abswheel=False, no_50perc=False, one=one):
    outtuple = fit_session(session_id, kernlen, nbases, t_before, t_after, max_len, probe_idx,
                           contnorm, binwidth, abswheel, no_50perc, one=one)
    nglm, sequences, scores = outtuple
    outdict = {'sessinfo': {'eid': session_id, 'nickname': nickname, 'sessdate': sessdate},
               'kernlen': kernlen, 'nbases': nbases,
               'binwidth': binwidth, 'sequences': sequences, 'scores': scores,
               'fitobj': nglm}
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
    from glob import glob
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    currdate = str(date.today())
    # currdate = '2021-05-04'

    savepath = '/home/gercek/scratch/fits/'

    target_regions = ['']

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
