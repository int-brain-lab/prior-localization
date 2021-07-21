"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import pickle
from datetime import date
import os
from one.api import ONE
from bbglm_sessfit_stimonly import fit_session

one = ONE()


def fit_and_save(session_id, kernlen, nbases, nickname, sessdate, filename, probe,
                 t_before=1., t_after=0.6, max_len=2., contnorm=5., binwidth=0.02,
                 abswheel=False, num_pseudosess=100, target_regressor='prior',
                 prior_estimate='charles'):
    outtuple = fit_session(session_id, kernlen, nbases, t_before, t_after, prior_estimate, max_len,
                           probe, contnorm, binwidth, abswheel, num_pseudosess, False,
                           target_regressor)
    nglm, realscores, scoreslist, weightslist = outtuple
    outdict = {'sessinfo': {'eid': session_id, 'nickname': nickname, 'sessdate': sessdate},
               'kernlen': kernlen, 'nbases': nbases,
               'binwidth': binwidth, 'realscores': realscores, 'scores': scoreslist,
               'weightslist': weightslist, 'fitobj': nglm, 'probe': probe}
    subjfilepath = os.path.abspath(filename)
    fw = open(subjfilepath, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return True


def sessions_with_region(acronym, one=None):
    if one is None:
        one = ONE()
    query_str = (f'channels__brain_region__acronym__icontains,{acronym},'
                 'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                 'probe_insertion__session__qc__lt,50,'
                 '~probe_insertion__json__qc,CRITICAL')
    traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                         django=query_str)
    eids = np.array([i['session']['id'] for i in traj])
    sessinfo = [i['session'] for i in traj]
    probes = np.array([i['probe_name'] for i in traj])
    return eids, sessinfo, probes


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
    import numpy as np
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    target_regions = ['ACA', 'VISa', 'VISrl', 'ORB', 'RSP', 'VISp', 'MRN', 'SCm']

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

    currdate = str(date.today())
    # currdate = '2021-05-04'

    savepath = '/home/gercek/scratch/fits/'

    kernlen = 0.1
    nbases = 3
    t_before = 0.
    t_after = 0.1
    prior_estimate = 'charles'
    max_len = 2.
    abswheel = False
    binwidth = 0.02
    num_pseudosess = 200
    target = 'prior'

    fit_kwargs = {'t_before': t_before, 't_after': t_after, 'prior_estimate': prior_estimate,
                  'max_len': max_len, 'binwidth': binwidth, 'abswheel': abswheel,
                  'num_pseudosess': num_pseudosess, 'target_regressor': target}

    argtuples = []
    for eid in sessdict:
        nickname = sessdict[eid]['subject']
        sessdate = sessdict[eid]['start_time'][:10]
        for probe in sessdict[eid]['probe']:
            filename = savepath +\
                f'{nickname}/{sessdate}_session_{currdate}_{probe}_{target}_fit.p'
            if not check_fit_exists(filename):
                argtuples.append(
                    (eid, kernlen, nbases, nickname, sessdate,
                     filename, probe)
                )

    # !!! NB: This is highly specific to institutional cluster configuration. You WILL need to
    # change this if you want to run on something other than UNIGE-Yggdrasil.

    cluster = SLURMCluster(cores=1, memory='32GB', processes=1, queue="shared-cpu",
                           walltime="01:00:00", log_directory='/home/gercek/dask-worker-logs',
                           interface='eno1',
                           extra=["--lifetime", "1h", "--lifetime-stagger", "4m"],
                           job_cpu=4, env_extra=['export OMP_NUM_THREADS=4',
                                                 'export MKL_NUM_THREADS=4',
                                                 'export OPENBLAS_NUM_THREADS=4'])
    cluster.adapt(0, len(argtuples) / 2)
    client = Client(cluster)
    outputs = []
    for argtuple in argtuples:
        outputs.append(client.submit(fit_and_save, *argtuple, **fit_kwargs,
                                     pure=False))
