"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import pickle
from datetime import date
import traceback
import dask
import os
from oneibl import one
from bbglm_bwmfit_pseudoblocks import fit_session, get_bwm_ins_alyx

@dask.delayed
def fit_and_save(session_id, kernlen, nbases, nickname, sessdate, filename,
                 t_before=1., t_after=0.6, max_len=2., probe_idx=0,
                 contnorm=5., binwidth=0.02, abswheel=False, no_50perc=False, num_pseudosess=100,
                 target_regressor='pLeft'):
    try:
        outtuple = fit_session(session_id, kernlen, nbases, t_before, t_after, max_len, probe_idx,
                               contnorm, binwidth, abswheel, no_50perc, num_pseudosess,
                               target_regressor)
        nglm, realscores, scoreslist, weightslist = outtuple
        outdict = {'sessinfo': {'eid': sessid, 'nickname': nickname, 'sessdate': sessdate},
                    'kernlen': kernlen, 'nbases': nbases, 'method': method,
                    'binwidth': binwidth, 'realscores': realscores, 'scores': scoreslist,
                    'weightslist': weightslist, 'fitobj': nglm}
        subjfilepath = os.path.abspath(filename)
        fw = open(subjfilepath, 'wb')
        pickle.dump(outdict, fw)
        fw.close()
    except Exception as err:
        return err, traceback.format_exc()
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

    # currdate = str(date.today())
    currdate = '2021-05-04'
    sessions = get_bwm_ins_alyx(one)

    savepath = '/home/berk/scratch/fits/'

    abswheel = True
    kernlen = 0.6
    nbases = 10
    stepwise = True
    wholetrial_step = False
    no_50perc = True
    method = 'pure'
    blocking = False
    prior_estimate = None
    binwidth = 0.02
    n_pseudo = 100
    target = 'pLeft'

    print(f'Fitting {len(sessions)} sessions...')

    fitout = []
    for sessid in sessions:
        info = one.get_details(sessid)

        nickname = info['subject']
        sessdate = info['start_time'][:10]
        for i in range(len(sessions[sessid])):
            probe = i
            filename = savepath + f'{nickname}/{sessdate}_session_{currdate}_probe{probe}_fit.p'
            if not check_fit_exists(filename):
                fitout.append(
                    fit_and_save(sessid, kernlen, nbases, nickname, sessdate,
                                 filename, probe_idx=probe, binwidth=binwidth, abswheel=abswheel,
                                 no_50perc=no_50perc, num_pseudosess=n_pseudo,
                                 target_regressor=target)
                )
