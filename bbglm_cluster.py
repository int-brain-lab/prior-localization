"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import pickle
from datetime import date
import traceback
import os
from oneibl import one
from bbglm_bwmfit import fit_session
from utils import get_bwm_ins_alyx

one = one.ONE()


def fit_and_save(session_id, kernlen, nbases, nickname, sessdate, filename, probe_idx,
                 t_before=1., t_after=0.6, max_len=2., contnorm=5., binwidth=0.02,
                 abswheel=False, no_50perc=False, one=one):
    outtuple = fit_session(session_id, kernlen, nbases, t_before, t_after, max_len, probe_idx,
                            contnorm, binwidth, abswheel, no_50perc, progress=False, one=one)
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

    currdate = str(date.today())
    currdate = '2021-05-04'
    sessions = get_bwm_ins_alyx(one)

    savepath = '/home/gercek/scratch/fits/'

    abswheel = True
    kernlen = 0.6
    nbases = 10
    no_50perc = True
    prior_estimate = None
    binwidth = 0.02

    fit_kwargs = {'binwidth': binwidth, 'abswheel': abswheel,
                  'no_50perc': no_50perc, 'one': one}

    argtuples = []
    for sessid in sessions:
        info = one.get_details(sessid)

        nickname = info['subject']
        sessdate = info['start_time'][:10]
        for i in range(len(sessions[sessid])):
            probe = i
            filename = savepath +\
                f'{nickname}/{sessdate}_session_{currdate}_probe{probe}_stepwise_fit.p'
            if not check_fit_exists(filename):
                argtuples.append(
                    (sessid, kernlen, nbases, nickname, sessdate,
                     filename, probe)
                )
