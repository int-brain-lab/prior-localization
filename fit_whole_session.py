"""
April 11 2020

Rather than fitting per-condition kernels (as in fit_session_conds.py, which may be
deprecated), fits the entire session in one go, with modulatory gain terms related to
the prior, contrast, etc.

Berk
"""

from export_funs import session_trialwise, filter_trials
from prior_funcs import fit_sess_psytrack
from oneibl import one
from scipy.io import savemat
import pandas as pd
import numpy as np
from iofuns import loadmat
import time
from datetime import date
import pickle
from tqdm import tqdm
import os
import subprocess as sbp

wts_per_kern = 10
kern_length = 0.6
glm_binsize = 0.020
t_bef = 0.4


def fit_session(session_id, subject_name, sessdate, prior_estimate='psytrack',
                max_len=2., probe_idx=0, logging=False):
    trials, clu_ids = session_trialwise(session_id, t_before=t_bef, t_after=kern_length,
                                        probe_idx=probe_idx)
    trials, clu_ids = filter_trials(trials, clu_ids, max_len)
    if prior_estimate == 'psytrack':
        print('Fitting psytrack model...')
        wts, stds = fit_sess_psytrack(session_id, maxlength=max_len, as_df=True)
        prior_est = wts['bias']
    else:
        raise NotImplementedError('Only psytrack currently supported')

    for trial in trials:
        if trial['trialnum'] == 0:
            continue
        trial['prior'] = prior_est.loc[trial['trialnum']]
    trials = list(filter(lambda x: x['trialnum'] != 0, trials))
    datafile = os.path.abspath(f'./data/{subject_name}_{sessdate}_tmp.mat')
    logfile = datafile[:-4] + '.log'
    lf = open(logfile, 'w')
    startargs = ['matlab', '-nodisplay', '-r']
    startcom = 'iblglm_add2path; fitsess('
    endcom = f',{wts_per_kern}, {glm_binsize}, {kern_length}); exit;'
    outdict = {'trials': trials, 'subject_name': subject_name, 'clusters': clu_ids}
    savemat(datafile, outdict)
    startargs.append(startcom + f'\'{datafile}\'' + endcom)
    process = sbp.Popen(startargs, stdin=sbp.PIPE, stdout=lf, stderr=lf)
    dotnum = 1

    # Simple loop to check on the matlab process intermittently and keep the user informed
    while True:
        retcode = process.poll()
        dots = '.' * dotnum
        if retcode is not None:
            if retcode == 1:
                raise RuntimeError('Something went wrong in the MATLAB executable.')
            elif retcode == 0:
                print('Finished fitting.')
                break
        print('Running' + dots, end='\r')
        time.sleep(0.5)
        print(' ' * 12, end='\r')
        dotnum += 1
        if dotnum % 6 == 0:
            dotnum = 1

    lf.close()
    if not logging:
        os.system('rm ./data/' + logfile)

    # Coerce fit data into a pandas array
    nanarr = np.nan * np.ones(wts_per_kern)
    defaultentry = {'stim_L': nanarr, 'stim_R': nanarr, 'fdbck_corr': nanarr,
                    'fdbck_incorr': nanarr, 'prior': np.nan, 'varstim_L': nanarr,
                    'varstim_R': nanarr, 'varfdbck_corr': nanarr, 'varfdbck_incorr': nanarr,
                    'varprior': nanarr}
    nulldict = [{'cell': cell, **defaultentry} for cell in clu_ids]
    allfits = pd.DataFrame(nulldict).set_index('cell')
    print('Loading fit data into pandas array and saving...')
    fits = loadmat(f'./fits/{subject_name}_{sessdate}_tmp_fit.mat')
    for cell in tqdm(fits['cellweights']):
        allfits.loc[cell] = (fits['cellweights'][cell]['stonL']['data'],
                             fits['cellweights'][cell]['stonR']['data'],
                             fits['cellweights'][cell]['fdbckCorr']['data'],
                             fits['cellweights'][cell]['fdbckInc']['data'],
                             fits['cellweights'][cell]['prvec']['data'],
                             fits['cellstats'][cell][:wts_per_kern],
                             fits['cellstats'][cell][wts_per_kern: 2 * wts_per_kern],
                             fits['cellstats'][cell][2 * wts_per_kern: 3 * wts_per_kern],
                             fits['cellstats'][cell][3 * wts_per_kern: 4 * wts_per_kern],
                             fits['cellstats'][cell][-2],)

    if not os.path.exists(os.path.abspath(f'./fits/{subject_name}')):
        os.mkdir(f'./fits/{subject_name}')

    today = str(date.today())
    subjfilepath = os.path.abspath(f'./fits/{subject_name}/'
                                   f'{sessdate}_session_{today}_probe{probe_idx}_fit.p')
    outdict = {'subject': subject_name, 'session_uuid': session_id, 'wts_per_kern': wts_per_kern,
               'kern_length': kern_length, 'glm_binsize': glm_binsize,
               'prior_est': prior_est, 'probe_idx': probe_idx, 'fits': allfits}
    fw = open(subjfilepath, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    os.system('rm ./fits/*.mat')
    return allfits


if __name__ == "__main__":
    SUBJECT = 'ZM_2240'
    KEEPLOGS = True
    DATE = '2020-01-22'
    one = one.ONE()
    ids = one.search(subject=SUBJECT, date_range=[DATE, DATE],
                     dataset_types=['spikes.clusters'])
    fit_session(ids[0], SUBJECT, DATE, logging=KEEPLOGS, probe_idx=0)
