"""
Feb 28 2020

Core of this repo for now. Loads in session given, breaks in to chunks of trials for each
combination of experimental conditions, and then fits GLMs for each one in parallel.

The actual GLM fitting is done through a MATLAB call via the os, which is clunky but also the most
stable implementation given that MATLAB's python hooks are trash and version-dependent on buying
the latest version of MATLAB if you use Python >= 3.7.

This is ugly and should be improved soon.

By Berk
"""

from export_data import session_to_trials, sep_trials_conds
import os
import numpy as np
import pandas as pd
from oneibl import one
import subprocess as sbp
from scipy.io import savemat
from iofuns import loadmat
import time
from datetime import date
import pickle
from tqdm import tqdm


wts_per_kern = 10
kern_length = 0.6  # seconds
glm_binsize = 0.025  # seconds


def fit_cond(condtrials, cond, sess_info):
    bias, stimside, contr = cond
    fittrials = condtrials[cond]
    datafile = os.path.abspath(f'./data/bias_{bias}_{stimside}Trial_{contr}contrast.mat')
    logfile = datafile[:-4] + '.log'
    lf = open(logfile, 'w')
    startargs = ['matlab', '-nodisplay', '-r', ]
    startcom = 'iblglm_add2path; full_fit('
    endcom = '); exit;'

    outdict = {'trials': fittrials, 'subject_name': sess_info['subject'],
               'clusters': sess_info['clu_ids']}
    savemat(datafile, outdict)
    startargs.append(startcom + f'\'{datafile}\'' + endcom)
    process = sbp.Popen(startargs, stdin=sbp.PIPE, stdout=lf, stderr=lf)
    # os.remove(datafile)
    return process, lf


def fit_session(session_id, subject_name, sessdate, batch_size, log=False):
    # Take the session data and extract trial-by-trial spike times using export_data's s2tr fun
    trials, clu_ids = session_to_trials(session_id, t_after=0.5)
    # Break trials apart into different condition sets
    condtrials = sep_trials_conds(trials)
    condkeys = list(condtrials.keys())
    sess_info = {'subject': subject_name, 'clu_ids': clu_ids}
    # Empty lists to store processes running fits and the logs they produce in
    procs = []
    logs = []
    # Keep the user informed
    print('\nR = Running\nD = Done\nF = Failed\n')
    # print('\n'.join([f'{i} = {c}' for i, c in enumerate(condkeys)]))
    # Iterate over conditions. For each condition spawn a process running a matlab instance
    # That will fit all cells in the session for the given condition. When the batch size is
    # reached, stop iterating for a while (loop) and let the fits run.
    # after the whole batch of fits is done move on to the next
    for i in range(len(condkeys)):
        cond = condkeys.pop(0)
        p, f = fit_cond(condtrials, cond, sess_info)
        procs.append(p)
        logs.append(f)
        if (i + 1) % batch_size == 0:
            print('\nNew batch of fits\n')  # flake8: noqa
            statuses = {None: 'R', 1: 'F', 0: 'D'}
            while True:
                retcodes = [p.poll() for p in procs]
                procstat = [(i, statuses[c]) for i, c in enumerate(retcodes)]
                if all(r is not None for r in retcodes):
                    _ = [f.close() for f in logs]
                    break
                print(''.join([c for i, c in procstat]), end='\r')
                time.sleep(0.1)
            os.system('rm ./data/*contrast.mat')
            procs = []
            logs = []
            if not log:
                os.system('rm ./data/*.log')

    # This is a bit gross, but makes for a pretty pandas DF with easy indexing.
    # Start by filling the template dict with NaN values so we can know later which cells weren't
    # fit via a glm.
    nanarr = np.nan * np.ones(wts_per_kern)
    defaultentry = {'stimOn': nanarr, 'fdbck': nanarr, 'bias': np.nan,
                    'stimOnStat': nanarr, 'fdbckStat': nanarr, 'biasStat': np.nan}
    allfits = []
    for clu in clu_ids:
        for b, s, c in condtrials.keys():
            updict = {'cell_name': f'cell{clu}', 'stim': s, 'contr': c, 'bias': b}
            currcopy = defaultentry.copy()
            currcopy.update(updict)
            allfits.append(currcopy)
    # We will use multi-indexing with pandas to identify fit entries.
    allfits = pd.DataFrame(allfits).set_index(['cell_name', 'stim', 'bias', 'contr'])
    # iterate through each fit, add the results to the dataframe
    print('Converting all fits to pandas data frame')
    for i, cond in tqdm(enumerate(condtrials.keys())):
        b, s, c = cond
        filename = f'./fits/bias_{b}_{s}Trial_{c}contrast_fit.mat'
        fit = loadmat(filename)
        for cell in fit['cellweights']:
            allfits.loc[cell, s, b, c] = (fit['cellweights'][cell]['stimOn']['data'],
                                          fit['cellweights'][cell]['feedback_t']['data'],
                                          fit['cellstats'][cell][:wts_per_kern],
                                          fit['cellstats'][cell][wts_per_kern:-1],
                                          fit['cellstats'][cell][-1])

    if not os.path.exists(os.path.abspath(f'./fits/{subject_name}')):
        os.mkdir(f'./fits/{subject_name}')

    today = str(date.today())
    subjfilepath = os.path.abspath(f'./fits/{subject_name}/{sessdate}_session_{today}_fit.p')
    outdict = {'subject': subject_name, 'session_uuid': session_id, 'wts_per_kern': wts_per_kern,
               'kern_length': kern_length, 'glm_binsize': glm_binsize, 'fits': allfits}
    fw = open(subjfilepath, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    os.system('rm ./fits/*.mat')
    return


if __name__ == "__main__":
    SUBJECT = 'ZM_2240'
    KEEPLOGS = False
    BATCH_SIZE = 15  # Number of parallel fits
    DATE = '2020-01-21'
    one = one.ONE()
    ids = one.search(subject=SUBJECT, date_range=[DATE, DATE],
                     dataset_types=['spikes.clusters'])
    fit_session(ids[0], SUBJECT, DATE, BATCH_SIZE)
