from psytrack.hyperOpt import hyperOpt
import numpy as np
import pandas as pd
from oneibl import one
from export_funs import trialinfo_to_df
import os


def fit_sess_psytrack(session, maxlength=2.5, normfac=5., as_df=False):
    '''
    Use Nick's Psytrack code to fit a session. Very slow to run.

    Parameters
    ----------
    session : str
        UUID of the session to fit. Only fits single sessions.

    Returns
    -------
    wMode : np.ndarray
        5 x N_trials array, in which each row is a set of weights per trial for a given type.
        Row 0: Bias
        Row 1: Left contrast
        Row 2: Previous value of left contrast
        Row 3: Right contrast
        Row 4: Previous value of right contrast
    W_std : np.ndarray
        5 x N_trials array. Elementwise corresponds to stdev of each element of wMode.
    '''
    trialdf = trialinfo_to_df(session, maxlen=maxlength)
    choices = trialdf.choice
    # Remap choice values from -1, 1 to 1, 2 (because psytrack demands it)
    newmap = {-1: 1, 1: 2}
    choices.replace(newmap, inplace=True)
    choices = choices.values[1:]
    tmp = trialdf[['contrastLeft', 'contrastRight']]
    sL = tmp.contrastLeft.apply(np.nan_to_num).values[1:].reshape(-1, 1)
    # sL = np.vstack((sL[1:], sL[:-1])).T  # Add a history term going back 1 trial
    sR = tmp.contrastRight.apply(np.nan_to_num).values[1:].reshape(-1, 1)
    # sR = np.vstack((sR[1:], sR[:-1])).T  # History term is prior value of contr on that side
    if normfac is not None:
        sL = np.tanh(normfac * sL) / np.tanh(normfac)
        sR = np.tanh(normfac * sR) / np.tanh(normfac)

    data = {'inputs': {'sL': sL, 'sR': sR},
            'y': choices, }
    weights = {'bias': 1, 'sL': 1, 'sR': 1}
    # K = np.sum(list(weights.values()))
    hyper_guess = {'sigma': 2**-5,
                   'sigInit': 2**5,
                   'sigDay': None}
    opt_list = ['sigma']
    hyp, evd, wMode, hess = hyperOpt(data, hyper_guess, weights, opt_list)
    if as_df:
        wMode = pd.DataFrame(wMode.T, index=trialdf.index[1:], columns=['left', 'right', 'bias'])
    return wMode, hess['W_std']


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ibl_pipeline import subject, ephys
    one = one.ONE()
    sessions = subject.Subject * subject.SubjectProject *\
        ephys.acquisition.Session * ephys.ProbeTrajectory()
    bwm_sess = sessions & 'subject_project = "ibl_neuropixel_brainwide_01"' & \
        'task_protocol = "_iblrig_tasks_ephysChoiceWorld6.2.5"'
    for s in bwm_sess:
        sess_id = str(s['session_uuid'])
        nickname = s['subject_nickname']
        date = str(s['session_start_time'].date())
        if not os.path.exists(f'./fits/{nickname}'):
            os.mkdir(f'./fits/{nickname}')
        plotfile = f'./fits/{nickname}/{date}_psytrackfit.png'
        wMode, std = fit_sess_psytrack(sess_id)
        trialnumbers = np.arange(1, wMode.shape[1] + 1)
        trialdf = trialinfo_to_df(sess_id, maxlen=2.5)
        plt.figure(figsize=(9, 9))
        plt.plot(trialnumbers, wMode[0], label='Bias', c='b')
        plt.plot(range(len(trialdf)), 3 * (trialdf.probabilityLeft - 0.5), color='k')
        plt.fill_between(trialnumbers, wMode[0] - std[0], wMode[0] + std[0],
                         color='b', alpha=0.5)
        plt.plot(trialnumbers, wMode[1], label='Weight contrast Left',
                 color='orange')
        plt.fill_between(trialnumbers, wMode[1] - std[1], wMode[1] + std[1],
                         color='orange', alpha=0.5)
        plt.plot(trialnumbers, wMode[2], label='Weight contrast Right',
                 color='purple')
        plt.fill_between(trialnumbers, wMode[2] - std[2], wMode[2] + std[2],
                         color='purple', alpha=0.5)
        plt.xlabel('Trial number', fontsize=18)
        plt.ylabel('Weight value', fontsize=18)
        plt.title(f'{nickname} Psytrack fit', fontsize=24)
        plt.legend(fontsize=18)
        plt.savefig(plotfile)
        plt.close()
