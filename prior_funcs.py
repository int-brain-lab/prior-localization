from psytrack.hyperOpt import hyperOpt
import numpy as np
from oneibl import one
from export_funs import trialinfo_to_df


def fit_sess_psytrack(session):
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
    trialdf = trialinfo_to_df(session, maxlen=99)
    choices = trialdf.choice
    # Remap choice values from -1, 1 to 1, 2 (because psytrack demands it)
    newmap = {-1: 1, 1: 2}
    choices.replace(newmap, inplace=True)
    choices = choices.values[1:]
    tmp = trialdf[['contrastLeft', 'contrastRight']]
    sL = tmp.contrastLeft.apply(np.nan_to_num).values
    sL = np.vstack((sL[1:], sL[:-1])).T  # Add a history term going back 1 trial
    sR = tmp.contrastRight.apply(np.nan_to_num).values
    sR = np.vstack((sR[1:], sR[:-1])).T  # History term is prior value of contr on that side

    data = {'inputs': {'sL': sL, 'sR': sR},
            'y': choices, }
    weights = {'bias': 1, 'sL': 2, 'sR': 2}
    K = np.sum(list(weights.values()))
    hyper_guess = {'sigma': [2**-5] * K,
                   'sigInit': 2**5,
                   'sigDay': None}
    opt_list = ['sigma']
    hyp, evd, wMode, hess = hyperOpt(data, hyper_guess, weights, opt_list)
    return wMode, hess['W_std']


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    one = one.ONE()
    session = one.search(subject='ZM_2240', date_range=['2020-01-23', '2020-01-23'],
                         dataset_types=['spikes.clusters'])[0]
    wMode, std = fit_sess_psytrack(session)
    trialnumbers = np.arange(1, wMode.shape[1] + 1)
    trialdf = trialinfo_to_df(session)
    plt.figure(figsize=(9, 9))
    plt.plot(trialnumbers, wMode[0], label='Bias', c='b')
    plt.plot(trialdf.index, 3 * (trialdf.probabilityLeft - 0.5), color='k')
    plt.fill_between(trialnumbers, wMode[0] - std[0], wMode[0] + std[0],
                     color='b', alpha=0.5)
    plt.plot(trialnumbers, wMode[1], label='Weight contrast Left',
             color='orange')
    plt.fill_between(trialnumbers, wMode[1] - std[1], wMode[1] + std[1],
                     color='orange', alpha=0.5)
    plt.plot(trialnumbers, wMode[3], label='Weight contrast Right',
             color='purple')
    plt.fill_between(trialnumbers, wMode[3] - std[3], wMode[3] + std[3],
                     color='purple', alpha=0.5)
    plt.xlabel('Trial number', fontsize=18)
    plt.ylabel('Weight value', fontsize=18)
    plt.title('ZM_2240 (Frontal cortex) Psytrack fit', fontsize=24)
    plt.legend(fontsize=18)
