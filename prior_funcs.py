import numpy as np
import pandas as pd
from export_funs import trialinfo_to_df
import os
from scipy.special import logsumexp
from itertools import accumulate

def stable_softmax(x):
    z = x - np.expand_dims(np.max(x, axis=1), -1)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1)
    softmax = numerator/np.expand_dims(denominator, -1)
    return softmax

def trunc_exp(n, tau):
    return np.exp(-n/tau) * (n >= 20) * (n <= 100)

def hazard_f(x, tau):
    return trunc_exp(x, tau)/np.sum(trunc_exp(np.linspace(x,x+100,101), tau), axis=0)

def perform_inference(stim_side, figures=False, p_left=None):
    """
    Performs inference in the generative model of the task with the contrasts observed by the mouse
    clamped.
    The inference is realized with a likelihood recursion.

    Input:
    - stim_side: 1D array with -1 and 1 indicating a stimulus on the left and right respectively
    - figures: if you want to look at the results of the inference process. Matplotlib is required
        when figures=True. Also p_left needs to be an input.
    - p_left: 1D array with probability of left trial, only required when figures=True

    Output:
    - `marginal_blocktype` of size (nb_trials, 3) gives at each trial, the prior probability that
        we are in block right-biased `0`,
        unbiased `1`, or left-biased `2`.
    - `marginal_currentlength` of size (nb_trials, 100) gives at each trial, the prior probability
        of the current length of the
        block that we are in.
    - priors of size (nb_trials, 100, 3) gives at each trial the joint likelihood
        p(b_t, l_t, s_{1:(t-1)} | theta) with b_t the block l_t the current length and
        s_t the stimuli (contrast) side
    - h of size (nb_trials, 100, 3) gives at each trial the joint likelihood
        p(b_t, l_t, s_{1:t} | theta) with b_t the block l_t the current length and
        s_t the stimuli (contrast) side
    """
    if figures: assert(p_left is not None), 'if figures is True, you must specify the pLeft'

    nb_trials, nb_blocklengths, nb_typeblocks = len(stim_side), 100, 3
    h      = np.zeros([nb_trials, nb_blocklengths, nb_typeblocks])
    priors = np.zeros([nb_trials, nb_blocklengths, nb_typeblocks]) - np.inf
    tau, gamma = 60, 0.8
    # at the beginning of the task (0), current length is 1 (0) and block type is unbiased (1)
    h[0, 0, 1], priors[0, 0, 1] = 0, 0
    hazard = hazard_f(np.arange(1, 101), tau=tau)
    l = np.concatenate((np.expand_dims(hazard, -1), np.concatenate(
                (np.diag(1 - hazard[:-1]), np.zeros(len(hazard)-1)[np.newaxis]), axis=0)), axis=-1)
    b = np.zeros([len(hazard), 3, 3])
    b[1:][:,0,0], b[1:][:,1,1], b[1:][:,2,2] = 1, 1, 1 # case when l_t > 0
    b[0][0][-1], b[0][-1][0], b[0][1][np.array([0, 2])] = 1, 1, 1./2 # case when l_t = 1
    # transition matrix l_{t-1}, b_{t-1}, l_t, b_t
    t = np.log(np.swapaxes(l[:,:,np.newaxis,np.newaxis]
                           * b[np.newaxis], 1, 2)).reshape(nb_typeblocks * nb_blocklengths, -1)
    priors = priors.reshape(-1, nb_typeblocks * nb_blocklengths)
    h = h.reshape(-1, nb_typeblocks * nb_blocklengths)

    for i_trial in range(nb_trials):
        s = stim_side[i_trial]
        loglks = np.log(np.array([gamma*(s==-1) + (1-gamma)
                                  * (s==1), 1./2, gamma*(s==1) + (1-gamma)*(s==-1)]))

        # save priors
        if i_trial > 0:
            priors[i_trial] = logsumexp(h[i_trial - 1][:, np.newaxis] + t, axis=(0))
        h[i_trial]          = priors[i_trial] + np.tile(loglks, 100)

    priors = priors - np.expand_dims(logsumexp(priors, axis=1), -1)
    h = h - np.expand_dims(logsumexp(h, axis=1), -1)
    priors = priors.reshape(-1, nb_blocklengths, nb_typeblocks)
    h = h.reshape(-1, nb_blocklengths, nb_typeblocks)
    marginal_blocktype     =  np.exp(priors).sum(axis=1)
    marginal_currentlength = np.exp(priors).sum(axis=2)

    if figures:
        import matplotlib.pyplot as plt
        block_id = np.array((p_left==0.5) * 1 + (p_left==0.8) * 2)
        plt.figure(figsize=(15,7))
        plt.subplot(2, 1, 1)
        plt.imshow(marginal_blocktype.T, aspect='auto', label='inferred', cmap='coolwarm')
        plt.plot(block_id, '--', label='actual block', color='black')
        plt.plot(stim_side + 1, 'x', label='stimuli', color='green')
        plt.xlabel('trial')
        plt.ylabel('block type')
        plt.yticks([2, 1, 0], ['left', 'unbiased', 'right'])
        plt.legend(loc='upper right')
        currentlength = np.array(list(accumulate((
                    np.concatenate((np.array([True]), block_id[:-1] == block_id[1:]))),
                            lambda x, y: (x + 1) * (y==True) + 1 * (y==False))))
        plt.subplot(2, 1, 2)
        plt.imshow(marginal_currentlength.T, aspect='auto', label='inferred', cmap='coolwarm')
        plt.plot(currentlength, '--', label='actual block', color='black')
        plt.xlabel('trial')
        plt.ylabel('block current length')
        plt.legend(loc='upper right')

    return marginal_blocktype, marginal_currentlength, priors, h

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
    from psytrack.hyperOpt import hyperOpt
    trialdf = trialinfo_to_df(session, maxlen=maxlength, ret_wheel=False)
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
        wMode = pd.DataFrame(wMode.T, index=trialdf.index[1:], columns=['bias', 'left', 'right'])
    return wMode, hess['W_std']


if __name__ == "__main__":
    # for Charles' function
    import matplotlib.pyplot as plt
    from ibl_pipeline import subject, ephys
    from oneibl.one import ONE

    one = ONE()
    subj_info = one.alyx.rest('subjects', 'list', lab='cortexlab')
    subject_names = [subj['nickname'] for subj in subj_info]
    sess_id, sess_info = one.search(subject='KS022', task_protocol='biased', details=True)
    trialdf = trialinfo_to_df(sess_id[0], maxlen=2.5, ret_wheel=False)
    stim_side = (np.array(np.isnan(trialdf['contrastLeft'])==False) * -1
                 + np.array(np.isnan(trialdf['contrastRight'])==False)) * 1
    marginal_blocktype, marginal_currentlength, priors, h = perform_inference(stim_side,
                                                                              figures=True)

    # for Nick's functions
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
