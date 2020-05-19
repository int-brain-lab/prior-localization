"""
Functions for loading and exporting data into various formats, particularly for transferring data
to MATLAB, and for breaking data into trial-wise representations (with or without spiking data)

By Berk Gercek, Spring 2020
"""
from oneibl import one
import numpy as np
import pandas as pd
import itertools as it
one = one.ONE()

trialstypes = ['trials.choice',
               'trials.response_times',
               'trials.probabilityLeft',
               'trials.feedbackType',
               'trials.feedback_times',
               'trials.contrastLeft',
               'trials.contrastRight',
               'trials.goCue_times',
               'trials.stimOn_times', ]


def remap_trialp(probs):
    # Block probabilities in trial data aren't accurate and need to be remapped
    validvals = np.array([0.2, 0.5, 0.8])
    diffs = np.abs(np.array([x - validvals for x in probs]))
    maps = diffs.argmin(axis=1)
    return validvals[maps]


def session_trialwise(session_id, probe_idx=0, t_before=0.2, t_after=0.6, wheel=False):
    '''
    Utility function for loading a session from Alyx in a trial-wise format. All times are relative
    to trial start time, as defined by stim on time. Returns a list of dicts in which each element
    is the information about a single trial.
    '''
    # Load trial start and end times
    starttimes = one.load(session_id, 'trials.stimOn_times')[0] - t_before
    endtimes = one.load(session_id, 'trials.feedback_times')[0] + t_after
    # Check to see if t_before and t_after result in overlapping trial windows
    if np.any(starttimes[1:] < endtimes[:-1]):
        raise ValueError("Current values of t_before and t_after result in overlapping trial "
                         "windows.")
    # Load spikes. Throws an error if it is a multi-probe session. Pass probe index in that case.
    try:
        spiket, clu = one.load(session_id, ['spikes.times', 'spikes.clusters'])
    except ValueError:
        spiket = one.load(session_id, ['spikes.times'])[probe_idx]
        clu = one.load(session_id, ['spikes.clusters'])[probe_idx]
    # Get cluster ids
    clu_ids = np.unique(clu)
    # Array indicating whether cluster i spiked during trial j
    trialspiking = np.zeros((clu_ids.max() + 1, len(starttimes)))
    # Container for all trial type objects
    tmp = one.load(session_id, dataset_types=trialstypes)
    # Break container out into a dict with labels
    trialdata = {x.split('.')[1]: tmp[i] for i, x in enumerate(trialstypes)}
    # Fix weird block probabilities in some sessions
    trialdata['probabilityLeft'] = remap_trialp(trialdata['probabilityLeft'])

    # load in wheel position and timestamps if requested
    if wheel:
        whlpos, whlt = one.load(session_id, ['wheel.position', 'wheel.timestamps'])

    # Run a sliding window through the length of a trial, assigning spikes to the appropriate
    # trial identity. st_endlast is the last spike time before the trial ended.
    st_endlast = 0
    wh_endlast = 0
    trials = []
    for i, (start, end) in enumerate(np.vstack((starttimes, endtimes)).T):
        if any(np.isnan((start, end))):
            continue
        st_startind = np.searchsorted(spiket[st_endlast:], start) + st_endlast
        st_endind = np.searchsorted(spiket[st_endlast:], end, side='right') + st_endlast
        st_endlast = st_endind
        # Find which clusters spiked during a trial, and set the i,j element of trialspiking to 1
        # for those clusters which fired a spike.
        trial_clu = np.unique(clu[st_startind:st_endind])
        trialspiking[trial_clu, i] = 1
        # Build a dict of relevant information for the given trial
        trialdict = {x: (trialdata[x][i] if x[-5:] != 'times' else trialdata[x][i] - start)
                     for x in trialdata}
        # Align spike times s.t. trial start = 0
        trialdict['spikes'] = spiket[st_startind:st_endind] - start
        # Clusters for spikes
        trialdict['clu'] = clu[st_startind:st_endind]
        # Actual trial number
        trialdict['trialnum'] = i
        # If wheel data is requested, perform same processing as above on wheel data
        if wheel:
            wh_startind = np.searchsorted(whlt[wh_endlast:], start) + wh_endlast
            wh_endind = np.searchsorted(whlt[wh_endlast:], end, side='right') + wh_endlast + 4
            wh_endlast = wh_endind
            trialdict['wheel_pos'] = whlpos[wh_startind - 1:wh_endind + 1]
            trialdict['wheel_t'] = whlt[wh_startind - 1:wh_endind + 1] - start

        trials.append(trialdict)
    return trials, clu_ids


def trialinfo_to_df(session_id, maxlen=None):
    '''
    Takes all trial-related data types out of Alyx and stores them in a pandas dataframe, with an
    optional limit on the length of trials. Will retain trial numbers from the experiment as
    indices for reference.
    '''
    starttimes = one.load(session_id, 'trials.stimOn_times')[0]
    endtimes = one.load(session_id, 'trials.feedback_times')[0]
    if maxlen is not None:
        with np.errstate(invalid='ignore'):
            keeptrials = (endtimes - starttimes) <= maxlen
    else:
        keeptrials = range(len(starttimes))
    tmp = one.load(session_id, dataset_types=trialstypes)
    trialdata = {x.split('.')[1]: tmp[i][keeptrials] for i, x in enumerate(trialstypes)}
    trialdata['probabilityLeft'] = remap_trialp(trialdata['probabilityLeft'])
    trialdf = pd.DataFrame(trialdata)
    if maxlen is not None:
        trialdf.set_index(np.nonzero(keeptrials)[0], inplace=True)
    return trialdf


def sep_trials_conds(trials):
    '''
    Separate trials (passed as a list of dicts) into different IBL-task conditions, and returns
    trials in a dict with conditions being the keys. Condition key is product of:

    Contrast: Nonzero or Zero
    Bias block: P(Left) = [0.2, 0.5, 0.8]
    Stimulus side: ['Left', 'Right']

    example key is ('Nonzero', 0.5, 'Right')
    '''
    df = pd.DataFrame(trials)
    contr = {'Nonzero': [1., 0.25, 0.125, 0.0625], 'Zero': [0., ]}
    bias = [0.2, 0.5, 0.8]
    stimulus = ['Left', 'Right']
    conditions = it.product(bias, stimulus, contr)
    condtrials = {}
    for b, s, c in conditions:
        trialinds = df[np.isin(df['contrast' + s], contr[c]) & (df['probabilityLeft'] == b)].index
        condtrials[(b, s, c)] = [x for i, x in enumerate(trials) if i in trialinds]
    return condtrials


def filter_trials(trials, clu_ids, max_len=2., recomp_clusters=True):
    keeptrials = []
    if recomp_clusters:
        newclu = np.zeros(clu_ids.max() + 1)
    for i, trial in enumerate(trials):
        if trial['feedback_times'] - trial['stimOn_times'] > max_len:
            continue
        else:
            keeptrials.append(trial)
            if recomp_clusters:
                trialclu = np.unique(trial['clu'])
                newclu[trialclu] = 1
    if recomp_clusters:
        filtered_clu = np.nonzero(newclu)[0]
        return keeptrials, filtered_clu
    else:
        return keeptrials
