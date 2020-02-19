from oneibl import one
import numpy as np
import pandas as pd
import itertools as it
one = one.ONE()


def session_to_trials(session_id, t_before=0.2, t_after=0.6, maxlen=1.):
    trialstypes = ['trials.choice',
                   'trials.response_times',
                   'trials.probabilityLeft',
                   'trials.feedbackType',
                   'trials.feedback_times',
                   'trials.contrastLeft',
                   'trials.contrastRight',
                   'trials.goCue_times',
                   'trials.stimOn_times', ]
    starttimes = one.load(session_id, 'trials.stimOn_times')[0] - t_before
    endttimes = one.load(session_id, 'trials.feedback_times')[0] + t_after
    with np.errstate(invalid='ignore'):
        keeptrials = (endttimes - starttimes) <= maxlen + t_before + t_after
    # Check to see if t_before and t_after result in overlapping trial windows
    if np.any(starttimes[keeptrials][1:] < endttimes[keeptrials][:-1]):
        raise ValueError("Current values of t_before and t_after result in overlapping trial "
                         "windows.")
    spiket, clu = one.load(session_id, ['spikes.times', 'spikes.clusters'])
    clu_ids = np.unique(clu)
    trialspiking = np.zeros((clu_ids.max() + 1, np.sum(keeptrials)))
    tmp = one.load(session_id, dataset_types=trialstypes)
    trialdata = {x.split('.')[1]: tmp[i][keeptrials] for i, x in enumerate(trialstypes)}

    # Block probabilities in trial data aren't accurate and need to be remapped 🙄
    probs = trialdata['probabilityLeft']
    validvals = np.array([0.2, 0.5, 0.8])
    diffs = np.abs(np.array([x - validvals for x in probs]))
    maps = diffs.argmin(axis=1)
    trialdata['probabilityLeft'] = validvals[maps]

    endlast = 0
    trials = []
    for i, (start, end) in enumerate(np.vstack((starttimes, endttimes)).T[keeptrials]):
        startind = np.searchsorted(spiket[endlast:], start) + endlast
        endind = np.searchsorted(spiket[endlast:], end, side='right') + endlast
        endlast = endind
        trial_clu = np.unique(clu[startind:endind])
        trialspiking[trial_clu, i] = 1
        trialdict = {x: (trialdata[x][i] if x[-5:] != 'times' else trialdata[x][i] - start)
                     for x in trialdata}
        trialdict['spikes'] = spiket[startind:endind] - start
        trialdict['clu'] = clu[startind:endind]
        trials.append(trialdict)
    return trials, clu_ids


def sep_trials_conds(trials):
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
