from oneibl import one
import numpy as np
from scipy.io import savemat

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
    starttimes = one.load(session_id, 'trials.goCue_times')[0] - t_before
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


def sep_trials_conds(trials, clu_ids):
    

def matexport_alltrials(session_id, details):
    # Load raw one output and then remove overarching data type name (e.g. trails, spikes)
    sess = {}
    # Save some session metadata
    sess['subject_name'] = details['subject']
    sess['session_date'] = details['start_time']
    sess['subject_lab'] = details['lab']
    trials, clu_ids = session_to_trials(session_id, maxlen=1.5)
    # For loop to delete trials with ipsilateral stimulus or zero contrast
    for i, trial in enumerate(trials):
        if (trial['contrastRight'] == np.nan) | (trial['contrastRight'] == 0):
            del trials[i]
    # Save to a matlab file
    sess['trials'] = trials
    sess['clusters'] = clu_ids
    savemat(f"./data/{sess['subject_name']}_{sess['session_date'].split('T')[0]}.mat",
            sess)


if __name__ == "__main__":
    # Get VisP probe from Yang Dan lab, primary visual areas in upper part of probe
    ids, details = one.search(subject='ZM_2240', dataset_types=['spikes.clusters'],
                              date_range=['2020-01-24', '2020-01-24'], details=True)
    matexport_alltrials(ids[0], details[0])
