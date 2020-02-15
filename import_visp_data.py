from oneibl import one
import numpy as np
from scipy.io import savemat

one = one.ONE()


def session_to_trials(session_id, t_before=0.2, t_after=0.3, maxlen=1.):
    trialstypes = ['trials.choice',
                   'trials.response_times',
                   'trials.probabilityLeft',
                   'trials.feedbackType',
                   'trials.feedback_times',
                   'trials.contrastLeft',
                   'trials.contrastRight',
                   'trials.goCue_times',
                   'trials.stimOn_times', ]
    starttimes = one.load(session_id, 'trials.goCue_times')[0]
    endttimes = one.load(session_id, 'trials.feedback_times')[0]
    with np.errstate(invalid='ignore'):
        keeptrials = (endttimes - starttimes) <= maxlen
    # Check to see if t_before and t_after result in overlapping trial windows
    if np.any(starttimes[keeptrials][1:] - t_before < endttimes[keeptrials][:-1] + t_after):
        raise ValueError("Current values of t_before and t_after result in overlapping trial "
                         "windows.")
    spiket, clu = one.load(session_id, ['spikes.times', 'spikes.clusters'])
    clu_ids = np.unique(clu)
    trialspiking = np.zeros((clu_ids.max() + 1, np.sum(keeptrials)))
    tmp = one.load(session_id, dataset_types=trialstypes)
    trialdata = {x.split('.')[1]: tmp[i][keeptrials] for i, x in enumerate(trialstypes)}
    endlast = 0
    trials = []
    for i, (start, end) in enumerate(np.vstack((starttimes, endttimes)).T[keeptrials]):
        startind = np.searchsorted(spiket[endlast:], start - t_before) + endlast
        endind = np.searchsorted(spiket[endlast:], end + t_after, side='right') + endlast
        endlast = endind
        trial_clu = np.unique(clu[startind:endind])
        trialspiking[trial_clu, i] = 1
        trialdict = {x: (trialdata[x][i] if x[-5:] != 'times' else trialdata[x][i] - start)
                     for x in trialdata}
        trialdict['spikes'] = spiket[startind:endind] - start
        trialdict['clu'] = clu[startind:endind]
        trials.append(trialdict)
    return trials


def matexport_trials_spikes(session_id, details):
    # Load raw one output and then remove overarching data type name (e.g. trails, spikes)
    sess = {}
    # Save some session metadata
    sess['subject_name'] = details['subject']
    sess['session_date'] = details['start_time']
    sess['subject_lab'] = details['lab']
    trials = session_to_trials(session_id, maxlen=1.5)
    # For loop to delete trials with ipsilateral stimulus or zero contrast
    for i, trial in enumerate(trials):
        if (trial['contrastRight'] == np.nan) | (trial['contrastRight'] == 0):
            del trials[i]
    # Save to a matlab file
    savemat(f"./data/{sess['subject_name']}_{sess['session_date'].split('T')[0]}.mat",
            sess)


if __name__ == "__main__":
    oneinst = one.ONE()
    # Get VisP probe from Yang Dan lab, primary visual areas in upper part of probe
    ids, details = oneinst.search(subject='DY_010', dataset_types=['spikes.clusters'],
                                  date_range=['2020-02-03', '2020-02-05'], details=True)
    matexport_trials_spikes(ids[0], details[0])
