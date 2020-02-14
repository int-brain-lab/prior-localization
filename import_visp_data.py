from oneibl import one
import numpy as np
from scipy.io import savemat


def matexport_trials_spikes(session_id):
    # Which datatypes will be saved to file
    dtypeslist = ['spikes.times',
                  'spikes.clusters',
                  'spikes.depths',
                  'trials.choice',
                  'trials.response_times',
                  'trials.probabilityLeft',
                  'trials.feedbackType',
                  'trials.feedback_times',
                  'trials.contrastLeft',
                  'trials.contrastRight',
                  'trials.stimOn_times', ]
    # Load raw one output and then remove overarching data type name (e.g. trails, spikes)
    datavals = oneinst.load(ids[0], dataset_types=dtypeslist)
    sess = dict(zip([x.split('.')[1] for x in dtypeslist], datavals))
    # Save some session metadata
    sess['subject_name'] = details[0]['subject']
    sess['session_date'] = details[0]['start_time']
    sess['subject_lab'] = details[0]['lab']
    sess['cluster_ids'] = np.unique(sess['clusters'])
    # Get a N x 2 vec of contrast values. NaN is expected.
    contrasts = np.vstack((sess['contrastLeft'], sess['contrastRight'])).T
    # Extract only contralateral non-zero contrast stimulus on times
    contra_stimt = sess['stimOn_times'][np.isfinite(contrasts[:, 0]) & (contrasts[:, 0] != 0)]
    contra_ctrst = sess['contrastRight'][np.isfinite(contrasts[:, 0]) & (contrasts[:, 0] != 0)]
    # Don't need side-separated contrasts anymore
    del sess['contrastLeft'], sess['contrastRight'], sess['stimOn_times']
    # Break individual unit spike times into separate arrays
    sess['spiket'] = {f'clu{x}': sess['times'][sess['clusters'] == x] for x in sess['cluster_ids']}
    del sess['times'], sess['clusters']
    # Add the new filtered contrast times and save
    sess['contrastContra'], sess['stimtContra'] = contra_ctrst, contra_stimt
    savemat(f"./data/{sess['subject_name']}_{sess['session_date'].split('T')[0]}.mat",
            sess)


if __name__ == "__main__":
    oneinst = one.ONE()
    # Get VisP probe from Yang Dan lab, primary visual areas in upper part of probe
    ids, details = oneinst.search(subject='DY_010', dataset_types=['spikes.clusters'],
                                  date_range=['2020-02-03', '2020-02-05'], details=True)
    matexport_trials_spikes(ids[0])
