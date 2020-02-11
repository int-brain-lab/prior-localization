from oneibl import one
import numpy as np
from scipy.io import savemat

oneinst = one.ONE()
# Get VisP probe from Yang Dan lab, primary visual areas in upper part of probe
ids, details = oneinst.search(subject='DY_010', dataset_types=['spikes.clusters'],
                              date_range=['2020-02-03', '2020-02-05'], details=True)
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
datavals = oneinst.load(ids[0], dataset_types=dtypeslist)
sess_data = dict(zip([x.split('.')[1] for x in dtypeslist], datavals))
sess_data['subject_name'] = details[0]['subject']
sess_data['session_date'] = details[0]['start_time']
sess_data['subject_lab'] = details[0]['lab']
contrasts = np.vstack((sess_data['contrastLeft'], sess_data['contrastRight'])).T
# Extract only contralateral non-zero contrast stimulus on times to test for visual responsiveness
contralat_stimt = sess_data['stimOn_times'][np.isfinite(contrasts[:, 0]) & (contrasts[:, 0] != 0)]
contralat_ctrst = sess_data['contrastRight'][np.isfinite(contrasts[:, 0]) & (contrasts[:, 0] != 0)]

del sess_data['contrastLeft'], sess_data['contrastRight'], sess_data['stimOn_times']

savemat(f"./data/{sess_data['subject_name']}_{sess_data['session_date'].split('T')[0]}.mat",
        sess_data)
