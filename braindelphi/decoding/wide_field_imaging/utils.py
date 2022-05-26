import numpy
import glob
from pathlib import Path
import pandas as pd


def load_wfi(path_to_wfi):
    downsampled = numpy.load(path_to_wfi + '/downsampled_ims-%s.npy' % path_to_wfi.split('-')[-1], allow_pickle=True)
    behavior = numpy.load(path_to_wfi + '/behavior.npy', allow_pickle=True)
    ds_atlas = numpy.load(path_to_wfi + '/ds_atlas.npy', allow_pickle=True)
    frame_df = numpy.load(path_to_wfi + '/frame_df.npy', allow_pickle=True)
    ds_map = numpy.load(path_to_wfi + '/ds_map.npy', allow_pickle=True)
    return downsampled, behavior, ds_atlas, ds_map, frame_df


def load_wfi_session(path_to_wfi, sessid):
    downsampled, behavior, ds_atlas, ds_map, frame_df = load_wfi(path_to_wfi)
    atlas = pd.read_csv('wide_field_imaging/ccf_regions.csv')
    for k in range(behavior.size):
        behavior[k]['eid'] = k
        behavior[k]['session_to_decode'] = True if k == sessid else False
    sessiondf = pd.concat(behavior, axis=0).reset_index(drop=True)
    sessiondf['subject'] = 'wfi%i' % int(path_to_wfi.split('-')[-1])
    sessiondf = sessiondf[['choice', 'stimOn_times', 'feedbackType', 'feedback_times', 'contrastLeft', 'goCue_times',
                           'contrastRight', 'probabilityLeft', 'session_to_decode', 'subject', 'signedContrast',
                           'eid', 'firstMovement_times']]
    sessiondf = sessiondf.assign(stim_side=(sessiondf.choice * (sessiondf.feedbackType == 1) -
                                            sessiondf.choice * (sessiondf.feedbackType == -1)))
    sessiondf['eid'] = sessiondf['eid'].apply(lambda x: ('wfi' + str(int(path_to_wfi.split('-')[-1]))
                                                                + 's' + str(x)))
    downsampled = downsampled[sessid]
    frame_df = frame_df[sessid]
    wideFieldImaging_dict = {'activity': downsampled, 'timings': frame_df, 'regions': ds_atlas, 'atlas': atlas}
    return sessiondf, wideFieldImaging_dict


