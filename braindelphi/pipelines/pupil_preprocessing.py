
import pandas as pd
import numpy as np
import torch
import math
from one.api import ONE
import pandas as pd
import pickle 
import glob

from brainbox.io.one import SessionLoader
from brainbox.task.trials import get_event_aligned_raster

one = ONE()

### parameters for pupil position  preprocessing ###
# time bin size in seconds
# T_BIN = 0.02
# # duration of the interval in seconds
# duration =0.5
# # lag in seconds
# lag = -0.6
# alignement
align_event = 'stimOn_times'
epoch = (-0.6, -0.1)
# => this will give us the pupil position between -0.6 and -0.1s before stimulus onset, with a time bin of 0.02s


### the functions belows are taken from michael pipeline for the dlc preprocessing for the BWM paper ###
# def find_nearest(array,value):
#     idx = np.searchsorted(array, value, side="left")
#     if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
#         return idx-1
#     else:
#         return idx

def get_pupil_pos(eid,df_tracking):

    video_type='left'
    times_l = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy',
                             query_type='auto') 
    center_raw = df_tracking[['x','y']].apply(pd.to_numeric)

    sl = SessionLoader(one, eid)
    sl.load_trials()

    series_x, _ = get_event_aligned_raster(np.asarray(times_l), sl.trials[align_event],
                                           values=np.asarray(center_raw['x']), epoch=epoch, bin=False)
    series_y, _ = get_event_aligned_raster(np.asarray(times_l), sl.trials[align_event],
                                           values=np.asarray(center_raw['y']), epoch=epoch, bin=False)
    # times, _ = get_event_aligned_raster(np.asarray(times_l), sl.trials[align_event],
    #                                     values=np.asarray(times_l), epoch=epoch, bin=False)
    D = np.concatenate((series_x[:, :-1, np.newaxis], series_y[:, :-1, np.newaxis]), axis=2)
    # T = (times - np.asarray(sl.trials[align_event])[:, np.newaxis])[:, :-1]

    return D

def preprocess_pupil_pos(eid,tracking_file,old_files=False):

    if not old_files :
        df_pupil = pd.read_parquet(tracking_file)
        df_pupil['x'] = df_pupil['pupil_top_r_x'] 
        df_pupil['y'] = (df_pupil['pupil_bottom_r_y'] + df_pupil['pupil_top_r_y'])/2
    else :
        df_pupil = pd.read_csv(tracking_file).drop(index=0).reset_index()
        df_pupil['x'] = df_pupil['heatmap_mhcrnn_tracker.1'] 
        df_pupil['y'] = df_pupil['heatmap_mhcrnn_tracker.2']
    # array of size (nbtrials, nb time bins in interval, 2)
    pupil_position = get_pupil_pos(eid,df_pupil)

    return pupil_position

# location of pupil tracking files given by Matt
# files = glob.glob("C:/Users/fphub/Documents/ibl/decoding_results/ephys/pupils/data/dataframes_prior/*")
# eids = [file.replace('\\','.').split('.')[1] for file in files]
files = glob.glob("/home/julia/data/prior_review/dataframes_prior/*")
eids = [file.split('/')[-1].split('.')[0] for file in files]

pupil_positions = []

for i_sess in range(len(files)):
    print(eids[i_sess])
    pupil_position = preprocess_pupil_pos(eids[i_sess],files[i_sess],old_files=False)
    pupil_positions.append(pupil_position)

dict_eye_position = {k:v for k,v in zip(eids,pupil_positions)}

with open('dictionnary_eye_position.pkl', 'wb') as f:
    pickle.dump(dict_eye_position, f)

