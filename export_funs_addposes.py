"""
Functions for loading and exporting data into various formats, particularly for transferring data
to MATLAB, and for breaking data into trial-wise representations (with or without spiking data)

By Berk Gercek, Spring 2020
"""
from oneibl import one
import numpy as np
import pandas as pd
import itertools as it
from brainbox.core import TimeSeries
from brainbox.processing import sync
from scipy import interpolate
from pathlib import Path
import alf.io

one = one.ONE()

trialstypes = ['trials.choice',
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


def good_trial(nan_timesteps, threshold = 3):
    """
    Return false when data lost in the continuous 4(default) timesteps.
    """
    if len(nan_timesteps) - threshold <= 0:
        return True
    for i in range(len(nan_timesteps) - threshold):
        if nan_timesteps[i] + threshold == nan_timesteps[i + threshold]:
            return False
    return True


def trialinfo_to_df(session_id,
                    maxlen=None, t_before=0.4, t_after=0.6, ret_wheel=False, ret_abswheel=False, 
                    abs_poses_vel = False, glm_binsize=0.02):
    '''
    Takes all trial-related data types out of Alyx and stores them in a pandas dataframe, with an
    optional limit on the length of trials. Will retain trial numbers from the experiment as
    indices for reference.
    '''
    if ret_wheel and ret_abswheel:
        raise ValueError('wheel and abswheel cannot both be true.')
    starttimes = one.load(session_id, dataset_types=['trials.stimOn_times'])[0]
    endtimes = one.load(session_id, dataset_types=['trials.feedback_times'])[0]

    if maxlen is not None:
        with np.errstate(invalid='ignore'):
            keeptrials = (endtimes - starttimes) <= maxlen
    else:
        keeptrials = range(len(starttimes))
    tmp = one.load(session_id, dataset_types=trialstypes)
    trialdata = {x.split('.')[1]: tmp[i][keeptrials] for i, x in enumerate(trialstypes)}
    trialdata['probabilityLeft'] = remap_trialp(trialdata['probabilityLeft'])
    trialsdf = pd.DataFrame(trialdata)
    if maxlen is not None:
        trialsdf.set_index(np.nonzero(keeptrials)[0], inplace=True)
    trialsdf['trial_start'] = trialsdf['stimOn_times'] - t_before
    trialsdf['trial_end'] = trialsdf['feedback_times'] + t_after
    starttimes = trialsdf['trial_start']
    endtimes = trialsdf['trial_end']
    
    def format_series(times, values, interp='zero'):
        """
        Pre-process a series of data ordered in time across a session:
        1. Spliting for trials
        2. Resampling based on given time bin

        Parameters
            ----------
            times: array
                An ordered timestamps for values
            values: array
                An ordered values associate to each time stamp
        Returns
            -------
            trials: 2D list
                One row for one trial, contains binned data.
        """
        endlast = 0
        trials = []
        for i, (start, end) in enumerate(np.vstack((starttimes, endtimes)).T):
            startind = np.searchsorted(times[endlast:], start) + endlast
            endind = np.searchsorted(times[endlast:], end, side='right') + endlast + 4
            if endind >= len(values):
                print(f'WARNING: data lost since trial {i}')
                for _ in range(i, len(starttimes)):
                    trials.append(np.nan)
                return trials
            endlast = endind
            tr_values = values[startind - 1:endind + 1]
            nan_index = np.where(np.isnan(tr_values))[0]
            if len(nan_index) != 0: # Check for NaN
                if good_trial(nan_index): # Qualify trial data
                    # Avoid failing to interpolation when the first or last value in tr_values is NaN
                    offset = 10
                    if (startind - 1 - offset) < 0:
                        y = values[:endind + 1 + offset]
                        offset = startind - 1
                    else:
                        y = values[startind - 1 - offset:endind + 1 + offset]
                    valid_index = np.where(~np.isnan(y))[0]
                    # Uses scipy's interpolation library to interpolate NaN value
                    finterp = interpolate.interp1d(valid_index, y[valid_index], bounds_error=False, 
                                                   kind=interp, fill_value='extrapolate')
                    tr_values[nan_index] = finterp(nan_index + offset)
                else:
                    trials.append(np.nan)
                    continue        
            tr_times = times[startind - 1:endind + 1] - start
            tr_times[0] = 0.  # Manual previous-value interpolation
            series = TimeSeries(tr_times, tr_values)
            tr_sync = sync(glm_binsize, timeseries=series, interp=interp)
            trialstartind = np.searchsorted(tr_sync.times, 0)
            trialendind = np.ceil((end - start) / glm_binsize).astype(int)
            tr = tr_sync.values[trialstartind:trialendind + trialstartind]
            trials.append(tr)
        return trials

    def get_velocity(position, abs=True):
        velocity = []
        for item in position:
            if np.isnan(item).any():
                velocity.append(np.nan)
            else:
                trial_vel = item[1:] - item[:-1]
                trial_vel = np.insert(trial_vel, 0, 0)
                if abs:
                    velocity.append(np.abs(trial_vel))
                else:
                    velocity.append(trial_vel)
        return velocity

    # Loading wheel data
    wheel = one.load_object(session_id, 'wheel')
    whlpos, whlt = wheel.position, wheel.timestamps
    trials_whlpos = format_series(whlt, whlpos, interp='previous')
    trialsdf['wheel_velocity'] = get_velocity(trials_whlpos, ret_abswheel)

    # Loading 3D poses data, required pre-computed 3D points via IBL_3d.py
    # pose_times = np.load(f'./poses/{session_id}/times_left.npy', allow_pickle=True)
    # poses = np.load(f'./poses/{session_id}/pts3d.npy', allow_pickle=True).item()
    # points = ['paw1', 'paw2', 'nose']
    # coordinates = ['x','y', 'z']
    # for i, p in enumerate(points):
    #     for c in coordinates:
    #         raw_data = poses[f'{c}_coords'][:,i]
    #         position = format_series(pose_times, raw_data, interp='cubic')
    #         trialsdf[f'{p}_{c}'] = position
    #         trialsdf[f'{p}_{c}_velocity'] = get_velocity(position, abs_poses_vel)

    # Loading 2D poses data
    XYs_left, times_left = GetXYs(session_id, 'left')
    XYs_right, times_right = GetXYs(session_id, 'right')
    # Loading each paw from closest camera, which is called paw_r in dataset
    for point, XYs, times in zip(['paw1', 'paw2'], 
                                 [XYs_left, XYs_right], 
                                 [times_left, times_right]):
        for i, c in enumerate(['x','y']):
            raw_data = XYs['paw_r'][i]
            position = format_series(times, raw_data, interp='slinear')
            trialsdf[f'{point}_{c}_velocity'] = get_velocity(position, abs_poses_vel)

    # One-hot encodng lick timing, using data from left camera with higher resolution
    tongue_pos = (XYs_left['tongue_end_r'] + XYs_left['tongue_end_l']).T
    nan_frames = np.where(np.isnan(tongue_pos)) # Where the tongue position is NaN as well as no tongue was detected
    nan_frames = np.unique(nan_frames[0])
    lick_timing = np.ones_like(times_left)
    lick_timing[nan_frames] = 0
    lick_timing = format_series(times_left, lick_timing, interp='zero')
    trialsdf['lick_timing'] = lick_timing

    return trialsdf


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


'''
Get IBL 2D points for a given trial.
Adapted from Michael Schartner's edit;
first version from Sunand Raghupathi, Paninski Lab, 51N84D/3D-Animal-Pose.
'''
def GetXYs(eid, video_type):
    '''
    eid: session id, e.g. '3663d82b-f197-4e8b-b299-7b803a155b84'
    video_type: one of 'left', 'right', 'body'
    '''
    def find_nearest(array, value):
        # array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    dataset_types = ['camera.times',
                     'trials.intervals',
                     'camera.dlc']
    D = one.load(eid, dataset_types=dataset_types, dclass_output=True)
    alf_path = Path(D.local_path[0]).parent.parent / 'alf'
    # that gives cam time stamps and DLC output (change to alf_path eventually)
    cam = alf.io.load_object(alf_path, '%sCamera' % video_type, namespace = 'ibl')
    trials = alf.io.load_object(alf_path, 'trials', namespace = 'ibl')
    # Pick frames during trials
    session_endtime = trials['intervals'][-1][1]
    if session_endtime > cam['times'][-1]:
        print(f"For session {eid}, DLC outputs don't cover the whole session")
    frame_start = find_nearest(cam['times'], trials['intervals'][0][0])
    frame_stop = find_nearest(cam['times'], session_endtime)

    '''
    DLC related stuff
    '''
    # just to read in times for newer data (which has DLC results in pqt format
    times = cam['times'][frame_start:frame_stop]     
    dlc_name = '_ibl_%sCamera.dlc.pqt' % video_type
    dlc_path = alf_path / dlc_name
    cam = pd.read_parquet(dlc_path, engine = "fastparquet")    

    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    
    if video_type != 'body':
        d = list(points) 
        d.remove('tube_top')
        d.remove('tube_bottom')   
        points = np.array(d)

    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x[frame_start:frame_stop], y[frame_start:frame_stop]])
    return XYs, times