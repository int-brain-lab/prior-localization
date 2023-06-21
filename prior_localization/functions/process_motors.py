import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import RidgeCV

from brainbox.io.one import SessionLoader
from brainbox.processing import bincount2D
from brainbox.behavior.dlc import get_licks

from prior_localization.params import MOTOR_BIN

CACHE_PATH = Path(__file__).parent.joinpath('tests', 'fixtures', 'inputs')


sr = {'licking':'T_BIN','whisking_l':'T_BIN', 'whisking_r':'T_BIN',
      'wheeling':'T_BIN','nose_pos':'T_BIN', 'paw_pos_r':'T_BIN',
      'paw_pos_l':'T_BIN'}


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])):
        return idx - 1
    else:
        return idx


def prepare_motor(one, eid, time_window):
    motor_regressors = cut_behavior(one, eid, duration=2, lag=-1, align_event='stimOn_times')  # very large interval
    motor_signals_of_interest = ['licking', 'whisking_l', 'whisking_r', 'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']
    regressors = dict(filter(lambda i: i[0] in motor_signals_of_interest, motor_regressors.items()))
    motor_binned = aggregate_on_timeWindow(regressors, motor_signals_of_interest, time_window)
    return motor_binned


def cut_behavior(one, eid, duration=0.4, lag=-0.6,
                 align_event='stimOn_times', stim_to_stim=False,
                 endTrial=False, pawex=False):
    """
    cut segments of behavioral time series for PSTHs

    param: eid: session eid
    param: align: in stimOn_times, firstMovement_times, feedback_times
    param: lag: time in sec wrt to align time to start segment
    param: duration: length of cut segment in sec
    stim_to_stim will just be align_event stimOn_times and lag 0
    """
    # Initiate session loader and load trials
    sl = SessionLoader(one, eid)
    sl.load_trials()

    # Load wheel velocity , take the absolute and normalize to max
    sl.load_wheel(fs=1/MOTOR_BIN)
    wheel_acc = abs(sl.wheel['velocity'])
    wheel_acc = wheel_acc / max(wheel_acc)

    # Load motion energy and pose estimates
    sl.load_motion_energy(views=['left', 'right'])
    sl.load_pose(views=['left', 'right'], likelihood_thr=0.9)

    # Get the licks from both cameras combined
    lick_times = [get_licks(sl.pose[f'{side}Camera'], sl.pose[f'{side}Camera']['times']) for side in ['left', 'right']]
    lick_times = list(set(sorted(np.concatenate(lick_times))))
    binned_licks, times_licks, _ = bincount2D(lick_times, np.ones(len(lick_times)), MOTOR_BIN)

    # get root of sum of squares for paw position
    paw_pos_r = sl.pose['rightCamera'][['paw_r_x', 'paw_r_y']].pow(2).sum(axis=1, skipna=False).apply(np.sqrt)
    paw_pos_l = sl.pose['leftCamera'][['paw_r_x', 'paw_r_y']].pow(2).sum(axis=1, skipna=False).apply(np.sqrt)

    # continuous time series of behavior and stamps
    behaves = {
        'wheeling': [sl.wheel['times'], wheel_acc],
        'licking': [times_licks, binned_licks[0]],
        'nose_pos': [sl.pose['leftCamera']['times'], sl.pose['leftCamera']['nose_tip_x']],
        'whisking_l': [sl.motion_energy['leftCamera']['times'], sl.motion_energy['leftCamera']['whiskerMotionEnergy']],
        'whisking_r': [sl.motion_energy['rightCamera']['times'], sl.motion_energy['rightCamera']['whiskerMotionEnergy']],
        'paw_pos_r': [sl.pose['rightCamera']['times'], paw_pos_r],
        'paw_pos_l': [sl.pose['leftCamera']['times'], paw_pos_l]
    }

    # Get stimulus side, 0 for NaN, 1 for all values
    sides = (~sl.trials['contrastLeft'].isna()).astype('int')
    # Create dictionary to fill
    D = {'licking': [], 'whisking_l': [], 'whisking_r': [], 'wheeling': [],
         'nose_pos': [], 'paw_pos_r': [], 'paw_pos_l': [],
         'pleft': sl.trials['probabilityLeft'], 'sides': sides, 'choices': sl.trials['choice']}

    # Get the time of the align events and add the lag, if you want the align event to be the start, choose lag 0
    start_times = sl.trials[align_event] + lag
    for tr, start_t in enumerate(start_times):
        for be in behaves:
            times = behaves[be][0]
            series = behaves[be][1]
            start_idx = find_nearest(times, start_t)
            if stim_to_stim:
                end_idx = find_nearest(times, sl.trials['stimOn_times'][tr + 1])
            else:
                if sr[be] == 'T_BIN':
                    end_idx = start_idx + int(duration / MOTOR_BIN)
                else:
                    fs = sr[be]
                    end_idx = start_idx + int(duration * fs)
            if start_idx > len(series):
                print('start_idx > len(series)')
                break
            D[be].append(series[start_idx:end_idx])

    return D


def aggregate_on_timeWindow(regressors, motor_signals_of_interest, time_window):
    # format the signals
    t_min = time_window[0]
    t_max = time_window[1]
    T_bin = 0.02
    i_min = int((t_min + 1)/T_bin)
    i_max =  int((t_max + 1)/T_bin)

    motor_signals = np.zeros((len(regressors['licking']),len(motor_signals_of_interest)))
    for i in range(len(regressors['licking'])):
        for j,motor in enumerate(motor_signals_of_interest) :
            # we add all bin values to get a unique regressor value for decoding interval
            try :
                motor_signals[i][j] = np.nansum(regressors[motor][i][i_min:i_max])
            except :
                print('time bounds reached')
                motor_signals[i][j] = np.nansum(regressors[motor][i]) # TO CORRECT

    # normalize the motor signals
    motor_signals = stats.zscore(motor_signals,axis=0)
    motor_signals = np.expand_dims(motor_signals,1)
    return list(motor_signals)


def compute_motor_prediction(eid, target, time_window):

    motor_signals = aggregate_on_timeWindow(regressors, motor_signals_of_interest, time_window)
    motor_signals_arr = np.squeeze(np.array(motor_signals))
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1]).fit(motor_signals_arr, target)
    print(clf.score(motor_signals_arr, target))
    print(clf.alpha_)
    motor_prediction = clf.predict(motor_signals_arr)

    return motor_prediction














    


