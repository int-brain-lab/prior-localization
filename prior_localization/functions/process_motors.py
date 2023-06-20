import math
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import RidgeCV

from brainbox.io.one import SessionLoader
from brainbox.processing import bincount2D

from prior_localization.params import MOTOR_BIN

CACHE_PATH = Path(__file__).parent.joinpath('tests', 'fixtures', 'inputs')


sr = {'licking':'T_BIN','whisking_l':'T_BIN', 'whisking_r':'T_BIN',
      'wheeling':'T_BIN','nose_pos':'T_BIN', 'paw_pos_r':'T_BIN',
      'paw_pos_l':'T_BIN'}


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def prepare_motor(one, eid, time_window):
    motor_regressors = cut_behavior(one, eid, duration=2, lag=-1, query_type='auto')  # very large interval
    motor_signals_of_interest = ['licking', 'whisking_l', 'whisking_r', 'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']
    regressors = dict(filter(lambda i: i[0] in motor_signals_of_interest, motor_regressors.items()))
    motor_binned = aggregate_on_timeWindow(regressors, motor_signals_of_interest, time_window)
    return motor_binned


def cut_behavior(one, eid, duration=0.4, lag=-0.6,
                 align='stimOn_times', stim_to_stim=False,
                 endTrial=False, query_type='remote', pawex=False):
    """
    cut segments of behavioral time series for PSTHs

    param: eid: session eid
    param: align: in stimOn_times, firstMovement_times, feedback_times
    param: lag: time in sec wrt to align time to start segment
    param: duration: length of cut segment in sec
    """
    # Initiate session loader
    sl = SessionLoader(one, eid)

    # get wheel speed and normalize
    sl.load_wheel(fs=1/MOTOR_BIN)
    v = abs(sl.wheel['velocity'])
    v = v / max(v)  # else the units are very small

    # Get wheel moves
    wheelMoves = one.load_object(eid, 'wheelMoves')

    # Load motion energy and dlc
    sl.load_motion_energy(views=['left', 'right'])
    sl.load_pose(views=['left', 'right'], likelihood_thr=0.9)

    # Load trials
    # TODO: we already have this in the main function, not clear it is needed?
    sl.load_trials()

    # TODO: double check that this is doing wht we want it to do
    # get licks using both cameras
    lick_times = []
    for video in ['left', 'right']:
        times = sl.pose[f'{video}Camera']['times']
        licks = []
        for point in ['tongue_end_l', 'tongue_end_r']:
            for c in np.array(sl.pose[f'{video}Camera'][[f'{point}_x', f'{point}_y']]).T:
                thr = np.nanstd(np.diff(c)) / 4
                licks.append(set(np.where(abs(np.diff(c)) > thr)[0]))
        r = sorted(list(set.union(*licks)))
        idx = np.where(np.array(r) < len(times))[0][-1]  # ERROR HERE ...
        lick_times.append(times[r[:idx]])
    binned_licks, times_lick, _ = bincount2D(sorted(np.concatenate(lick_times)), np.ones(len(lick_times)), MOTOR_BIN)

    # get paw position, for each cam separate
    if pawex:
        paw_pos_r0 = sl.pose['rightCamera'][['paw_r_x', 'paw_r_y']]
        paw_pos_l0 = sl.pose['leftCamera'][['paw_r_x', 'paw_r_y']]
    else:
        paw_pos_r0 = (sl.pose['rightCamera']['paw_r_x'] ** 2 + sl.pose['rightCamera']['paw_r_y'] ** 2) ** 0.5
        paw_pos_l0 = (sl.pose['leftCamera']['paw_r_x'] ** 2 + sl.pose['leftCamera']['paw_r_y'] ** 2) ** 0.5

    licking = []
    whisking_l = []
    whisking_r = []
    wheeling = []
    nose_pos = []
    paw_pos_r = []
    paw_pos_l = []

    pleft = []
    sides = []
    choices = []
    T = []
    difs = []  # difference between stim on and last wheel movement
    d = (licking, whisking_l, whisking_r, wheeling,
         nose_pos, paw_pos_r, paw_pos_l,
         pleft, sides, choices, T, difs)
    ds = ('licking', 'whisking_l', 'whisking_r', 'wheeling',
          'nose_pos', 'paw_pos_r', 'paw_pos_l',
          'pleft', 'sides', 'choices', 'T', 'difs')

    D = dict(zip(ds, d))

    # continuous time series of behavior and stamps
    behaves = {'licking': [times_lick, binned_licks[0]],
               'whisking_l': [sl.motion_energy['leftCamera']['times'], sl.motion_energy['leftCamera']['whiskerMotionEnergy']],
               'whisking_r': [sl.motion_energy['rightCamera']['times'], sl.motion_energy['rightCamera']['whiskerMotionEnergy']],
               'wheeling': [sl.wheel['times'], v],
               'nose_pos': [sl.pose['leftCamera']['times'], sl.pose['leftCamera']['nose_tip_x']],
               'paw_pos_r': [sl.pose['rightCamera']['times'], paw_pos_r0],
               'paw_pos_l': [sl.pose['leftCamera']['times'], paw_pos_l0]}

    print('cutting data')
    kk = 0
    for tr in range(sl.trials.shape[0]):

        a = wheelMoves['intervals'][:, 1]

        if stim_to_stim:
            start_t = sl.trials['stimOn_times'][tr]

        elif align == 'wheel_stop':
            start_t = a + lag

        else:
            start_t = sl.trials[align][tr] + lag

        if np.isnan(sl.trials['contrastLeft'][tr]):
            side = 0  # right side stimulus
        else:
            side = 1  # left side stimulus

        sides.append(side)

        if endTrial:
            choices.append(sl.trials['choice'][tr + 1])
        else:
            choices.append(sl.trials['choice'][tr])

        pleft.append(sl.trials['probabilityLeft'][tr])

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

            if (pawex and ('paw' in be)):  # for illustration on frame
                D[be].append([series[0][start_idx:end_idx],
                              series[1][start_idx:end_idx]])
            else:
                # bug inducing
                if start_idx > len(series):
                    print('start_idx > len(series)')
                    break
                D[be].append(series[start_idx:end_idx])

        T.append(tr)
        kk += 1

    print(kk, 'trials used')
    return (D)


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














    


