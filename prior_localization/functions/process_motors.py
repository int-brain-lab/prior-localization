import math
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import RidgeCV

from brainbox.io.one import SessionLoader
from brainbox.processing import bincount2D
import brainbox.behavior.wheel as wh

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


def get_licks(XYs):
    '''
    define a frame as a lick frame if
    x or y for left or right tongue point
    change more than half the sdt of the diff
    '''

    licks = []
    for point in ['tongue_end_l', 'tongue_end_r']:
        for c in XYs[point]:
            thr = np.nanstd(np.diff(c)) / 4
            licks.append(set(np.where(abs(np.diff(c)) > thr)[0]))
    return sorted(list(set.union(*licks)))


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
    sl = SessionLoader(one, eid)
    # get wheel speed and normalize
    # wheel = one.load_object(eid, 'wheel', query_type=query_type)
    # pos, times_w = wh.interpolate_position(wheel.timestamps,
    #                                        wheel.position, freq=1/MOTOR_BIN)
    # v = np.append(np.diff(pos),np.diff(pos)[-1])
    # v = abs(v)
    # v = v/max(v)  # else the units are very small
    sl.load_wheel(fs=1/MOTOR_BIN)
    v = np.append(np.diff(sl.wheel['position']), np.diff(sl.wheel['position'])[-1])
    v = abs(v)
    v = v / max(v)  # else the units are very small

    # get motion energy
    sl.load_motion_energy(views=['left', 'right'])
    sl.load_pose(views=['left', 'right'], likelihood_thr=0.9)

    # load whisker motion energy, separate for both cams
    times_me_l, whisking_l0 = sl.motion_energy['leftCamera']['times'], sl.motion_energy['leftCamera'][
        'whiskerMotionEnergy']
    times_me_r, whisking_r0 = sl.motion_energy['rightCamera']['times'], sl.motion_energy['rightCamera'][
        'whiskerMotionEnergy']

    points = np.unique(['_'.join(x.split('_')[:-1]) for x in sl.pose['leftCamera'].columns])
    times_l, XYs_l = sl.pose['leftCamera']['times'], {point: np.array([sl.pose['leftCamera'][f'{point}_x'],
                                                                       sl.pose['leftCamera'][f'{point}_y']])
                                                      for point in points if point != ''}
    times_r, XYs_r = sl.pose['rightCamera']['times'], {point: np.array([sl.pose['rightCamera'][f'{point}_x'],
                                                                        sl.pose['rightCamera'][f'{point}_y']])
                                                       for point in points if point != ''}

    DLC = {'left': [times_l, XYs_l], 'right': [times_r, XYs_r]}

    # get licks using both cameras
    lick_times = []
    for video_type in ['right', 'left']:
        times, XYs = DLC[video_type]
        r = get_licks(XYs)
        try:
            idx = np.where(np.array(r) < len(times))[0][-1]  # ERROR HERE ...
            lick_times.append(times[r[:idx]])
        except:
            print('ohoh')

    lick_times = sorted(np.concatenate(lick_times))
    R, times_lick, _ = bincount2D(lick_times, np.ones(len(lick_times)), MOTOR_BIN)
    lcs = R[0]
    # get paw position, for each cam separate

    if pawex:
        paw_pos_r0 = XYs_r['paw_r']
        paw_pos_l0 = XYs_l['paw_r']
    else:
        paw_pos_r0 = (XYs_r['paw_r'][0] ** 2 + XYs_r['paw_r'][1] ** 2) ** 0.5
        paw_pos_l0 = (XYs_l['paw_r'][0] ** 2 + XYs_l['paw_r'][1] ** 2) ** 0.5

    # get nose x position from left cam only
    nose_pos0 = XYs_l['nose_tip'][0]
    licking = []
    whisking_l = []
    whisking_r = []
    wheeling = []
    nose_pos = []
    paw_pos_r = []
    paw_pos_l = []

    DD = []

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
    behaves = {'licking': [times_lick, lcs],
               'whisking_l': [times_me_l, whisking_l0],
               'whisking_r': [times_me_r, whisking_r0],
               'wheeling': [sl.wheel['times'], v],
               'nose_pos': [times_l, nose_pos0],
               'paw_pos_r': [times_r, paw_pos_r0],
               'paw_pos_l': [times_l, paw_pos_l0]}
    trials = one.load_object(eid, 'trials', query_type=query_type)
    wheelMoves = one.load_object(eid, 'wheelMoves', query_type=query_type)

    print('cutting data')
    trials = one.load_object(eid, 'trials', query_type=query_type)
    evts = ['stimOn_times', 'feedback_times', 'probabilityLeft',
            'choice', 'feedbackType', 'firstMovement_times']

    kk = 0
    for tr in range(len(trials['intervals'])):

        a = wheelMoves['intervals'][:, 1]

        if stim_to_stim:
            start_t = trials['stimOn_times'][tr]

        elif align == 'wheel_stop':
            start_t = a + lag

        else:
            start_t = trials[align][tr] + lag

        if np.isnan(trials['contrastLeft'][tr]):
            cont = trials['contrastRight'][tr]
            side = 0  # right side stimulus
        else:
            cont = trials['contrastLeft'][tr]
            side = 1  # left side stimulus

        sides.append(side)

        if endTrial:
            choices.append(trials['choice'][tr + 1])
        else:
            choices.append(trials['choice'][tr])

        pleft.append(trials['probabilityLeft'][tr])

        for be in behaves:
            times = behaves[be][0]
            series = behaves[be][1]
            start_idx = find_nearest(times, start_t)
            if stim_to_stim:
                end_idx = find_nearest(times, trials['stimOn_times'][tr + 1])
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














    


