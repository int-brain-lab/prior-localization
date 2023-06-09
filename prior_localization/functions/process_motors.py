import numpy as np
from scipy import stats
import glob
from datetime import datetime
import pickle

from sklearn.linear_model import RidgeCV

from pathlib import Path
CACHE_PATH = Path(__file__).parent.joinpath('tests', 'fixtures', 'inputs')


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


def preprocess_motors(eid, time_window, cache_path=None):
    # the cache_path argment was added to enable to test this function
    # this cache was generated use the pipelines/04_cache_motor.py script
    if cache_path is None:
        neural_dtype_paths = glob.glob(CACHE_PATH.joinpath('*_motor_metadata.pkl').as_posix())
    else:
        neural_dtype_paths = glob.glob(cache_path.joinpath('*_motor_metadata.pkl').as_posix())

    neural_dtype_dates = [datetime.strptime(p.split('/')[-1].split('_')[0], '%Y-%m-%d %H:%M:%S.%f')
                            for p in neural_dtype_paths]

    path_id = np.argmax(neural_dtype_dates)

    motor_metadata = pickle.load(open(neural_dtype_paths[path_id], 'rb'))

    try :
        regressor_path = motor_metadata['dataset_filenames'][ motor_metadata['dataset_filenames']['eid'] == eid]['reg_file'].values[0]
        print("found cached motor regressors")
    except :
        print('not cached...')
        # preprocess_motors_old(eid,kwargs)
        raise Exception('No cached motor regressors for this session !')
    
    if cache_path is not None:
        regressor_path = cache_path.joinpath(*regressor_path.parts[10:])

    regressors = pickle.load(open(regressor_path, 'rb'))

    motor_signals_of_interest = ['licking', 'whisking_l', 'whisking_r', 'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']
    motor_signals = aggregate_on_timeWindow(regressors, motor_signals_of_interest, time_window)

    return motor_signals


def compute_motor_prediction(eid, target, time_window):

    motor_signals = preprocess_motors(eid, time_window)
    motor_signals_arr = np.squeeze(np.array(motor_signals))
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1]).fit(motor_signals_arr, target)
    print(clf.score(motor_signals_arr, target))
    print(clf.alpha_)
    motor_prediction = clf.predict(motor_signals_arr)

    return motor_prediction














    


