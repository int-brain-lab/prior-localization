import copy
from datetime import date
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import sklearn.linear_model as sklm
from tqdm import tqdm
import glob
from datetime import datetime

from ibllib.atlas import BrainRegions
from behavior_models.models.utils import build_path as build_path_mut


def load_metadata(neural_dtype_path_regex, date=None):
    '''
    Parameters
    ----------
    neural_dtype_path_regex
    date
    Returns
    -------
    the metadata neural_dtype from date if specified, else most recent
    '''
    neural_dtype_paths = glob.glob(neural_dtype_path_regex)
    neural_dtype_dates = [datetime.strptime(p.split('/')[-1].split('_')[0], '%Y-%m-%d %H:%M:%S.%f')
                          for p in neural_dtype_paths]
    if date is None:
        path_id = np.argmax(neural_dtype_dates)
    else:
        path_id = np.argmax(np.array(neural_dtype_dates) == date)
    return pickle.load(open(neural_dtype_paths[path_id], 'rb')), neural_dtype_dates[path_id].strftime("%m-%d-%Y_%H:%M:%S")


def check_bhv_fit_exists(subject, model, eids, resultpath, modeldispatcher):
    '''
    subject: subject_name
    eids: sessions on which the model was fitted
    check if the behavioral fits exists
    return Bool and filename
    '''
    if model not in modeldispatcher.keys():
        raise KeyError('Model is not an instance of a model from behavior_models')
    path_results_mouse = 'model_%s_' % modeldispatcher[model]
    trunc_eids = [eid.split('-')[0] for eid in eids]
    filen = build_path_mut(path_results_mouse, trunc_eids)
    subjmodpath = Path(resultpath).joinpath(Path(subject))
    fullpath = subjmodpath.joinpath(filen)
    return os.path.exists(fullpath), fullpath


def compute_mask(trials_df, align_time, time_window, min_len, max_len, no_unbias, min_rt, **kwargs):
    """Create a mask that denotes "good" trials which will be used for further analysis.

    Parameters
    ----------
    trials_df : dict
        contains relevant trial information like goCue_times, firstMovement_times, etc.
    align_time : str
        event in trial on which to align intervals
        'firstMovement_times' | 'stimOn_times' | 'feedback_times'
    time_window : tuple
        (window_start, window_end), relative to align_time
    min_len : float, optional
        minimum length of trials to keep (seconds), bypassed if trial_start column not in trials_df
    max_len : float, original
        maximum length of trials to keep (seconds), bypassed if trial_start column not in trials_df
    no_unbias : bool
        True to remove unbiased block trials, False to keep them
    min_rt : float
        minimum reaction time; trials with fast reactions will be removed
    kwargs

    Returns
    -------
    pd.Series

    """

    # define reaction times
    if 'react_times' not in trials_df.keys():
        trials_df['react_times'] = trials_df.firstMovement_times - trials_df.goCue_times

    # successively build a mask that defines which trials we want to keep

    # ensure align event is not a nan
    mask = trials_df[align_time].notna()

    # ensure animal has moved
    mask = mask & trials_df.firstMovement_times.notna()

    # get rid of unbiased trials
    if no_unbias:
        mask = mask & (trials_df.probabilityLeft != 0.5).values

    # keep trials with reasonable reaction times
    if min_rt is not None:
        mask = mask & (~(trials_df.react_times < min_rt)).values

    if 'trial_start' in trials_df.columns and max_len is not None and min_len is not None:
        # get rid of trials that are too short or too long
        start_diffs = trials_df.trial_start.diff()
        start_diffs.iloc[0] = 2
        mask = mask & ((start_diffs > min_len).values & (start_diffs < max_len).values)

        # get rid of trials with decoding windows that overlap following trial
        tmp = (trials_df[align_event].values[:-1] + time_window[1]) < trials_df.trial_start.values[1:]
        tmp = np.concatenate([tmp, [True]])  # include final trial, no following trials
        mask = mask & tmp

    # get rid of trials where animal does not respond
    mask = mask & (trials_df.choice != 0)

    return mask


def get_save_path(
        pseudo_id, subject, eid, neural_dtype, probe, region, output_path, time_window, today, target,
        add_to_saving_path):
    subjectfolder = Path(output_path).joinpath(neural_dtype, subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    start_tw, end_tw = time_window
    fn = '_'.join([today, region, 'target', target,
                   'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_'),
                   'pseudo_id', str(pseudo_id), add_to_saving_path]) + '.pkl'
    save_path = probefolder.joinpath(fn)
    return save_path


def save_region_results(fit_result, pseudo_id, subject, eid, probe, region, n_units, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    outdict = {
        'fit': fit_result, 'pseudo_id': pseudo_id, 'subject': subject, 'eid': eid, 'probe': probe,
        'region': region, 'N_units': n_units
    }
    fw = open(save_path, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return save_path


def check_settings(settings):
    """Error check on pipeline settings.

    Parameters
    ----------
    settings : dict

    Returns
    -------
    dict

    """

    # options for decoding targets
    target_options_singlebin = [
        'prior',     # some estimate of the block prior
        'choice',    # subject's choice (L/R)
        'feedback',  # correct/incorrect
        'signcont',  # signed contrast of stimulus
    ]
    target_options_multibin = [
        'wheel-vel',
        'wheel-speed',
        'pupil',
        'l-paw-pos',
        'l-paw-vel',
        'l-paw-speed',
        'l-whisker-me',
        'r-paw-pos',
        'r-paw-vel',
        'r-paw-speed',
        'r-whisker-me',
    ]

    # options for align events
    align_event_options = [
        'firstMovement_times',
        'goCue_times',
        'stimOn_times',
        'feedback_times',
    ]

    # options for decoder
    decoder_options = {
        'linear': sklm.LinearRegression,
        'lasso': sklm.Lasso,
        'ridge': sklm.Ridge,
        'logistic': sklm.LogisticRegression
    }

    params = copy.copy(settings)

    if params['target'] not in target_options_singlebin + target_options_multibin:
        raise NotImplementedError('provided target option \'{}\' invalid; must be in {}'.format(
            params['target'], target_options_singlebin + target_options_multibin
        ))

    if params['align_time'] not in align_event_options:
        raise NotImplementedError('provided align event \'{}\' invalid; must be in {}'.format(
            params['align_time'], align_event_options
        ))

    # map estimator string to sklm class
    params['estimator'] = decoder_options[settings['estimator']]
    if settings['estimator'] == 'logistic':
        params['hyperparam_grid'] = {'C': settings['hyperparam_grid']['C']}
    else:
        params['hyperparam_grid'] = {'alpha': settings['hyperparam_grid']['alpha']}

    params['n_jobs_per_session'] = params['n_pseudo'] // params['n_pseudo_per_job']

    # TODO: settle on 'date' or 'today'
    # update date if not given
    if params['date'] is None or params['date'] == 'today':
        params['date'] = str(date.today())
    params['today'] = params['date']

    # TODO: settle on n_runs or nb_runs
    if 'n_runs' in params:
        params['nb_runs'] = params['n_runs']

    # TODO: settle on align_time or align_event
    params['align_event'] = params['align_time']

    return params
