import copy
from datetime import date
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import sklearn.linear_model as sklm
from tqdm import tqdm

from ibllib.atlas import BrainRegions
from behavior_models.models.utils import build_path as build_path_mut


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


def compute_mask(trialsdf, **kwargs):
    trialsdf['react_times'] = trialsdf['firstMovement_times'] - trialsdf['goCue_times']
    mask = trialsdf[kwargs['align_time']].notna() & trialsdf['firstMovement_times'].notna()
    if kwargs['no_unbias']:
        mask = mask & (trialsdf.probabilityLeft != 0.5).values
    if kwargs['min_rt'] is not None:
        mask = mask & (~(trialsdf.react_times < kwargs['min_rt'])).values
    mask = mask & (trialsdf.choice != 0)
    return mask


def get_save_path(
        pseudo_id, subject, eid, probe, region, output_path, time_window, today, target,
        add_to_saving_path):
    subjectfolder = Path(output_path).joinpath(subject)
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
