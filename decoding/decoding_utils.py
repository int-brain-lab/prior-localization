import os
import numpy as np
import pandas as pd
import models.utils as mut
from pathlib import Path
from ibllib.atlas import BrainRegions
from iblutil.numerical import ismember
from one.api import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction
from models.expSmoothing_stimside import expSmoothing_stimside
from models.biasedApproxBayesian import biased_ApproxBayesian
from models.biasedBayesian import biased_Bayesian
from models.optimalBayesian import optimal_Bayesian

br = BrainRegions()

modeldispatcher = {expSmoothing_prevAction: 'expSmoothingPrevActions',
                   expSmoothing_stimside: 'expSmoothingStimSides',
                   biased_ApproxBayesian: 'biased_Approxbayesian',
                   biased_Bayesian: 'biased_Bayesian',
                   optimal_Bayesian: 'optimal_bayesian'
                   }

# Loading data and input utilities


def query_sessions(selection='all'):
    '''
    Filters sessions on some canonical filters
    returns subjects: nicknames <- array of size nbSubjects x nbSessions
          eid: eid <- array of size nbSubjects x nbSessions
          probes: indvividual prob identity per eid: probes00 or probes01 <- array of size nbSubjects x nbSessions
    '''
    one = ONE()
    if selection == 'all':
        # Query all ephysChoiceWorld sessions
        ins = one.alyx.rest('insertions', 'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                            'session__qc__lt,50')
    elif selection == 'aligned':
        # Query all sessions with at least one alignment
        ins = one.alyx.rest('insertions', 'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                            'session__qc__lt,50,'
                            'json__extended_qc__alignment_count__gt,0')
    elif selection == 'resolved':
        # Query all sessions with resolved alignment
        ins = one.alyx.rest('insertions', 'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                            'session__qc__lt,50,'
                            'json__extended_qc__alignment_resolved,True')
    elif selection == 'aligned-behavior':
        # Query sessions with at least one alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                            'session__qc__lt,50,'
                            'json__extended_qc__alignment_count__gt,0,'
                            'session__extended_qc__behavior,1')
    elif selection == 'resolved-behavior':
        # Query sessions with resolved alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                            'session__qc__lt,50,'
                            'json__extended_qc__alignment_resolved,True,'
                            'session__extended_qc__behavior,1')
    else:
        raise ValueError('Invalid selection was passed.'
                         'Must be in [\'all\', \'aligned\', \'resolved\', \'aligned-behavior\','
                         ' \'resolved-behavior\']')

    # Get list of eids and probes
    all_eids = np.array([i['session'] for i in ins])
    all_probes = np.array([i['name'] for i in ins])
    all_subjects = np.array([i['session_info']['subject'] for i in ins])
    retdf = pd.DataFrame({'subject': all_subjects, 'eid': all_eids, 'probe': all_probes})
    retdf.sort_values('subject', inplace=True)
    return retdf


def check_bhv_fit_exists(subject, model, eids, resultpath):
    '''
    subject: subject_name
    eids: sessions on which the model was fitted
    check if the behavioral fits exists
    return Bool and filename
    '''
    trainmeth = 'MCMC'  # This needs to be un-hard-coded if charles changes to diff. methods
    trunc_eids = [eid.split('-')[0] for eid in eids]
    str_sessionuuids = '_'.join(f'sess{k+1}_{eid}' for k, eid in enumerate(trunc_eids))
    if model not in modeldispatcher.keys():
        raise KeyError('Model is not an instance of a model from behavior_models')
    subjmodpath = Path(resultpath).joinpath(Path(subject))
    modstr = modeldispatcher[model]
    filen = f'model_{modstr}_train_{trainmeth}_train_' + str_sessionuuids + '.pkl'
    fullpath = subjmodpath.joinpath(filen)
    return os.path.exists(fullpath), fullpath


def singlesess_fit_load_bhvmod(target, subject, savepath, eid, remove_old=False,
                               modeltype=expSmoothing_prevAction, one=None):
    '''
    load/fit a behavioral model on a single session
    '''
    one = one or ONE()  # Instant. one if not passed

    data = mut.load_session(eid, one=one)
    if not data['choice']:
        raise ValueError('Session choices produced are None. Debug models.utils.load_session.')
    stim_side, stimuli, actions, _ = mut.format_data(data)
    stimuli, actions, stim_side = mut.format_input([stimuli], [actions], [stim_side])
    model = modeltype(savepath, np.array([eid]), subject,
                      actions, stimuli, stim_side)
    model.load_or_train(remove_old=remove_old)
    return model.compute_signal(signal=target)[target].squeeze()


def multisess_fit_load_bhvmod(target, subject, savepath, eids, remove_old=False,
                              modeltype=expSmoothing_prevAction, one=None):
    '''
    load/fit a behavioral model on a multiple sessions
    '''  
    one = one or ONE()

    datadict = {'stim_side': [], 'actions': [], 'stimuli': []}
    for eid in eids:
        data = mut.load_session(eid, one=one)
        if not data['choice']:
            raise ValueError('Session choices produced are None. Debug models.utils.load_session,'
                             f' or remove the eid {eid} from your input list.')
        stim_side, stimuli, actions, _ = mut.format_data(data)
        datadict['stim_side'].append(stim_side)
        datadict['stimuli'].append(stimuli)
        datadict['actions'].append(actions)
    stimuli, actions, stim_side = mut.format_input(datadict['stimuli'], datadict['actions'],
                                                   datadict['stim_sides'])
    eids = np.array(eids)
    model = modeltype(savepath, eids, subject,
                      actions, stimuli, stim_side)
    model.load_or_train(remove_old=remove_old)
    return model.compute_signal(signal=target)[target].squeeze()


def remap_region(ids, source='Allen-lr', dest='Beryl-lr', output='acronym'):
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == 'id':
        return br.id[br.mappings[dest][inds]]
    elif output == 'acronym':
        return br.get(br.id[br.mappings[dest][inds]])['acronym']
    elif output == 'name':
        return br.get(br.id[br.mappings[dest][inds]])['name']
    else:
        return br.get(br.id[br.mappings[dest][inds]])


def compute_target(target, subject, eid):
    """
    Computes regression target for use with regress_target, using subject, eid, and a string
    identifying the target parameter to output a vector of N_trials length containing the target

    Parameters
    ----------
    target : str
        String in ['prior', 'prederr', 'signcont'], indication model-based prior, prediction error,
        or simple signed contrast per trial
    subject : str
        Subject identity in the IBL database, e.g. KS022
    eid : str
        UUID identifying session for which the target will be produced.

    Returns
    -------
    pandas.Series
        Pandas series in which index is trial number, and value is the target
    """
    ## Do some stuff
    return tvec


def regress_target(tvec, binned, estimator,
                   hyperparam_grid=None, test_prop=0.2, interleave_test=True, grid_cv=None):
    """
    Regresses binned neural activity against a target, using a provided sklearn estimator

    Parameters
    ----------
    tvec : pandas.Series
        Series in which trial number is the index and the value is the regression target
    binned : numpy.ndarray
        N_trials X N_neurons array, in which a single element is a spike count on a trial for a
        given neuron
    estimator : sklearn.linear_model estimator
        Estimator from sklearn which provides .fit, .score, and .predict methods. CV estimators
        are NOT SUPPORTED. Must be a normal estimator, which is internally wrapped with
        GridSearchCV
    hyperparam_grid : dict
        Dictionary with key indicating hyperparameter to grid search over, and value being a the
        nodes on the grid. See sklearn.model_selection.GridSearchCV : param_grid for more specs.
        Defaults to None, which means no hyperparameter estimation or GridSearchCV use.
    test_prop : float in (0, 1)
        Proportion of data to hold out for testing after fitting (with or without grid search)
    interleave_test : bool
        Whether or not test trials should be randomly selected from among the experiment. False
        means last 20% of the experiment will be used.
    grid_cv : See sklearn.model_selection.GridSearchCV, cv argument
        passed through to determine how hyperparameter estimation is done.

    Returns
    -------
    dict
        Dictionary of fitting outputs including:
            - Regression score (from estimator)
            - Decoding coefficients
            - Per-trial target values (copy of tvec)
            - Per-trial predictions from model
    """
    ## Do some stuff
    outdict = dict()
    return outdict