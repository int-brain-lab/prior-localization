import os
import numpy as np
import models.utils as mut
from pathlib import Path
from ibllib.atlas import BrainRegions
from one.api import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction
from models.expSmoothing_stimside import expSmoothing_stimside
from models.biasedApproxBayesian import biased_ApproxBayesian
from models.biasedBayesian import biased_Bayesian
from models.optimalBayesian import optimal_Bayesian

br = BrainRegions

modeldispatcher = {expSmoothing_prevAction: 'expSmoothingPrevActions',
                   expSmoothing_stimside: 'expSmoothingStimSides',
                   biased_ApproxBayesian: 'biased_Approxbayesian',
                   biased_Bayesian: 'biased_Bayesian',
                   optimal_Bayesian: 'optimal_bayesian'
                   }

# Loading data and input utilities


def query_sessions(selection='all', return_subjects=False):
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
    eids, ind_unique = np.unique(all_eids, return_index=True)
    subjects = all_subjects[ind_unique]
    probes = []
    for i, eid in enumerate(eids):
        probes.append(all_probes[[s == eid for s in all_eids]])
    if return_subjects:
        return eids, probes, subjects
    else:
        return eids, probes


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


def remap_beryl_acr(allen_ids):
    return br.get(ids=allen_ids)['acronym']

  
def remap_beryl_id(allen_ids):
    return br.get(ids=allen_ids)['id']
