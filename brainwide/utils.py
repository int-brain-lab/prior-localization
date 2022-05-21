"""
Utility functions for the prior-localization repository
"""
# Standard library
import logging
from pathlib import Path
import sys
import os

# Third party libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# IBL libraries
import brainbox.io.one as bbone
from ibllib.atlas import BrainRegions
from iblutil.numerical import ismember
from one.api import ONE

_logger = logging.getLogger('brainwide')


def query_sessions(selection='all', one=None):
    '''
    Filters sessions on some canonical filters
    returns dataframe with index being EID, so indexing results in subject name and probe
    identities in that EID.
    '''
    one = one or ONE()
    if selection == 'all':
        # Query all ephysChoiceWorld sessions
        ins = one.alyx.rest('insertions',
                            'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                            'session__qc__lt,50')
    elif selection == 'aligned':
        # Query all sessions with at least one alignment
        ins = one.alyx.rest('insertions',
                            'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                            'session__qc__lt,50,'
                            'json__extended_qc__alignment_count__gt,0')
    elif selection == 'resolved':
        # Query all sessions with resolved alignment
        ins = one.alyx.rest('insertions',
                            'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                            'session__qc__lt,50,'
                            'json__extended_qc__alignment_resolved,True')
    elif selection == 'aligned-behavior':
        # Query sessions with at least one alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions',
                            'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                            'session__qc__lt,50,'
                            'json__extended_qc__alignment_count__gt,0,'
                            'session__extended_qc__behavior,1')
    elif selection == 'resolved-behavior':
        # Query sessions with resolved alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions',
                            'list',
                            django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                            'session__qc__lt,50,'
                            'json__extended_qc__alignment_resolved,True,'
                            'session__extended_qc__behavior,1')
    else:
        raise ValueError('Invalid selection was passed.'
                         'Must be in [\'all\', \'aligned\', \'resolved\', \'aligned-behavior\','
                         ' \'resolved-behavior\']')

    #  Get list of eids and probes
    all_eids = np.array([i['session'] for i in ins])
    all_probes = np.array([i['name'] for i in ins])
    all_subjects = np.array([i['session_info']['subject'] for i in ins])
    all_pids = np.array([i['id'] for i in ins])
    retdf = pd.DataFrame({
        'subject': all_subjects,
        'eid': all_eids,
        'probe': all_probes,
        'pid': all_pids
    })
    retdf.sort_values('subject', inplace=True)
    return retdf


def get_impostor_df(subject,
                    one,
                    ephys=False,
                    tdf_kwargs={},
                    progress=False,
                    ret_template=False,
                    max_sess=10000):
    """
    Produce an impostor DF for a given subject, i.e. a dataframe which joins all trials from
    ephys sessions for that mouse. Will have an additional column listing the source EID of each
    trial.

    Parameters
    ----------
    subject : str
        Subject nickname
    one : oneibl.one.ONE instance
        ONE instance to use for data loading
    ephys : bool
        Whether to use ephys sessions (True) or behavior sessions (False)
    tdf_kwargs : dict
        Dictionary of keyword arguments for brainbox.io.one.load_trials_df
    progress : bool
        Whether or not to display a progress bar of sessions processed
    ret_template : bool
        Whether to return the template session identity (for ephys sessions)
    """
    if ephys:
        sessions = one.alyx.rest('insertions',
                                 'list',
                                 django='session__project__name__icontains,'
                                 'ibl_neuropixel_brainwide_01,'
                                 'session__subject__nickname__icontains,'
                                 f'{subject},'
                                 'session__task_protocol__icontains,'
                                 '_iblrig_tasks_ephysChoiceWorld,'
                                 'session__qc__lt,50,'
                                 'session__extended_qc__behavior,1')
        eids = [item['session_info']['id'] for item in sessions]
    else:
        eids = one.search(project='ibl_neuropixel_brainwide_01',
                          task_protocol='biasedChoiceWorld',
                          subject=[subject] if len(subject) > 0 else [])
    if len(eids) > max_sess:
        rng = np.random.default_rng()
        eids = rng.choice(eids, size=max_sess, replace=False)

    dfs = []
    timing_vars = [
        'feedback_times', 'goCue_times', 'stimOn_times', 'trial_start', 'trial_end',
        'firstMovement_times'
    ]
    for eid in tqdm(eids, desc='Eid :', leave=False, disable=not progress):
        try:
            tmpdf = bbone.load_trials_df(eid, one=one, **tdf_kwargs)
        except Exception as e:
            _logger.warning(f'eid {eid} df load failed with exception {e}.')
            continue
        if ret_template and ephys:
            det = one.get_details(eid, full=True)
            tmpdf['template'] = det['json']['SESSION_ORDER'][det['json']['SESSION_IDX']]
        tmpdf[timing_vars] = tmpdf[timing_vars].subtract(tmpdf['trial_start'], axis=0)
        tmpdf['orig_eid'] = eid
        tmpdf['duration'] = tmpdf['trial_end'] - tmpdf['trial_start']
        dfs.append(tmpdf)
    return pd.concat(dfs).reset_index()


def sessions_with_region(acronym, one=None):
    if one is None:
        one = ONE()
    query_str = (f'channels__brain_region__acronym__icontains,{acronym},'
                 'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                 'probe_insertion__session__qc__lt,50,'
                 '~probe_insertion__json__qc,CRITICAL')
    traj = one.alyx.rest('trajectories',
                         'list',
                         provenance='Ephys aligned histology track',
                         django=query_str)
    eids = np.array([i['session']['id'] for i in traj])
    sessinfo = [i['session'] for i in traj]
    probes = np.array([i['probe_name'] for i in traj])
    return eids, sessinfo, probes


def make_batch_slurm(filename,
                     scriptpath,
                     job_name='brainwide',
                     partition='shared-cpu',
                     time='01:00:00',
                     condapath=Path('~/mambaforge/'),
                     envname='iblenv',
                     logpath=Path('~/worker-logs/'),
                     cores_per_job=4,
                     memory='16GB',
                     array_size='1-100',
                     f_args=[]):
    fw = open(filename, 'wt')
    fw.write('#!/bin/sh\n')
    fw.write(f'#SBATCH --job-name={job_name}\n')
    fw.write(f'#SBATCH --time={time}\n')
    fw.write(f'#SBATCH --partition={partition}\n')
    fw.write(f'#SBATCH --array={array_size}\n')
    fw.write(f'#SBATCH --output={logpath.joinpath(job_name)}.%a.out\n')
    fw.write('#SBATCH --ntasks=1\n')
    fw.write(f'#SBATCH --cpus-per-task={cores_per_job}\n')
    fw.write(f'#SBATCH --mem={memory}\n')
    fw.write('\n')
    fw.write(f'source {condapath}/bin/activate\n')
    fw.write(f'conda activate {envname}\n')
    fw.write(f'python {scriptpath} {" ".join(f_args)}\n')
    fw.close()
    return
