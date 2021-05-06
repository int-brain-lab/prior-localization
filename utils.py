"""
Utility functions for the prior-localization repository
"""
import pandas as pd
from brainbox.io.one import load_trials_df


def get_impostor_df(subject, one, ephys_only=False, tdf_kwargs={}):
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
    ephys_only : bool
        Whether or not to include only ephys sessions in the output dataframe
    tdf_kwargs : dict
        Dictionary of keyword arguments for brainbox.io.one.load_trials_df
    """
    sessions = one.alyx.rest('insertions', 'list',
                             django='session__project__name__icontains,'
                                    'ibl_neuropixel_brainwide_01,'
                                    'session__subject__nickname__icontains,'
                                    f'{subject},'
                                    'session__task_protocol__icontains,'
                                    '_iblrig_tasks_ephysChoiceWorld')
    if not ephys_only:
        bhsessions = one.alyx.rest('insertions', 'list',
                                   django='session__project__name__icontains,'
                                          'ibl_neuropixel_brainwide_01,'
                                          'session__subject__nickname__icontains,'
                                          f'{subject},'
                                          'session__task_protocol__icontains,'
                                          '_iblrig_tasks_biasChoiceWorld')
    else:
        bhsessions = []
    sessions.extend(bhsessions)
    eids = [item['session_info']['id'] for item in sessions]
    dfs = []
    timing_vars = ['feedback_times', 'goCue_times', 'stimOn_times', 'trial_start', 'trial_end']
    t_last = 0
    for eid in eids:
        tmpdf = load_trials_df(eid, one=one, **tdf_kwargs)
        tmpdf[timing_vars] += t_last
        dfs.append(tmpdf)
        t_last = tmpdf.iloc[-1]['trial_end']
    return pd.concat(dfs).reset_index()