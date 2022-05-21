# Standard library
import logging

# Third party libraries
import pandas as pd
from tqdm import tqdm

# IBL libraries
import brainbox.io.one as bbone

_logger = logging.getLogger('brainwide')

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
