from pathlib import Path

import numpy as np
import pandas as pd

from behavior_models.utils import build_path


def create_neural_path(output_path, date, neural_dtype, subject, session_id, probe,
                       region_str, target, time_window,  pseudo_ids, add_to_path=None):
    full_path = Path(output_path).joinpath('neural', date, neural_dtype, subject, session_id, probe)
    full_path.mkdir(exist_ok=True, parents=True)

    pseudo_str = f'{pseudo_ids[0]}_{pseudo_ids[-1]}' if isinstance(pseudo_ids, np.ndarray) else str(pseudo_ids)
    time_str = f'{time_window[0]}_{time_window[1]}'.replace('.', '_')
    file_name = f'{region_str}_target_{target}_timeWindow_{time_str}_pseudo_id_{pseudo_str}'
    if add_to_path:
        for k, v in add_to_path.items():
            file_name = f'{file_name}_{k}_{v}'
    return full_path.joinpath(f'{file_name}.pkl')


def check_bhv_fit_exists(subject, model, eids, resultpath, single_zeta):
    """
    Check if the fit for a given model exists for a given subject and session.

    Parameters
    ----------
    subject: str
        Subject nick name
    model: str
        Model class name
    eids: list
        List of session ids for sessions on which model was fitted
    resultpath: str
        Path to the results
    single_zeta: bool
        Whether or not the model was fitted with a single zeta

    Returns
    -------
    bool
        Whether or not the fit exists
    Path
        Path to the fit
    """
    path_results_mouse = f'model_{model}_single_zeta_{single_zeta}'
    trunc_eids = [eid.split('-')[0] for eid in eids]
    filen = build_path(path_results_mouse, trunc_eids)
    subjmodpath = Path(resultpath).joinpath(Path(subject))
    fullpath = subjmodpath.joinpath(filen)
    return fullpath.exists(), fullpath


def compute_mask(trials_df, align_time, time_window, min_len=None, max_len=None, no_unbias=False,
                 min_rt=0.08, max_rt=None, n_trials_crop_end=0):
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
    n_trials_crop_end : int
        number of trials to crop from the end of the session

    Returns
    -------
    pd.Series
        boolean mask of good trials

    """

    # define reaction times
    if "react_times" not in trials_df.keys():
        trials_df["react_times"] = (
            trials_df.firstMovement_times - trials_df.stimOn_times
        )

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
    if max_rt is not None:
        mask = mask & (~(trials_df.react_times > max_rt)).values

    if (
        "goCue_times" in trials_df.columns
        and max_len is not None
        and min_len is not None
    ):
        # get rid of trials that are too short or too long
        start_diffs = trials_df.goCue_times.diff()
        start_diffs.iloc[0] = 2
        mask = mask & ((start_diffs > min_len).values & (start_diffs < max_len).values)

        # get rid of trials with decoding windows that overlap following trial
        tmp = (
            trials_df[align_time].values[:-1] + time_window[1]
        ) < trials_df.trial_start.values[1:]
        tmp = np.concatenate([tmp, [True]])  # include final trial, no following trials
        mask = mask & tmp

    # get rid of trials where animal does not respond
    mask = mask & (trials_df.choice != 0)

    if n_trials_crop_end > 0:
        mask[-int(n_trials_crop_end):] = False

    return mask


def derivative(y):
    dy = np.zeros(y.shape, np.float)
    dy[0:-1] = np.diff(y)
    dy[-1] = y[-1] - y[-2]
    return dy
