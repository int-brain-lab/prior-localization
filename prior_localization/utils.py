import numpy as np


def average_data_in_epoch(times, values, trials_df, align_event='stimOn_times', epoch=(-0.6, -0.1)):
    """
    Aggregate values in a given epoch relative to align_event for each trial. For trials for which the align_event
    is NaN or the epoch contains timestamps outside the times array, the value is set to NaN.

    Parameters
    ----------
    times: np.array
        Timestamps associated with values, assumed to be sorted
    values: np.array
        Data to be aggregated in epoch per trial, one value per timestamp
    trials_df: pd.DataFrame
        Dataframe with trials information
    align_event: str
        Event to align to, must be column in trials_df
    epoch: tuple
        Start and stop of time window to aggregate values in, relative to align_event in seconds


    Returns
    -------
    epoch_array: np.array
        Array of average values in epoch, one per trial in trials_df
    """

    # Make sure timestamps and values are arrays and of same size
    times = np.asarray(times)
    values = np.asarray(values)
    if not len(times) == len(values):
        raise ValueError(f'Inputs to times and values must be same length but are {len(times)} and {len(values)}')
    # Make sure times are sorted
    if not np.all(np.diff(times) >= 0):
        raise ValueError('Times must be sorted')
    # Get the events to align to and compute the ideal intervals for each trial
    events = trials_df[align_event].values
    intervals = np.c_[events + epoch[0], events + epoch[1]]
    # Make a mask to exclude trials were the event is nan, or interval starts before or ends after bin_times
    valid_trials = (~np.isnan(events)) & (intervals[:, 0] >= times[0]) & (intervals[:, 1] <= times[-1])
    # This is the first index to include to be sure to get values >= epoch[0]
    epoch_idx_start = np.searchsorted(times, intervals[valid_trials, 0], side='left')
    # This is the first index to exclude (NOT the last to include) to be sure to get values <= epoch[1]
    epoch_idx_stop = np.searchsorted(times, intervals[valid_trials, 1], side='right')
    # Create an array to fill in with the average epoch values for each trial
    epoch_array = np.full(events.shape, np.nan)
    epoch_array[valid_trials] = np.asarray([np.nanmean(values[start:stop]) for start, stop in
                                            zip(epoch_idx_start, epoch_idx_stop)], dtype=float)

    return epoch_array