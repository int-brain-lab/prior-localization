import numpy as np
import pandas as pd
import pytest

from braindelphi.decoding.functions.process_targets import get_target_data_per_trial
from braindelphi.decoding.functions.process_targets import get_target_data_per_trial_wrapper


# TODO: write tests for:
# - optimal_Bayesian
# - compute_beh_target


def test_get_target_data_per_trial():

    offset = 1000
    int_len = 2
    n_trials = 10

    target_times = np.arange(0, 100, 0.1)
    target_vals = offset + target_times
    interval_begs = np.arange(0, 100, n_trials)
    interval_ends = int_len + interval_begs
    binsize = 0.2

    target_times_list, target_data_list, mask = get_target_data_per_trial(
        target_times, target_vals, interval_begs, interval_ends, binsize)

    # new interpolated vals should be directly related to times
    for t_times, t_data in zip(target_times_list, target_data_list):
        assert t_times.shape == t_data.shape
        assert len(t_times) == (int_len / binsize + 1)
        assert np.allclose(t_times + offset, t_data)

    # test when nans in target data
    target_vals[:5] = np.nan
    target_times_list, target_data_list, mask = get_target_data_per_trial(
        target_times, target_vals, interval_begs, interval_ends, binsize, allow_nans=True)
    assert len(target_times_list) == n_trials
    assert mask[0]
    target_times_list, target_data_list, mask = get_target_data_per_trial(
        target_times, target_vals, interval_begs, interval_ends, binsize, allow_nans=False)
    assert len(target_times_list) == (n_trials - 1)
    assert ~mask[0]

    # test when target times fall outside range of requested intervals
    interval_begs = np.arange(0, 200, n_trials)
    interval_ends = int_len + interval_begs
    target_times_list, target_data_list, mask = get_target_data_per_trial(
        target_times, target_vals, interval_begs, interval_ends, binsize, allow_nans=True)
    assert ~mask[-1]

    # test when target times start too late
    target_times = np.arange(10, 100, 0.1)
    interval_begs = np.arange(0, 100, n_trials)
    interval_ends = int_len + interval_begs
    target_times_list, target_data_list, mask = get_target_data_per_trial(
        target_times, target_vals, interval_begs, interval_ends, binsize, allow_nans=True)
    assert ~mask[0]

    # test when target times stop too early
    target_times = np.arange(0, 80, 0.1)
    interval_begs = np.arange(0, 100, n_trials)
    interval_ends = int_len + interval_begs
    target_times_list, target_data_list, mask = get_target_data_per_trial(
        target_times, target_vals, interval_begs, interval_ends, binsize, allow_nans=True)
    assert ~mask[-1]

    # TODO: test when target_vals is a matrix instead of array


def test_get_target_data_per_trial_wrapper():

    # just make sure this does the formatting of intervals properly; most of the heavy lifting is
    # done by `test_get_target_data_per_trial`

    offset = 1000
    n_trials = 10

    target_times = np.arange(0, 100, 0.1)
    target_vals = offset + target_times
    align_times = np.arange(0, 100, n_trials)
    align_event = 'align_str'
    align_interval = (0, 1)
    binsize = 0.2

    # function runs normally, returns output of proper length
    trials_df = pd.DataFrame({align_event: align_times})
    target_times_list, target_data_list, mask = get_target_data_per_trial_wrapper(
        target_times, target_vals, trials_df, align_event, align_interval, binsize)
    assert len(target_times_list) == n_trials

    # key error when alignment string does not exist in dataframe
    trials_df = pd.DataFrame({'bad_align_str': align_times})
    with pytest.raises(KeyError):
        get_target_data_per_trial_wrapper(
            target_times, target_vals, trials_df, align_event, align_interval, binsize)
