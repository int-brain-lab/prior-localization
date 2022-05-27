import numpy as np
import pandas as pd
import pytest

from braindelphi.decoding.functions.process_inputs import build_predictor_matrix
from braindelphi.decoding.functions.process_inputs import get_spike_data_per_trial
from braindelphi.decoding.functions.process_inputs import preprocess_ephys
from braindelphi.decoding.functions.process_inputs import select_ephys_regions


# TODO: write tests for:
# - preprocess_widefiled_imaging
# - select_widefield_imaging_regions


def test_preprocess_ephys():

    n_trials = 10
    spikes_per_trial = 1000
    n_spikes = spikes_per_trial * n_trials
    align_event = 'stimOn_times'
    time_window = (-1.0, 1.0)

    reg_clu_ids = np.array([0, 2, 3])
    neural_df = {
        'spk_clu': np.random.randint(0, 6, (n_spikes,)),
        'spk_times': np.arange(0, n_spikes) / spikes_per_trial,
    }
    trials_df = pd.DataFrame({
        align_event: np.arange(n_trials),
    })

    # process spikes for single-bin decoding
    binned_list = preprocess_ephys(
        reg_clu_ids=reg_clu_ids, neural_df=neural_df, trials_df=trials_df,
        align_time=align_event, time_window=time_window, binsize=2.0)
    assert len(binned_list) == n_trials
    for tr in binned_list:
        assert tr.shape == (1, len(reg_clu_ids))

    # process spikes for multi-bin decoding
    binsize = 0.2
    n_bins_lag = 0
    n_bins_per_trial = int((time_window[1] - time_window[0]) / binsize) + 1
    binned_list = preprocess_ephys(
        reg_clu_ids=reg_clu_ids, neural_df=neural_df, trials_df=trials_df,
        align_time=align_event, time_window=time_window, binsize=binsize, n_bins_lag=n_bins_lag)
    print(binned_list[0].shape)
    print(binned_list[0])
    assert len(binned_list) == n_trials
    for tr in binned_list:
        assert tr.shape == (n_bins_per_trial, len(reg_clu_ids))


def test_get_spike_data_per_trial():

    n_trials = 10
    spikes_per_trial = 1000
    n_spikes = spikes_per_trial * n_trials
    n_clusters = 6
    trial_len = 0.5
    binsize = 0.1

    times = np.arange(0, n_spikes) / spikes_per_trial
    clusters = np.random.randint(0, n_clusters, (n_spikes,))
    interval_begs = np.arange(n_trials)
    interval_ends = interval_begs + trial_len

    n_bins_per_trial = int(trial_len / binsize) + 1
    times_list, spikes_list = get_spike_data_per_trial(
        times=times, clusters=clusters, interval_begs=interval_begs, interval_ends=interval_ends,
        binsize=binsize)
    assert len(times_list) == len(spikes_list)
    for tr in spikes_list:
        assert tr.shape == (n_bins_per_trial, n_clusters)


def test_build_predictor_matrix():

    n_t = 100
    n_clusters = 7
    array = np.random.randn(n_t, n_clusters)

    # no lags
    n_lags = 0
    mat = build_predictor_matrix(array, n_lags, return_valid=True)
    assert np.allclose(mat, array)

    # invalid lags
    n_lags = -1
    with pytest.raises(ValueError):
        build_predictor_matrix(array, n_lags, return_valid=True)

    # positive lags, with and without valid returns
    for n_lags in [1, 2, 3]:
        mat = build_predictor_matrix(array, n_lags, return_valid=False)
        assert mat.shape == (n_t, n_clusters * (n_lags + 1))
        assert np.allclose(array, mat[:, :n_clusters])
        mat = build_predictor_matrix(array, n_lags, return_valid=True)
        assert mat.shape == (n_t - n_lags, n_clusters * (n_lags + 1))
        assert np.allclose(array[n_lags:], mat[:, :n_clusters])


def test_select_ephys_regions():

    # let all clusters pass
    regressors = {'clu_qc': {'label': np.array([0, 0, 0.3, 0.3, 0.3, 0.6, 0.6, 1, 1, 1, 1])}}
    n_ids = len(regressors['clu_qc']['label'])
    beryl_reg = np.array(['R1'] * n_ids)
    reg_clu_ids = select_ephys_regions(
        regressors=regressors, beryl_reg=beryl_reg, region='R1', qc_criteria=0)
    assert len(reg_clu_ids) == n_ids

    # subselect based on qc
    reg_clu_ids = select_ephys_regions(
        regressors=regressors, beryl_reg=beryl_reg, region='R1', qc_criteria=0.3)
    assert len(reg_clu_ids) == n_ids - 2

    reg_clu_ids = select_ephys_regions(
        regressors=regressors, beryl_reg=beryl_reg, region='R1', qc_criteria=0.6)
    assert len(reg_clu_ids) == n_ids - 2 - 3

    reg_clu_ids = select_ephys_regions(
        regressors=regressors, beryl_reg=beryl_reg, region='R1', qc_criteria=1.0)
    assert len(reg_clu_ids) == n_ids - 2 - 3 - 2

    # subselect based on region
    n_r2 = 2
    for i in range(n_r2):
        beryl_reg[i] = 'R2'
    reg_clu_ids = select_ephys_regions(
        regressors=regressors, beryl_reg=beryl_reg, region='R1', qc_criteria=0)
    assert len(reg_clu_ids) == (n_ids - n_r2)

    # subselect based on region AND qc
    beryl_reg = np.array(['R1'] * n_ids)
    beryl_reg[-1] = 'R2'
    beryl_reg[-2] = 'R2'
    reg_clu_ids = select_ephys_regions(
        regressors=regressors, beryl_reg=beryl_reg, region='R1', qc_criteria=1.0)
    assert len(reg_clu_ids) == 2
