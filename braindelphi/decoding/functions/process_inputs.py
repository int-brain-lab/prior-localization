import numpy as np
import scipy
from brainbox.population.decode import get_spike_counts_in_bins


def preprocess_ephys(reg_clu_ids, regressors, trials_df, **kwargs):
    intervals = np.vstack([
        trials_df[kwargs['align_time']] + kwargs['time_window'][0],
        trials_df[kwargs['align_time']] + kwargs['time_window'][1]
    ]).T
    spikemask = np.isin(regressors['spk_clu'], reg_clu_ids)
    regspikes = regressors['spk_times'][spikemask]
    regclu = regressors['spk_clu'][spikemask]
    binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
    return binned.T


def get_spike_data_per_trial(times, clusters, interval_begs, interval_ends, binsize):
    """Select spiking data for specified interval on each trial.

    Parameters
    ----------
    times : array-like
        time in seconds for each spike
    clusters : array-like
        cluster id for each spike
    interval_begs : array-like
        beginning of each interval in seconds
    interval_ends : array-like
        end of each interval in seconds
    binsize : float
        width of each bin in seconds

    Returns
    -------
    tuple
        - (list): time in seconds for each trial
        - (list): data for each trial of shape (n_clusters, n_bins)

    """

    n_trials = len(interval_begs)
    n_bins = int((interval_ends[0] - interval_begs[0]) / binsize) + 1
    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    binned_spikes = np.zeros((n_trials, n_clusters_in_region, n_bins))
    spike_times_list = []
    for tr, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
        # just get spikes for this region/trial
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            binned_spikes_tmp = np.zeros((n_clusters_in_region, n_bins))
            t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            binned_spikes_tmp, t_idxs, cluster_idxs = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)

        # update data block
        binned_spikes[tr, idxs_tmp, :] += binned_spikes_tmp[:, :n_bins]
        spike_times_list.append(t_idxs[:n_bins])

    return spike_times_list, binned_spikes


def build_predictor_matrix(array, n_lags):
    """Build predictor matrix with time-lagged datapoints.

    Parameters
    ----------
    array : np.ndarray
        shape (n_time, n_clusters)
    n_lags : int
        number of lagged timepoints (includes zero lag)

    Returns
    -------
    np.ndarray
        shape (n_time, n_clusters * (n_lags + 1))

    """
    return np.hstack([np.roll(array, i, axis=0) for i in range(n_lags + 1)])[n_lags:]


def proprocess_widefield_imaging():
    frames_idx = wideFieldImaging_dict['timings'][kwargs['align_time']].values
    frames_idx = np.sort(
        frames_idx[:, None] +
        np.arange(0, kwargs['wfi_nb_frames'], np.sign(kwargs['wfi_nb_frames'])),
        axis=1,
    )
    binned = np.take(wideFieldImaging_dict['activity'][:, reg_mask],
                     frames_idx,
                     axis=0)
    binned = binned.reshape(binned.shape[0], -1).T
    return binned

def select_widefield_imaging_regions():
    region_labels = []
    reg_lab = wideFieldImaging_dict['atlas'][wideFieldImaging_dict['atlas'].acronym ==
                                             region].label.values.squeeze()
    if 'left' in kwargs['wfi_hemispheres']:
        region_labels.append(reg_lab)
    if 'right' in kwargs['wfi_hemispheres']:
        region_labels.append(-reg_lab)

    reg_mask = np.isin(wideFieldImaging_dict['clu_regions'], region_labels)
    reg_clu_ids = np.argwhere(reg_mask)
    return reg_clu_ids

def select_ephys_regions(regressors, beryl_reg, region, **kwargs):
    qc_pass = (regressors['clu_qc']['label'] >= kwargs['qc_criteria'])
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
    return reg_clu_ids