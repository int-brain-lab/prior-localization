
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import scipy
from sklearn import linear_model as sklm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score

from brainbox.behavior.wheel import interpolate_position, velocity_smoothed
import brainbox.io.one as bbone
from brainbox.processing import bincount2D
from ibllib.atlas import BrainRegions
from one.api import ONE

"""
Notes:
    - need to clean up imports from brainwide/decoding/functions/utils.py so that I don't have to
      install behavior models or pytorch
    -

"""
# -------------------------------------------------------------------------------------------------
# copied from brainwide/decoding/functions/utils.py
# -------------------------------------------------------------------------------------------------


def query_sessions(selection='all', one=None):
    '''
    Filters sessions on some canonical filters
    returns dataframe with index being EID, so indexing results in subject name and probe
    identities in that EID.
    '''
    one = one or ONE()
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

    #  Get list of eids and probes
    all_eids = np.array([i['session'] for i in ins])
    all_probes = np.array([i['name'] for i in ins])
    all_subjects = np.array([i['session_info']['subject'] for i in ins])
    all_pids = np.array([i['id'] for i in ins])
    retdf = pd.DataFrame({'subject': all_subjects, 'eid': all_eids, 'probe': all_probes, 'pid': all_pids})
    retdf.sort_values('subject', inplace=True)
    return retdf


def remap_region(ids, source='Allen-lr', dest='Beryl-lr', output='acronym', br=None):
    br = br or BrainRegions()
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


def return_regions(eid, sessdf, QC_CRITERIA=1, NUM_UNITS=10):
    df_insertions = sessdf.loc[sessdf['eid'] == eid]
    brainreg = BrainRegions()
    my_regions = {}
    for i, ins in tqdm(df_insertions.iterrows(), desc='Probe: ', leave=False):
        probe = ins['probe']
        spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
        clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
        beryl_reg = remap_region(clusters.atlas_id, br=brainreg)
        qc_pass = (clusters['label'] >= QC_CRITERIA).values
        regions = np.unique(beryl_reg)
        # warnings.filterwarnings('ignore')
        probe_regions = []
        for region in tqdm(regions, desc='Region: ', leave=False):
            reg_mask = (beryl_reg == region)
            reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
            if len(reg_clu_ids) > NUM_UNITS:
                probe_regions.append(region)
        my_regions[probe] = probe_regions
    return my_regions


# -------------------------------------------------------------------------------------------------
# data IO/manipulation
# -------------------------------------------------------------------------------------------------


def load_wheel_data(one, eid, sampling_rate=1000):
    # wheel_pos_og = one.load(eid, dataset_types='wheel.position')[0]
    # wheel_times_og = one.load(eid, dataset_types='wheel.timestamps')[0]
    wheel_pos_og = one.load_dataset(eid, '_ibl_wheel.position.npy')
    wheel_times_og = one.load_dataset(eid, '_ibl_wheel.timestamps.npy')
    if wheel_times_og.shape[0] != wheel_pos_og.shape[0]:
        raise TimestampError
    # resample the wheel position and compute velocity, acceleration
    wheel_pos, wheel_times = interpolate_position(wheel_times_og, wheel_pos_og, freq=sampling_rate)
    wheel_vel, wheel_acc = velocity_smoothed(wheel_pos, sampling_rate)
    return wheel_times, wheel_pos, wheel_vel, wheel_acc


def load_pupil_data(one, eid, view='left', algo='dlc', version=2, snr_thresh=5):

    assert view == 'left', 'Pupil diameter computation only implemented for left pupil'

    # alf_path = os.path.join(str(one.path_from_eid(eid)), 'alf')
    #
    # pupil_file = os.path.join(alf_path, '_ibl_pupil.%s%i.npy' % (algo, version))
    # pupil_diam = np.load(pupil_file)
    #
    # timestamps_file = os.path.join(alf_path, '_ibl_%sCamera.times.npy' % view)
    # times = np.load(timestamps_file)
    #
    # assert times.shape[0] == pupil_diam.shape[0]

    times, xys = get_dlc_traces(one, eid, view='left')
    if xys is None:
        return None, None

    diam0 = get_pupil_diameter(xys, view='left')
    if times.shape[0] != diam0.shape[0]:
        raise TimestampError
    pupil_diam = smooth_interpolate_signal(diam0, window=31, order=3, interp_kind='linear')
    good_idxs = np.where(~np.isnan(diam0) & ~np.isnan(pupil_diam))[0]
    snr = np.var(pupil_diam[good_idxs]) / np.var(diam0[good_idxs] - pupil_diam[good_idxs])
    if snr < snr_thresh:
        raise QCError('Bad SNR value (%1.2f)' % snr)

    return times, pupil_diam


def load_paw_data(
        one, eid, paw, kind, smooth=True, frac_thresh=0.9, jump_thresh_px=200, n_jump_thresh=50):

    if paw == 'l':
        view = 'left'
    else:
        view = 'right'

    times, paw_markers_all, frac_present, jumps = load_paw_markers(
        one, eid, view=view, smooth=smooth, jump_thresh_px=jump_thresh_px)
    if times is None or paw_markers_all is None:
        return None, None

    # note: the *right* paw returned by the dlc dataframe is always the paw closest to the camera:
    # the paws are labeled according to the viewer's sense of L/R
    # the left view is the base view; the right view is flipped to match the left view
    # the right view (vid and markers) is flipped back, but the DLC bodypart names stay the same

    # perform QC checks
    if frac_present['r_x'] < frac_thresh:
        raise QCError(
            'fraction of present markers (%1.2f) is below threshold (%1.2f)' %
            (frac_present['r_x'], frac_thresh))
    if jumps['r'] > n_jump_thresh:
        raise QCError(
            'number of large jumps (%i) is above threshold (%i)' % (jumps['r'], n_jump_thresh))

    # process; compute position/velocity/speed
    paw_markers = np.hstack(
        [paw_markers_all['paw_r_x'][:, None], paw_markers_all['paw_r_y'][:, None]])

    if kind == 'pos':
        # x-y coordinates; downstream functions can handle this
        paw_data = paw_markers
    elif kind == 'vel':
        # x-y coordinates; downstream functions can handle this
        paw_data = np.concatenate([[[0, 0]], np.diff(paw_markers, axis=0)])
    elif kind == 'speed':
        diffs = np.sqrt(np.sum(np.square(np.diff(paw_markers, axis=0)), axis=1))
        paw_data = np.concatenate([[0], diffs])
    else:
        raise NotImplementedError(
            '%s is not a valid kind of paw data; must be "pos", "vel" or "speed"' % kind)

    return times, paw_data


def load_paw_markers(one, eid, view='left', smooth=True, jump_thresh_px=200):

    times, xys = get_dlc_traces(one, eid, view=view)
    if times is None or xys is None:
        return None, None, None, None

    # separate data
    paw_r_x = xys['paw_r'][:, 0]
    paw_r_y = xys['paw_r'][:, 1]
    paw_l_x = xys['paw_l'][:, 0]
    paw_l_y = xys['paw_l'][:, 1]

    if times.shape[0] != paw_l_y.shape[0]:
        raise TimestampError

    # compute fraction of time points present
    frac_r_x = np.sum(~np.isnan(paw_r_x)) / paw_r_x.shape[0]
    frac_r_y = np.sum(~np.isnan(paw_r_y)) / paw_r_y.shape[0]
    frac_l_x = np.sum(~np.isnan(paw_l_x)) / paw_l_x.shape[0]
    frac_l_y = np.sum(~np.isnan(paw_l_y)) / paw_l_y.shape[0]

    # compute number of large jumps
    diffs_r = np.concatenate([[[0, 0]], np.diff(xys['paw_r'], axis=0)])
    jumps_r = np.sqrt(np.sum(np.square(diffs_r), axis=1))
    n_jumps_r = np.sum(jumps_r > jump_thresh_px)
    diffs_l = np.concatenate([[[0, 0]], np.diff(xys['paw_l'], axis=0)])
    jumps_l = np.sqrt(np.sum(np.square(diffs_l), axis=1))
    n_jumps_l = np.sum(jumps_l > jump_thresh_px)

    if smooth:
        if view == 'left':
            ws = 7
        else:
            ws = 17
        paw_r_x = smooth_interpolate_signal(paw_r_x, window=ws)
        paw_r_y = smooth_interpolate_signal(paw_r_y, window=ws)
        paw_l_x = smooth_interpolate_signal(paw_l_x, window=ws)
        paw_l_y = smooth_interpolate_signal(paw_l_y, window=ws)
    else:
        raise NotImplementedError("Need to implement interpolation w/o smoothing")

    markers = {'paw_l_x': paw_l_x, 'paw_r_x': paw_r_x, 'paw_l_y': paw_l_y, 'paw_r_y': paw_r_y}
    fracs = {'l_x': frac_l_x, 'r_x': frac_r_x, 'l_y': frac_l_y, 'r_y': frac_r_y}
    jumps = {'l': n_jumps_l, 'r': n_jumps_r}

    return times, markers, fracs, jumps


def load_whisker_data(one, eid, view):
    if view == 'l':
        view = 'left'
    elif view == 'r':
        view = 'right'
    else:
        raise NotImplementedError

    try:
        times = one.load_dataset(eid, '_ibl_%sCamera.times.npy' % view)
        me = one.load_dataset(eid, '%sCamera.ROIMotionEnergy.npy' % view)
    except:
        msg = 'whisker ME data not available'
        logging.exception(msg)
        return None, None

    if times.shape[0] != me.shape[0]:
        raise TimestampError

    return times, me


def get_dlc_traces(one, eid, view, likelihood_thresh=0.9):
    try:
        times = one.load_dataset(eid, '_ibl_%sCamera.times.npy' % view)
        cam = one.load_dataset(eid, '_ibl_%sCamera.dlc.pqt' % view)
    except:
        msg = 'not all dlc data available'
        logging.exception(msg)
        return None, None
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array([x, y]).T
    return times, XYs


def remove_bad_trials(
        trialsdf, align_event, align_interval, min_len=1, max_len=5, no_unbias=False, min_rt=0.08):
    """Filter trials."""

    # define reaction times
    trialsdf['react_times'] = trialsdf['firstMovement_times'] - trialsdf['goCue_times']

    # successively build a mask that defines which trials we want to keep

    # ensure align event is not a nan
    mask = trialsdf[align_event].notna()

    # ensure animal has moved
    mask = mask & trialsdf['firstMovement_times'].notna()

    # get rid of unbiased trials
    if no_unbias:
        mask = mask & (trialsdf.probabilityLeft != 0.5).values

    # keep trials with reasonable reaction times
    if min_rt is not None:
        mask = mask & (trialsdf.react_times > min_rt).values

    # get rid of trials that are too short or too long
    start_diffs = trialsdf.trial_start.diff()
    start_diffs.iloc[0] = 2
    mask = mask & ((start_diffs > min_len).values & (start_diffs < max_len).values)

    # get rid of trials with decoding windows that overlap following trial
    tmp = (trialsdf[align_event].values[:-1] + align_interval[1]) < trialsdf.trial_start.values[1:]
    tmp = np.concatenate([tmp, [True]])  # include final trial, no following trials
    mask = mask & tmp

    # only use successful trials if aligning to feedback
    if align_event == 'feedback':
        mask = mask & (trialsdf.feedbackType == 1).values

    trialsdf_masked = trialsdf[mask]

    return trialsdf_masked


def get_target_data_per_trial(
        target_times, target_data, interval_begs, interval_ends, binsize, allow_nans=False):
    """Select wheel data for specified interval on each trial.

    Parameters
    ----------
    target_times : array-like
        time in seconds for each sample
    target_data : array-like
        data samples
    interval_begs : array-like
        beginning of each interval in seconds
    interval_ends : array-like
        end of each interval in seconds
    binsize : float
        width of each bin in seconds
    allow_nans : bool, optional
        False to skip trials with >0 NaN values in target data

    Returns
    -------
    tuple
        - (list): time in seconds for each trial
        - (list): data for each trial

    """

    n_bins = int((interval_ends[0] - interval_begs[0]) / binsize) + 1
    idxs_beg = np.searchsorted(target_times, interval_begs, side='right')
    idxs_end = np.searchsorted(target_times, interval_ends, side='left')
    target_times_og_list = [target_times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_data_og_list = [target_data[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    # interpolate and store
    target_times_list = []
    target_data_list = []
    good_trial = [None for _ in range(len(target_times_og_list))]
    for i, (target_time, target_vals) in enumerate(zip(target_times_og_list, target_data_og_list)):
        if len(target_vals) == 0:
            print('target data not present on trial %i; skipping' % i)
            good_trial[i] = False
            continue
        if np.sum(np.isnan(target_vals)) > 0 and not allow_nans:
            print('nans in target data on trial %i; skipping' % i)
            good_trial[i] = False
            continue
        if np.abs(interval_begs[i] - target_time[0]) > binsize:
            print('target data starts too late on trial %i; skipping' % i)
            good_trial[i] = False
            continue
        if np.abs(interval_ends[i] - target_time[-1]) > binsize:
            print('target data ends too early on trial %i; skipping' % i)
            good_trial[i] = False
            continue
        # x_interp = np.arange(target_time[0], target_time[-1] + binsize / 2, binsize)
        x_interp = np.linspace(target_time[0], target_time[-1], n_bins)
        if len(target_vals.shape) > 1 and target_vals.shape[1] > 1:
            n_dims = target_vals.shape[1]
            y_interp_tmps = []
            for n in range(n_dims):
                y_interp_tmps.append(scipy.interpolate.interp1d(
                    target_time, target_vals[:, n], kind='linear',
                    fill_value='extrapolate')(x_interp))
            y_interp = np.hstack([y[:, None] for y in y_interp_tmps])
        else:
            y_interp = scipy.interpolate.interp1d(
                target_time, target_vals, kind='linear', fill_value='extrapolate')(x_interp)
        target_times_list.append(x_interp)
        target_data_list.append(y_interp)
        good_trial[i] = True

    return target_times_list, target_data_list, np.array(good_trial)


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


def smooth_interpolate_signal(signal, window=31, order=3, interp_kind='linear'):

    from scipy.interpolate import interp1d

    signal_noisy_w_nans = np.copy(signal)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    good_idxs = np.where(~np.isnan(signal_noisy_w_nans))[0]
    if len(good_idxs) <= window:
        # can't interpolate; return original signal
        return signal_noisy_w_nans

    # perform savitzky-golay filtering on non-nan points
    signal_smooth_nonans = non_uniform_savgol(
        timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
    signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
    signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
    # interpolate nan points
    interpolater = interp1d(
        timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')

    signal = interpolater(timestamps)

    return signal


def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    https://dsp.stackexchange.com/a/64313

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """

    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


# -------------------------------------------------------------------------------------------------
# pipeline-level data IO/manipulation with error handling
# -------------------------------------------------------------------------------------------------

def load_target_data(one, eid, target):
    """High-level function to load target data, with error handling.

    Parameters
    ----------
    one
    eid
    target

    Returns
    -------

    """

    target_times = None
    target_vals = None
    skip_session = False

    if target == 'wheel-vel' or target == 'wheel-speed':
        try:
            target_times, _, target_vals, _ = load_wheel_data(one, eid)
            if target == 'wheel-speed':
                target_vals = np.abs(target_vals)
        except TimestampError:
            msg = '%s timestamps/data mismatch' % target
            logging.exception(msg)
            skip_session = True
    elif target == 'pupil':
        try:
            target_times, target_vals = load_pupil_data(one, eid, snr_thresh=5)
        except TimestampError:
            msg = '%s timestamps/data mismatch' % target
            logging.exception(msg)
            skip_session = True
        except QCError:
            msg = 'pupil trace did not pass QC'
            logging.exception(msg)
            skip_session = True
        except ValueError:
            msg = 'not enough good pupil points'
            logging.exception(msg)
            skip_session = True
    elif target.find('paw') > -1:  # '[l/r]-paw-[pos/vel/speed]'
        try:
            target_times, target_vals = load_paw_data(
                one, eid, paw=target.split('-')[0], kind=target.split('-')[2])
        except TimestampError:
            msg = '%s timestamps/data mismatch' % target
            logging.exception(msg)
            skip_session = True
        except QCError:
            msg = 'paw trace did not pass QC tests'
            logging.exception(msg)
            skip_session = True
    elif target == 'l-whisker-me' or target == 'r-whisker-me':
        try:
            target_times, target_vals = load_whisker_data(
                one, eid, view=target.split('-')[0])
        except TimestampError:
            msg = '%s timestamps/data mismatch' % target
            logging.exception(msg)
            skip_session = True
    else:
        raise NotImplementedError('%s is an invalid decoding target' % target)

    if not skip_session:
        if target_times is None:
            msg = 'no %s times' % target
            logging.info(msg)
            skip_session = True
        if target_vals is None:
            msg = 'no %s vals' % target
            logging.info(msg)
            skip_session = True

    return target_times, target_vals, skip_session


def load_interval_data(one, eid, align_event, align_interval, no_unbias=False, min_rt=0.08):
    """High-level function to load interval data, with error handling.

    Parameters
    ----------
    one
    eid
    align_event
    align_interval
    no_unbias
    min_rt

    Returns
    -------

    """

    trialsdf = bbone.load_trials_df(eid, one=one, addtl_types=['firstMovement_times'])

    # remove bad trials
    # - reaction times below a threshold
    # - no movement onset time detected
    # - trial too short
    # - trial too long
    trialsdf_masked = remove_bad_trials(
        trialsdf, align_event=align_event, align_interval=align_interval, no_unbias=no_unbias,
        min_rt=min_rt)

    skip_session = False
    if trialsdf_masked.shape[0] == 0:
        msg = 'no %s aligment times found' % align_event
        logging.exception(msg)
        skip_session = True

    return trialsdf_masked, skip_session


def get_target_data_per_trial_error_check(
        target_times, target_vals, trialsdf, align_event, align_interval, binsize, min_trials):
    """High-level function to split target data over trials, with error checking.

    Parameters
    ----------
    target_times
    target_vals
    trialsdf
    align_event
    align_interval
    binsize
    min_trials

    Returns
    -------

    """

    align_times = trialsdf[align_event].values
    interval_beg_times = align_times + align_interval[0]
    interval_end_times = align_times + align_interval[1]

    # split data by trial
    target_times_list, target_val_list, good_trials = get_target_data_per_trial(
        target_times, target_vals, interval_beg_times, interval_end_times, binsize)

    bad_trials = np.sum(~good_trials)
    if bad_trials > 0:
        msg = 'no data for %i trials' % bad_trials
        logging.exception(msg)
        trialsdf = trialsdf[good_trials]

    skip_session = False
    n_trials = trialsdf.shape[0]
    if n_trials < min_trials:
        msg = 'session contains %i trials, below the threshold of %i' % (n_trials, min_trials)
        logging.exception(msg)
        skip_session = True

    return target_times_list, target_val_list, trialsdf, skip_session


# -------------------------------------------------------------------------------------------------
# decoding
# -------------------------------------------------------------------------------------------------

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


def get_save_path(
        pseudo_id, subject, eid, probe, region, output_path, time_window, today, target,
        add_to_saving_path):
    subjectfolder = Path(output_path).joinpath(subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    start_tw, end_tw = time_window
    fn = '_'.join([today, region, 'target', target,
                   'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_'),
                   'pseudo_id', str(pseudo_id), add_to_saving_path]) + '.pkl'
    save_path = probefolder.joinpath(fn)
    return save_path


def save_region_results(fit_result, pseudo_id, subject, eid, probe, region, N, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    outdict = {
        'fit': fit_result, 'pseudo_id': pseudo_id, 'subject': subject, 'eid': eid, 'probe': probe,
        'region': region, 'N_units': N
    }
    fw = open(save_path, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return save_path


def decode(
        Xs, ys, decoder=sklm.Ridge, test_prop=0.2, nFolds=5, save_predictions=True, shuffle=True,
        hyperparameter_grid=None, outer_cv=True, estimator_kwargs={}, rng_seed=0):
    """

    Parameters
    ----------
    Xs : list
        predictors
    ys : list
        targets
    decoder : sklearn.linear_model object
        Estimator from sklearn which provides .fit, .score, and .predict methods. CV estimators
        are NOT SUPPORTED. Must be a normal estimator, which is internally wrapped with
        GridSearchCV
    test_prop : float
        Proportion of data to hold out as the test set after running hyperparameter tuning.
        Default 0.2
    nFolds : int
        Number of folds for cross-validation during hyperparameter tuning.
    save_predictions : bool
        True to return predictions in results dict; False to leave blank. Note: this function does
        not actually save any files!
    shuffle : bool
        True for interleaved cross-validation, False for contiguous blocks
    hyperparameter_grid : dict
        regularization parameters
    outer_cv : bool
        Perform outer cross validation such that the testing spans the entire dataset
    estimator_kwargs : dict
        additional kwargs for sklearn estimator
    rng_seed : int
        control data splits

    Returns
    -------
    dict
        Dictionary of fitting outputs including:
            - Regression score (from estimator)
            - Decoding coefficients
            - Decoding intercept
            - Per-trial target values
            - Per-trial predictions from model

    """

    n_trials = len(Xs)

    # initialize outputs
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]

    # train / test split
    # Split the dataset in two parts, train and test
    # when shuffle=False, the method will take the end of the dataset to create the test set
    np.random.seed(rng_seed)
    indices = np.arange(n_trials)
    if outer_cv:
        outer_kfold = KFold(n_splits=nFolds, shuffle=shuffle).split(indices)
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])

    # scoring function
    scoring_f = r2_score

    # Select either the GridSearchCV estimator for a normal estimator, or use the native estimator
    # in the case of CV-type estimators
    if isinstance(decoder, sklm._coordinate_descent.LinearModelCV):
        raise NotImplemented('the code does not support a CV-type estimator for the moment.')
    else:
        for train_idxs, test_idxs in outer_kfold:

            X_train = np.vstack([Xs[i] for i in train_idxs])
            y_train = np.concatenate([ys[i] for i in train_idxs], axis=0)
            X_test = np.vstack([Xs[i] for i in test_idxs])
            y_test = np.concatenate([ys[i] for i in test_idxs], axis=0)

            idx_inner = np.arange(len(X_train))
            inner_kfold = KFold(n_splits=nFolds, shuffle=shuffle).split(idx_inner)

            key = list(hyperparameter_grid.keys())[0]

            # find best regularization hyperparameter over inner folds
            r2s = np.zeros([nFolds, len(hyperparameter_grid[key])])
            for ifold, (train_inner, test_inner) in enumerate(inner_kfold):
                X_train_inner, X_test_inner = X_train[train_inner], X_train[test_inner]
                y_train_inner, y_test_inner = y_train[train_inner], y_train[test_inner]
                for i_alpha, alpha in enumerate(hyperparameter_grid[key]):
                    estimator = decoder(**{**estimator_kwargs, key: alpha})
                    estimator.fit(X_train_inner, y_train_inner)
                    pred_test_inner = estimator.predict(X_test_inner)
                    r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)
            r2s_avg = r2s.mean(axis=0)
            best_alpha = hyperparameter_grid[key][np.argmax(r2s_avg)]

            # use best hparam to fit model on all data
            clf = decoder(**{**estimator_kwargs, key: best_alpha})
            clf.fit(X_train, y_train)

            # compute R2 on the train data
            y_pred_train = clf.predict(X_train)
            scores_train.append(scoring_f(y_train, y_pred_train))
            idxes_train.append(train_idxs)

            # compute R2 on held-out data
            y_pred_test = clf.predict(X_test)
            scores_test.append(scoring_f(y_test, y_pred_test))
            idxes_test.append(test_idxs)

            # save the raw predictions on test data
            y_pred_fold = [clf.predict(Xs[i]) for i in test_idxs]
            for i, i_test in enumerate(test_idxs):
                predictions[i_test] = y_pred_fold[i]

            # save out model parameters
            weights.append(clf.coef_)
            if clf.fit_intercept:
                intercepts.append(clf.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: best_alpha})

    outdict = dict()
    outdict['scores_test_full'] = scoring_f(np.concatenate(ys), np.concatenate(predictions))
    outdict['scores_train'] = scores_train
    outdict['scores_test'] = scores_test
    outdict['weights'] = weights
    outdict['intercepts'] = intercepts
    outdict['target'] = ys if save_predictions else None
    outdict['predictions'] = np.array(predictions) if save_predictions else None
    outdict['idxes_test'] = idxes_test
    outdict['idxes_train'] = idxes_train
    outdict['best_params'] = best_params
    outdict['nFolds'] = nFolds

    return outdict


# -------------------------------------------------------------------------------------------------
# misc
# -------------------------------------------------------------------------------------------------

class QCError(Exception):
    pass


class TimestampError(Exception):
    pass
