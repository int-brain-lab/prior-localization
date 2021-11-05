"""
Utility functions for the prior-localization repository
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import brainbox.io.one as bbone
from brainbox.singlecell import calculate_peths
from brainbox.plot import peri_event_time_histogram
import models.utils as mut
from iblutil.numerical import ismember
from ibllib.atlas import BrainRegions
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prevAct
from sklearn.decomposition import PCA

from one.api import ONE

brainregions = BrainRegions()


def get_bwm_ins_alyx(one):
    """
    Return insertions that match criteria :
    - project code
    - session QC not critical (TODO may need to add probe insertion QC)
    - at least 1 alignment
    - behavior pass
    :return:
    ins: dict containing the full details on insertion as per the alyx rest query
    ins_id: list of insertions eids
    sess_id: list of (unique) sessions eids
    """
    ins = one.alyx.rest('insertions', 'list',
                        provenance='Ephys aligned histology track',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               '~json__qc,CRITICAL,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1')
    sessions = {}
    for item in ins:
        s_eid = item['session_info']['id']
        if s_eid not in sessions:
            sessions[s_eid] = []
        sessions[s_eid].append(item['id'])
    return sessions


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
        tmpdf = bbone.load_trials_df(eid, one=one, **tdf_kwargs)
        tmpdf[timing_vars] += t_last
        dfs.append(tmpdf)
        t_last = tmpdf.iloc[-1]['trial_end']
    return pd.concat(dfs).reset_index()


def fit_exp_prev_act(session_id, one=None):
    if not one:
        one = ONE()

    subjects, _, _, sess_ids, _ = mut.get_bwm_ins_alyx(one)

    mouse_name = one.get_details(session_id)['subject']
    stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
    mcounter = 0
    for i in range(len(sess_ids)):
        if subjects[i] == mouse_name:
            data = mut.load_session(sess_ids[i])
            if data['choice'] is not None and data['probabilityLeft'][0] == 0.5:
                stim_side, stimuli, actions, pLeft_oracle = mut.format_data(data)
                stimuli_arr.append(stimuli)
                actions_arr.append(actions)
                stim_sides_arr.append(stim_side)
                session_uuids.append(sess_ids[i])
            if sess_ids[i] == session_id:
                j = mcounter
            mcounter += 1
    # format data
    stimuli, actions, stim_side = mut.format_input(
        stimuli_arr, actions_arr, stim_sides_arr)
    session_uuids = np.array(session_uuids)
    model = exp_prevAct('./results/inference/', session_uuids,
                        mouse_name, actions, stimuli, stim_side)
    model.load_or_train(remove_old=False)
    # compute signals of interest
    signals = model.compute_signal(signal=['prior', 'prediction_error', 'score'],
                                    verbose=False)
    if len(signals['prior'].shape) == 1:
        return signals['prior']
    else:
        return signals['prior'][j, :]


def peth_from_eid_blocks(eid, probe_idx, unit, one=None):
    if not one:
        one = bbone.ONE()
    trialsdf = bbone.load_trials_df(eid, one=one, t_before=0.6, t_after=0.6)
    trialsdf = trialsdf[np.isfinite(trialsdf.stimOn_times)]
    probestr = 'probe0' + str(probe_idx)
    spikes, clusters = bbone.load_spike_sorting(eid, one=one, probe=probestr)
    spkt, spk_clu = spikes[probestr].times, spikes[probestr].clusters
    fig, ax = plt.subplots(2, 1, figsize=(4, 12), gridspec_kw={'height_ratios': [1, 2]})
    highblock_t = trialsdf[trialsdf.probabilityLeft == 0.8].stimOn_times
    lowblock_t = trialsdf[trialsdf.probabilityLeft == 0.2].stimOn_times
    peri_event_time_histogram(spkt, spk_clu, highblock_t, unit, t_before=0.6, t_after=0.6,
                              error_bars='sem', ax=ax[0],
                              pethline_kwargs={'lw': 2, 'color': 'orange',
                                               'label': 'High probability L'},
                              errbar_kwargs={'color': 'orange', 'alpha': 0.5})
    yscale_orig = ax[0].get_ylim()
    yticks_orig = ax[0].get_yticks()[1:]
    peri_event_time_histogram(spkt, spk_clu, lowblock_t, unit, t_before=0.6, t_after=0.6,
                              error_bars='sem', ax=ax[0],
                              pethline_kwargs={'lw': 2, 'color': 'blue',
                                               'label': 'Low probability L'},
                              errbar_kwargs={'color': 'blue', 'alpha': 0.5})
    yscale_new = ax[0].get_ylim()
    ax[0].set_ylim([min(yscale_orig[0], yscale_new[0]), max(yscale_orig[1], yscale_new[1])])
    ax[0].set_yticks(np.append(ax[0].get_yticks(), yticks_orig))
    ax[0].legend()
    _, binned = calculate_peths(spkt, spk_clu, [unit], trialsdf.stimOn_times,
                                pre_time=0.6, post_time=0.6, bin_size=0.02)
    binned = np.squeeze(binned)
    ax[1].imshow(binned, aspect='auto', cmap='gray_r')
    ax[1].fill_betweenx(range(binned.shape[0]),
                        0, binned.shape[1],
                        (trialsdf.probabilityLeft == 0.8).values, label='P(Left) = 0.8',
                        color='orange', alpha=0.05)
    ax[1].fill_betweenx(range(binned.shape[0]),
                        0, binned.shape[1],
                        (trialsdf.probabilityLeft == 0.2).values, label='P(Left) = 0.2',
                        color='blue', alpha=0.05)
    ticks = [0, 30, 60]
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels([-0.6, 0, 0.6])
    ax[1].set_xlim([0, 60])
    return fig, ax


def remap(ids, source='Allen', dest='Beryl', output='acronym'):
    br = brainregions
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == 'id':
        return br.id[br.mappings[dest][inds]]
    elif output == 'acronym':
        return br.get(br.id[br.mappings[dest][inds]])['acronym']


def get_id(acronym):
    return brainregions.id[np.argwhere(brainregions.acronym == acronym)[0, 0]]


def plot_rate_prior(eid, probe, clu_id,
                    one=None, t_before=0., t_after=0.1, binwidth=0.1, smoothing=0, 
                    ax=None):
    if not one:
        one = ONE()
    trialsdf = bbone.load_trials_df(eid, one=one, t_before=t_before, t_after=t_after)
    prior = fit_exp_prev_act(eid, one=one)
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid, one=one, probe=probe)
    _, binned = calculate_peths(spikes[probe].times, spikes[probe].clusters, [clu_id],
                                trialsdf.stimOn_times, pre_time=t_before, post_time=t_after,
                                bin_size=binwidth, smoothing=0.)
    if not ax:
        fig, ax = plt.subplots(1, 1)
    if smoothing > 0:
        filt = norm().pdf(np.linspace(0, 10, smoothing))
        smoothed = np.convolve(binned.flat, filt)[:binned.size]
        smoothed /= smoothed.max()
    else:
        smoothed = binned.flat / binned.max()
    ax.plot(smoothed, label='Unit firing rate')
    ax.plot(prior[trialsdf.index], color='orange', label='Prev act prior est')
    ax.legend()
    return ax


def get_pca_prior(eid, probe, units, one=None, t_start=0., t_end=0.1):
    if not one:
        one = ONE()
    trialsdf = bbone.load_trials_df(eid, one=one, t_before=-t_start, t_after=0.)
    prior = fit_exp_prev_act(eid, one=one)
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid, one=one, probe=probe)
    targmask = np.isin(spikes[probe].clusters, units)
    subset_spikes = spikes[probe].times[targmask]
    subset_clu = spikes[probe].clusters[targmask]
    _, binned = calculate_peths(subset_spikes, subset_clu, units,
                                trialsdf.stimOn_times + t_start if t_start > 0 else trialsdf.stimOn_times,
                                pre_time=-t_start if t_start <0 else 0, post_time=t_end,
                                bin_size=t_end-t_start, smoothing=0., return_fr=False)
    embeddings = PCA().fit_transform(np.squeeze(binned))
    return binned, embeddings, prior


def sessions_with_region(acronym, one=None):
    if one is None:
        one = ONE()
    query_str = (f'channels__brain_region__acronym__icontains,{acronym},'
                 'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                 'probe_insertion__session__qc__lt,50,'
                 '~probe_insertion__json__qc,CRITICAL')
    traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                         django=query_str)
    eids = np.array([i['session']['id'] for i in traj])
    sessinfo = [i['session'] for i in traj]
    probes = np.array([i['probe_name'] for i in traj])
    return eids, sessinfo, probes


def load_trials_df(eid, one=None, maxlen=None, t_before=0., t_after=0., ret_wheel=False,
                   ret_abswheel=False, ext_DLC=False, wheel_binsize=0.02, addtl_types=[]):
    """
    TODO Test this with new ONE
    Generate a pandas dataframe of per-trial timing information about a given session.
    Each row in the frame will correspond to a single trial, with timing values indicating timing
    session-wide (i.e. time in seconds since session start). Can optionally return a resampled
    wheel velocity trace of either the signed or absolute wheel velocity.

    The resulting dataframe will have a new set of columns, trial_start and trial_end, which define
    via t_before and t_after the span of time assigned to a given trial.
    (useful for bb.modeling.glm)

    Parameters
    ----------
    eid : str
        Session UUID string to pass to ONE
    one : oneibl.one.OneAlyx, optional
        one object to use for loading. Will generate internal one if not used, by default None
    maxlen : float, optional
        Maximum trial length for inclusion in df. Trials where feedback - response is longer
        than this value will not be included in the dataframe, by default None
    t_before : float, optional
        Time before stimulus onset to include for a given trial, as defined by the trial_start
        column of the dataframe. If zero, trial_start will be identical to stimOn, by default 0.
    t_after : float, optional
        Time after feedback to include in the trail, as defined by the trial_end
        column of the dataframe. If zero, trial_end will be identical to feedback, by default 0.
    ret_wheel : bool, optional
        Whether to return the time-resampled wheel velocity trace, by default False
    ret_abswheel : bool, optional
        Whether to return the time-resampled absolute wheel velocity trace, by default False
    ext_DLC : bool, optional
        Whether to extract DLC data, by default False
    wheel_binsize : float, optional
        Time bins to resample wheel velocity to, by default 0.02
    addtl_types : list, optional
        List of additional types from an ONE trials object to include in the dataframe. Must be
        valid keys to the dict produced by one.load_object(eid, 'trials'), by default empty.

    Returns
    -------
    pandas.DataFrame
        Dataframe with trial-wise information. Indices are the actual trial order in the original
        data, preserved even if some trials do not meet the maxlen criterion. As a result will not
        have a monotonic index. Has special columns trial_start and trial_end which define start
        and end times via t_before and t_after
    """
    if not one:
        one = ONE()

    if ret_wheel and ret_abswheel:
        raise ValueError('ret_wheel and ret_abswheel cannot both be true.')

    # Define which datatypes we want to pull out
    trialstypes = ['choice',
                   'probabilityLeft',
                   'feedbackType',
                   'feedback_times',
                   'contrastLeft',
                   'contrastRight',
                   'goCue_times',
                   'stimOn_times']
    trialstypes.extend(addtl_types)

    # A quick function to remap probabilities in those sessions where it was not computed correctly
    def remap_trialp(probs):
        # Block probabilities in trial data aren't accurate and need to be remapped
        validvals = np.array([0.2, 0.5, 0.8])
        diffs = np.abs(np.array([x - validvals for x in probs]))
        maps = diffs.argmin(axis=1)
        return validvals[maps]

    trials = one.load_object(eid, 'trials')
    starttimes = trials.stimOn_times
    endtimes = trials.feedback_times
    tmp = {key: value for key, value in trials.items() if key in trialstypes}

    if maxlen is not None:
        with np.errstate(invalid='ignore'):
            keeptrials = (endtimes - starttimes) <= maxlen
    else:
        keeptrials = range(len(starttimes))
    trialdata = {x: tmp[x][keeptrials] for x in trialstypes}
    trialdata['probabilityLeft'] = remap_trialp(trialdata['probabilityLeft'])
    trialsdf = pd.DataFrame(trialdata)
    if maxlen is not None:
        trialsdf.set_index(np.nonzero(keeptrials)[0], inplace=True)
    trialsdf['trial_start'] = trialsdf['stimOn_times'] - t_before
    trialsdf['trial_end'] = trialsdf['feedback_times'] + t_after
    tdiffs = trialsdf['trial_end'] - np.roll(trialsdf['trial_start'], -1)
    if np.any(tdiffs[:-1] > 0):
        logging.warning(f'{sum(tdiffs[:-1] > 0)} trials overlapping due to t_before and t_after '
                        'values. Try reducing one or both!')
    if not ret_wheel and not ret_abswheel:
        return trialsdf

    wheel = one.load_object(eid, 'wheel')
    whlpos, whlt = wheel.position, wheel.timestamps
    starttimes = trialsdf['trial_start']
    endtimes = trialsdf['trial_end']
    wh_endlast = 0
    trials = []
    for (start, end) in np.vstack((starttimes, endtimes)).T:
        wh_startind = np.searchsorted(whlt[wh_endlast:], start) + wh_endlast
        wh_endind = np.searchsorted(whlt[wh_endlast:], end, side='right') + wh_endlast + 4
        wh_endlast = wh_endind
        tr_whlpos = whlpos[wh_startind - 1:wh_endind + 1]
        tr_whlt = whlt[wh_startind - 1:wh_endind + 1] - start
        tr_whlt[0] = 0.  # Manual previous-value interpolation
        whlseries = TimeSeries(tr_whlt, tr_whlpos, columns=['whlpos'])
        whlsync = sync(wheel_binsize, timeseries=whlseries, interp='previous')
        trialstartind = np.searchsorted(whlsync.times, 0)
        trialendind = np.ceil((end - start) / wheel_binsize).astype(int)
        trpos = whlsync.values[trialstartind:trialendind + trialstartind]
        whlvel = trpos[1:] - trpos[:-1]
        whlvel = np.insert(whlvel, 0, 0)
        if np.abs((trialendind - len(whlvel))) > 0:
            raise IndexError('Mismatch between expected length of wheel data and actual.')
        if ret_wheel:
            trials.append(whlvel)
        elif ret_abswheel:
            trials.append(np.abs(whlvel))
    trialsdf['wheel_velocity'] = trials

    if ext_DLC:
        dataset_types = ['camera.times', 'trials.intervals', 'camera.dlc']
        video_type_left = 'left'
        video_type_right = 'right'
        # video_type_body = 'body'
        DLC_prob_threshold = 0.9
        # try:
        D = one.load(eid, dataset_types=dataset_types, dclass_output=True)  # , clobber=True)

        alf_path = Path(D.local_path[0]).parent.parent / 'alf'
        video_data = alf_path.parent / 'raw_video_data'

        velocity_L_paw_x_leftcam = []
        velocity_L_paw_y_leftcam = []
        velocity_R_paw_x_leftcam = []
        velocity_R_paw_y_leftcam = []
        velocity_L_paw_x_rightcam = []
        velocity_L_paw_y_rightcam = []
        velocity_R_paw_x_rightcam = []
        velocity_R_paw_y_rightcam = []

        # pqt and npy cam files need to be extracted differently
        print('Extracting DLC data')
        try:
            cam_left = alf.io.load_object(alf_path, '%sCamera' % video_type_left, namespace='ibl')
            cam_right = alf.io.load_object(alf_path, '%sCamera' %
                                           video_type_right, namespace='ibl')
            # cam_body = alf.io.load_object(alf_path, '%sCamera' % video_type_body, namespace = 'ibl')
            cam_left_times = cam_left.times
            cam_right_times = cam_right.times
            # cam_body_times = cam_body.times
            paw_l_x_thresholded = np.empty(len(cam_right['dlc'].paw_l_likelihood),)
            paw_l_x_thresholded[:] = np.NaN
            paw_l_y_thresholded = np.empty(len(cam_right['dlc'].paw_l_likelihood),)
            paw_l_y_thresholded[:] = np.NaN
            paw_r_x_thresholded = np.empty(len(cam_right['dlc'].paw_l_likelihood),)
            paw_r_x_thresholded[:] = np.NaN
            paw_r_y_thresholded = np.empty(len(cam_right['dlc'].paw_l_likelihood),)
            paw_r_y_thresholded[:] = np.NaN
            paw_l_x_leftcam_thresholded = np.empty(len(cam_left['dlc'].paw_l_likelihood),)
            paw_l_x_leftcam_thresholded[:] = np.NaN
            paw_l_y_leftcam_thresholded = np.empty(len(cam_left['dlc'].paw_l_likelihood),)
            paw_l_y_leftcam_thresholded[:] = np.NaN
            paw_r_x_leftcam_thresholded = np.empty(len(cam_left['dlc'].paw_l_likelihood),)
            paw_r_x_leftcam_thresholded[:] = np.NaN
            paw_r_y_leftcam_thresholded = np.empty(len(cam_left['dlc'].paw_l_likelihood),)
            paw_r_y_leftcam_thresholded[:] = np.NaN
            for k in range(0, len(cam_left['dlc'].paw_l_likelihood)):
                if cam_left['dlc'].paw_l_likelihood[k] > DLC_prob_threshold:
                    paw_l_x_leftcam_thresholded[k, ] = cam_left['dlc'].paw_l_x[k]
                    paw_l_y_leftcam_thresholded[k, ] = cam_left['dlc'].paw_l_y[k]
                if cam_left['dlc'].paw_r_likelihood[k] > DLC_prob_threshold:
                    paw_r_x_leftcam_thresholded[k, ] = cam_left['dlc'].paw_r_x[k]
                    paw_r_y_leftcam_thresholded[k, ] = cam_left['dlc'].paw_r_y[k]
            for k in range(0, len(cam_right['dlc'].paw_l_likelihood)):
                if cam_right['dlc'].paw_l_likelihood[k] > DLC_prob_threshold:
                    paw_l_x_thresholded[k, ] = cam_right['dlc'].paw_l_x[k]
                    paw_l_y_thresholded[k, ] = cam_right['dlc'].paw_l_y[k]
                if cam_right['dlc'].paw_r_likelihood[k] > DLC_prob_threshold:
                    paw_r_x_thresholded[k, ] = cam_right['dlc'].paw_r_x[k]
                    paw_r_y_thresholded[k, ] = cam_right['dlc'].paw_r_y[k]
            is_pqt = 0
        except:
            dlc_name_left = '_ibl_%sCamera.dlc.pqt' % video_type_left
            dlc_name_right = '_ibl_%sCamera.dlc.pqt' % video_type_right
            # dlc_name_body = '_ibl_%sCamera.dlc.pqt' % video_type_body
            dlc_path_left = alf_path / dlc_name_left
            dlc_path_right = alf_path / dlc_name_right
            # dlc_path_body = alf_path / dlc_name_body
            cam_left = pd.read_parquet(dlc_path_left, engine="fastparquet")
            cam_right = pd.read_parquet(dlc_path_right, engine="fastparquet")
            # cam_body = pd.read_parquet(dlc_path_body, engine="fastparquet")
            cam_left_times = alf.io.load_object(
                alf_path, '%sCamera' % video_type_left, namespace='ibl')
            cam_right_times = alf.io.load_object(
                alf_path, '%sCamera' % video_type_right, namespace='ibl')
            # cam_body_times = alf.io.load_object(alf_path, '%sCamera' % video_type_body, namespace = 'ibl')
            cam_left_times = cam_left_times.times
            cam_right_times = cam_right_times.times
            # cam_body_times = cam_body_times.times
            paw_l_x_thresholded = np.empty(len(cam_right.paw_l_likelihood),)
            paw_l_x_thresholded[:] = np.NaN
            paw_l_y_thresholded = np.empty(len(cam_right.paw_l_likelihood),)
            paw_l_y_thresholded[:] = np.NaN
            paw_r_x_thresholded = np.empty(len(cam_right.paw_l_likelihood),)
            paw_r_x_thresholded[:] = np.NaN
            paw_r_y_thresholded = np.empty(len(cam_right.paw_l_likelihood),)
            paw_r_y_thresholded[:] = np.NaN
            paw_l_x_leftcam_thresholded = np.empty(len(cam_left.paw_l_likelihood),)
            paw_l_x_leftcam_thresholded[:] = np.NaN
            paw_l_y_leftcam_thresholded = np.empty(len(cam_left.paw_l_likelihood),)
            paw_l_y_leftcam_thresholded[:] = np.NaN
            paw_r_x_leftcam_thresholded = np.empty(len(cam_left.paw_l_likelihood),)
            paw_r_x_leftcam_thresholded[:] = np.NaN
            paw_r_y_leftcam_thresholded = np.empty(len(cam_left.paw_l_likelihood),)
            paw_r_y_leftcam_thresholded[:] = np.NaN
            for k in range(0, len(cam_left.paw_l_likelihood)):
                if cam_left.paw_l_likelihood[k] > DLC_prob_threshold:
                    paw_l_x_leftcam_thresholded[k, ] = cam_left.paw_l_x[k]
                    paw_l_y_leftcam_thresholded[k, ] = cam_left.paw_l_y[k]
                if cam_left.paw_r_likelihood[k] > DLC_prob_threshold:
                    paw_r_x_leftcam_thresholded[k, ] = cam_left.paw_r_x[k]
                    paw_r_y_leftcam_thresholded[k, ] = cam_left.paw_r_y[k]
            for k in range(0, len(cam_right.paw_l_likelihood)):
                if cam_right.paw_l_likelihood[k] > DLC_prob_threshold:
                    paw_l_x_thresholded[k, ] = cam_right.paw_l_x[k]
                    paw_l_y_thresholded[k, ] = cam_right.paw_l_y[k]
                if cam_right.paw_r_likelihood[k] > DLC_prob_threshold:
                    paw_r_x_thresholded[k, ] = cam_right.paw_r_x[k]
                    paw_r_y_thresholded[k, ] = cam_right.paw_r_y[k]
            # print('it is pqt...')
            is_pqt = 1

        # A helper function to find closest time stamps
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        # find mean x/y vals for relevant period of each trial
        # print('finding nearest x/y vals for relevant period of each trial...')
        starttimes = trialsdf['trial_start']
        endtimes = trialsdf['trial_end']
        for (start, end) in np.vstack((starttimes, endtimes)).T:
            # define first and last frames for analysis range
            frame_start_left = find_nearest(cam_left_times, start) - 1
            frame_stop_left = find_nearest(cam_left_times, end) + 1  # + 1 is workaround?
            frame_start_right = find_nearest(cam_right_times, start) - 1
            frame_stop_right = find_nearest(cam_right_times, end) + 1  # + 1 is workaround?

            # length for each trial
            trialendind = np.ceil((end - start) / wheel_binsize).astype(int)

            # turning frames into velocity + direction
            paw_l_x_vector_pf = np.empty(frame_stop_right - frame_start_right,)
            paw_l_x_vector_pf[:] = np.NaN
            paw_l_y_vector_pf = np.empty(frame_stop_right - frame_start_right,)
            paw_l_y_vector_pf[:] = np.NaN
            paw_r_x_vector_pf = np.empty(frame_stop_right - frame_start_right,)
            paw_r_x_vector_pf[:] = np.NaN
            paw_r_y_vector_pf = np.empty(frame_stop_right - frame_start_right,)
            paw_r_y_vector_pf[:] = np.NaN
            paw_l_x_leftcam_vector_pf = np.empty(frame_stop_left - frame_start_left,)
            paw_l_x_leftcam_vector_pf[:] = np.NaN
            paw_l_y_leftcam_vector_pf = np.empty(frame_stop_left - frame_start_left,)
            paw_l_y_leftcam_vector_pf[:] = np.NaN
            paw_r_x_leftcam_vector_pf = np.empty(frame_stop_left - frame_start_left,)
            paw_r_x_leftcam_vector_pf[:] = np.NaN
            paw_r_y_leftcam_vector_pf = np.empty(frame_stop_left - frame_start_left,)
            paw_r_y_leftcam_vector_pf[:] = np.NaN

            # defining movement vectors as subtractions of positions only where not NaN
            paw_l_x_thresholded_startstop = paw_l_x_thresholded[frame_start_right:frame_stop_right]
            paw_l_y_thresholded_startstop = paw_l_y_thresholded[frame_start_right:frame_stop_right]
            numreal = sum(~np.isnan(paw_l_x_thresholded_startstop))
            if numreal > 1:
                indices_real = np.where(~np.isnan(paw_l_x_thresholded_startstop))
                indices_real = indices_real[0]
                for l in indices_real:
                    if l == indices_real[0]:
                        prev_indx = indices_real[0]
                        continue
                    else:
                        paw_l_x_vector_pf[l, ] = paw_l_x_thresholded_startstop[l] - \
                            paw_l_x_thresholded_startstop[prev_indx]
                        paw_l_y_vector_pf[l, ] = paw_l_y_thresholded_startstop[l] - \
                            paw_l_y_thresholded_startstop[prev_indx]
                        prev_indx = l
            paw_r_x_thresholded_startstop = paw_r_x_thresholded[frame_start_right:frame_stop_right]
            paw_r_y_thresholded_startstop = paw_r_y_thresholded[frame_start_right:frame_stop_right]
            numreal = sum(~np.isnan(paw_r_x_thresholded_startstop))
            if numreal > 1:
                indices_real = np.where(~np.isnan(paw_r_x_thresholded_startstop))
                indices_real = indices_real[0]
                for l in indices_real:
                    if l == indices_real[0]:
                        prev_indx = indices_real[0]
                        continue
                    else:
                        paw_r_x_vector_pf[l, ] = paw_r_x_thresholded_startstop[l] - \
                            paw_r_x_thresholded_startstop[prev_indx]
                        paw_r_y_vector_pf[l, ] = paw_r_y_thresholded_startstop[l] - \
                            paw_r_y_thresholded_startstop[prev_indx]
                        prev_indx = l

            paw_l_x_leftcam_thresholded_startstop = paw_l_x_leftcam_thresholded[frame_start_left:frame_stop_left]
            paw_l_y_leftcam_thresholded_startstop = paw_l_y_leftcam_thresholded[frame_start_left:frame_stop_left]
            numreal = sum(~np.isnan(paw_l_x_leftcam_thresholded_startstop))
            if numreal > 1:
                indices_real = np.where(~np.isnan(paw_l_x_leftcam_thresholded_startstop))
                indices_real = indices_real[0]
                for l in indices_real:
                    if l == indices_real[0]:
                        prev_indx = indices_real[0]
                        continue
                    else:
                        paw_l_x_leftcam_vector_pf[l, ] = paw_l_x_leftcam_thresholded_startstop[l] - \
                            paw_l_x_leftcam_thresholded_startstop[prev_indx]
                        paw_l_y_leftcam_vector_pf[l, ] = paw_l_y_leftcam_thresholded_startstop[l] - \
                            paw_l_y_leftcam_thresholded_startstop[prev_indx]
                        prev_indx = l
            paw_r_x_leftcam_thresholded_startstop = paw_r_x_leftcam_thresholded[frame_start_left:frame_stop_left]
            paw_r_y_leftcam_thresholded_startstop = paw_r_y_leftcam_thresholded[frame_start_left:frame_stop_left]
            numreal = sum(~np.isnan(paw_r_x_leftcam_thresholded_startstop))
            if numreal > 1:
                indices_real = np.where(~np.isnan(paw_r_x_leftcam_thresholded_startstop))
                indices_real = indices_real[0]
                for l in indices_real:
                    if l == indices_real[0]:
                        prev_indx = indices_real[0]
                        continue
                    else:
                        paw_r_x_leftcam_vector_pf[l, ] = paw_r_x_leftcam_thresholded_startstop[l] - \
                            paw_r_x_leftcam_thresholded_startstop[prev_indx]
                        paw_r_y_leftcam_vector_pf[l, ] = paw_r_y_leftcam_thresholded_startstop[l] - \
                            paw_r_y_leftcam_thresholded_startstop[prev_indx]
                        prev_indx = l

            times_left = cam_left_times[frame_start_left: frame_stop_left]
            times_right = cam_right_times[frame_start_right: frame_stop_right]

            sync_paw_l_x_leftcam = sync(wheel_binsize, times=times_left,
                                        values=paw_l_x_leftcam_vector_pf, interp='previous')
            sync_paw_l_y_leftcam = sync(wheel_binsize, times=times_left,
                                        values=paw_l_y_leftcam_vector_pf, interp='previous')
            sync_paw_r_x_leftcam = sync(wheel_binsize, times=times_left,
                                        values=paw_r_x_leftcam_vector_pf, interp='previous')
            sync_paw_r_y_leftcam = sync(wheel_binsize, times=times_left,
                                        values=paw_r_y_leftcam_vector_pf, interp='previous')
            trialstartind = np.searchsorted(sync_paw_l_x_leftcam.times, 0)
            trialendind = np.ceil((end - start) / wheel_binsize).astype(int)
            sync_paw_l_x_leftcam_velocity = sync_paw_l_x_leftcam.values[trialstartind:trialendind + trialstartind]
            sync_paw_l_y_leftcam_velocity = sync_paw_l_y_leftcam.values[trialstartind:trialendind + trialstartind]
            sync_paw_r_x_leftcam_velocity = sync_paw_r_x_leftcam.values[trialstartind:trialendind + trialstartind]
            sync_paw_r_y_leftcam_velocity = sync_paw_r_y_leftcam.values[trialstartind:trialendind + trialstartind]
            # if np.abs((trialendind - len(sync_paw_l_x_leftcam_velocity))) > 0:
            #     raise IndexError('Mismatch between expected length of wheel data and actual.')
            # print(trialendind - len(sync_paw_l_x_leftcam_velocity))

            sync_paw_l_x_rightcam = sync(wheel_binsize, times=times_right,
                                         values=paw_l_x_vector_pf, interp='previous')
            sync_paw_l_y_rightcam = sync(wheel_binsize, times=times_right,
                                         values=paw_l_y_vector_pf, interp='previous')
            sync_paw_r_x_rightcam = sync(wheel_binsize, times=times_right,
                                         values=paw_r_x_vector_pf, interp='previous')
            sync_paw_r_y_rightcam = sync(wheel_binsize, times=times_right,
                                         values=paw_r_y_vector_pf, interp='previous')
            trialstartind = np.searchsorted(sync_paw_l_x_rightcam.times, 0)
            trialendind = np.ceil((end - start) / wheel_binsize).astype(int)
            sync_paw_l_x_rightcam_velocity = sync_paw_l_x_rightcam.values[trialstartind:trialendind + trialstartind]
            sync_paw_l_y_rightcam_velocity = sync_paw_l_y_rightcam.values[trialstartind:trialendind + trialstartind]
            sync_paw_r_x_rightcam_velocity = sync_paw_r_x_rightcam.values[trialstartind:trialendind + trialstartind]
            sync_paw_r_y_rightcam_velocity = sync_paw_r_y_rightcam.values[trialstartind:trialendind + trialstartind]
            # if np.abs((trialendind - len(sync_paw_l_x_rightcam_velocity))) > 0:
            #     raise IndexError('Mismatch between expected length of wheel data and actual.')

            # first value is always nan, can probably fix that, also still occasionally nans in variable
            sync_paw_l_x_leftcam_velocity = np.nan_to_num(sync_paw_l_x_leftcam_velocity, nan=0.0)
            sync_paw_l_y_leftcam_velocity = np.nan_to_num(sync_paw_l_y_leftcam_velocity, nan=0.0)
            sync_paw_r_x_leftcam_velocity = np.nan_to_num(sync_paw_r_x_leftcam_velocity, nan=0.0)
            sync_paw_r_y_leftcam_velocity = np.nan_to_num(sync_paw_r_y_leftcam_velocity, nan=0.0)
            sync_paw_l_x_rightcam_velocity = np.nan_to_num(sync_paw_l_x_rightcam_velocity, nan=0.0)
            sync_paw_l_y_rightcam_velocity = np.nan_to_num(sync_paw_l_y_rightcam_velocity, nan=0.0)
            sync_paw_r_x_rightcam_velocity = np.nan_to_num(sync_paw_r_x_rightcam_velocity, nan=0.0)
            sync_paw_r_y_rightcam_velocity = np.nan_to_num(sync_paw_r_y_rightcam_velocity, nan=0.0)
            velocity_L_paw_x_leftcam.append(sync_paw_l_x_leftcam_velocity)
            velocity_L_paw_y_leftcam.append(sync_paw_l_y_leftcam_velocity)
            velocity_R_paw_x_leftcam.append(sync_paw_r_x_leftcam_velocity)
            velocity_R_paw_y_leftcam.append(sync_paw_r_y_leftcam_velocity)
            velocity_L_paw_x_rightcam.append(sync_paw_l_x_rightcam_velocity)
            velocity_L_paw_y_rightcam.append(sync_paw_l_y_rightcam_velocity)
            velocity_R_paw_x_rightcam.append(sync_paw_r_x_rightcam_velocity)
            velocity_R_paw_y_rightcam.append(sync_paw_r_y_rightcam_velocity)

        trialsdf['DLC_Lpaw_xvel_leftcam'] = velocity_L_paw_x_leftcam
        # trialsdf['DLC_Lpaw_y_leftcam'] = velocity_L_paw_y_leftcam
        trialsdf['DLC_Rpaw_xvel_leftcam'] = velocity_R_paw_x_leftcam
        # trialsdf['DLC_Rpaw_y_leftcam'] = velocity_R_paw_y_leftcam
        # trialsdf['DLC_Lpaw_x_rightcam'] = velocity_L_paw_x_rightcam
        # trialsdf['DLC_Lpaw_y_rightcam'] = velocity_L_paw_y_rightcam
        # trialsdf['DLC_Rpaw_x_rightcam'] = velocity_R_paw_x_rightcam
        # trialsdf['DLC_Rpaw_y_rightcam'] = velocity_R_paw_y_rightcam

    return trialsdf