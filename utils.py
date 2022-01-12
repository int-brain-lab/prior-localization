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
import seaborn as sns
import matplotlib.pyplot as plt

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


def check_eid_for_matching_metrics(eid, sessdf, one=None):
    one = one or ONE()
    probesdict = {}
    for probe in sessdf.xs(eid, level='eid').probe:
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid,
                                                                    one=one,
                                                                    probe=probe,
                                                                    aligned=True)
        regions = clusters[probe].atlas_id
        try:
            metrics = clusters[probe].metrics
        except AttributeError:
            probesdict[probe] = 'No metrics'
            continue
        if (regions.shape[0] - 1) != metrics.index.max():
            probesdict[probe] = 'Metric mismatch to indices of clusters.atlas_id'
        else:
            probesdict[probe] = 'Matching metrics'
    return probesdict

def ridge_plot(df, xcol, ycol, palette=sns.cubehelix_palette(10, rot=-.25, light=.7)):
    g = sns.FacetGrid(df, row=ycol, hue=ycol, aspect=15., height=.5, palette=palette)
    g.map(sns.kdeplot, xcol,
          clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    
    g.map(label, xcol)
    g.figure.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    return g
