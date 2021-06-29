"""
Utility functions for the prior-localization repository
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import brainbox.io.one as bbone
from brainbox.singlecell import calculate_peths
from brainbox.plot import peri_event_time_histogram


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
