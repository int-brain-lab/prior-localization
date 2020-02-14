# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import listdir, mkdir
from os.path import join, isdir
import alf.io as ioalf
import matplotlib.pyplot as plt
import shutil
import brainbox as bb
import pandas as pd
import numpy as np
import seaborn as sns
from functions_5HT import download_data, paths, sessions

DOWNLOAD = False
OVERWRITE = True
FRONTAL_CONTROL = 'Frontal'
TRIAL_CENTERS = np.arange(-15, 31, 5)
TRIAL_WIN = 10
PRE_TIME = 1
POST_TIME = 0

if FRONTAL_CONTROL == 'Frontal':
    sessions, _ = sessions()
elif FRONTAL_CONTROL == 'Control':
    _, sessions = sessions()

DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'PSTH', 'BlockSwitch')
for i in range(sessions.shape[0]):
    # Download data if required
    if DOWNLOAD is True:
        download_data(sessions.loc[i, 'subject'], sessions.loc[i, 'date'])

    # Get paths
    ses_nr = listdir(join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects',
                          sessions.loc[i, 'subject'], sessions.loc[i, 'date']))[0]
    session_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects',
                        sessions.loc[i, 'subject'], sessions.loc[i, 'date'], ses_nr)
    alf_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects', sessions.loc[i, 'subject'],
                    sessions.loc[i, 'date'], ses_nr, 'alf')
    probe_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects', sessions.loc[i, 'subject'],
                      sessions.loc[i, 'date'], ses_nr, 'alf', 'probe%s' % sessions.loc[i, 'probe'])

    # Load in data
    spikes = ioalf.load_object(probe_path, 'spikes')
    clusters = ioalf.load_object(probe_path, 'clusters')
    trials = ioalf.load_object(alf_path, '_ibl_trials')

    # Only use single units
    spikes.times = spikes.times[np.isin(
            spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]
    spikes.clusters = spikes.clusters[np.isin(
            spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]

    # Calculate whether neuron discriminates blocks
    trial_times = trials.stimOn_times[
                        ((trials.probabilityLeft > 0.55)
                         | (trials.probabilityLeft < 0.45))]
    trial_blocks = (trials.probabilityLeft[
            (((trials.probabilityLeft > 0.55)
              | (trials.probabilityLeft < 0.45)))] > 0.55).astype(int)
    diff_units = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                             trial_times, trial_blocks,
                                             pre_time=PRE_TIME, post_time=POST_TIME,
                                             alpha=0.01)[0]

    # Get spike counts for all trials
    times = np.column_stack(((trials.stimOn_times - PRE_TIME), (trials.stimOn_times + POST_TIME)))
    spike_counts, cluster_ids = bb.task._get_spike_counts_in_bins(spikes.times,
                                                                  spikes.clusters, times)

    # Get block switches
    switch_to_l = [i for i, x in enumerate(np.diff(trials.probabilityLeft) > 0.3) if x]
    switch_to_r = [i for i, x in enumerate(np.diff(trials.probabilityLeft) < -0.3) if x]
    all_switches = np.append(switch_to_l, switch_to_r)
    switch_sides = np.append(['left']*len(switch_to_l), ['right']*len(switch_to_r))
    block_switch = pd.DataFrame(columns=['mean_spike_count', 'trial_center',
                                         'switch_side', 'cluster_id'])
    for s, switch in enumerate(all_switches):
        for t, trial in enumerate(TRIAL_CENTERS):
            this_counts = spike_counts[np.isin(cluster_ids, diff_units),
                                       int(switch+(trial-(TRIAL_WIN/2))):int(
                                                               switch+(trial+(TRIAL_WIN/2)))]
            block_switch = block_switch.append(pd.DataFrame(
                                        data={'mean_spike_count': np.mean(this_counts, axis=1),
                                              'trial_center': trial,
                                              'cluster_id': diff_units,
                                              'switch_side': switch_sides[s]}), sort=False)

    # Make directories
    if (isdir(join(FIG_PATH, FRONTAL_CONTROL, '%s_%s' % (sessions.loc[i, 'subject'],
                                                         sessions.loc[i, 'date'])))
            and (OVERWRITE is True)):
        shutil.rmtree(join(FIG_PATH, FRONTAL_CONTROL, '%s_%s' % (sessions.loc[i, 'subject'],
                                                                 sessions.loc[i, 'date'])))
    if not isdir(join(FIG_PATH, FRONTAL_CONTROL, '%s_%s' % (sessions.loc[i, 'subject'],
                                                            sessions.loc[i, 'date']))):
        mkdir(join(FIG_PATH, FRONTAL_CONTROL, '%s_%s' % (sessions.loc[i, 'subject'],
                                                         sessions.loc[i, 'date'])))

        for n, cluster in enumerate(diff_units):
            fig, ax = plt.subplots(1, 1)
            sns.lineplot(x='trial_center', y='mean_spike_count', hue='switch_side',
                         data=block_switch.loc[block_switch['cluster_id'] == cluster], ci=68)
            ax.set(ylabel='Mean spike rate (spk/s)', xlabel='Trials relative to block switch')
            plt.savefig(join(FIG_PATH, FRONTAL_CONTROL,
                             '%s_%s' % (sessions.loc[i, 'subject'], sessions.loc[i, 'date']),
                             'p%s_d%s_n%s' % (sessions.loc[i, 'probe'],
                                              int(clusters.depths[
                                                      clusters.metrics.cluster_id == cluster][0]),
                                              cluster)))
            plt.close(fig)
