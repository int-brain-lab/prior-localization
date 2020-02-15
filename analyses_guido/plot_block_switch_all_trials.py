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
PRE_TIME = 1
POST_TIME = 0
TRIAL_WIN_SIZE = 10

if FRONTAL_CONTROL == 'Frontal':
    sessions, _ = sessions()
elif FRONTAL_CONTROL == 'Control':
    _, sessions = sessions()

DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'PSTH', 'BlockSwitch_TrialProgression')
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
            # Get average spike rate in sliding window
            trial_win_centers = np.arange(int(TRIAL_WIN_SIZE/2),
                                          int(trials.stimOn_times.shape[0]-(TRIAL_WIN_SIZE/2)))
            avg_rate = np.empty(len(trial_win_centers))
            for w, trial in enumerate(trial_win_centers):
                avg_rate[w] = np.mean(spike_counts[
                                    cluster_ids == cluster,
                                    int(trial-(TRIAL_WIN_SIZE/2)):int(trial+(TRIAL_WIN_SIZE/2))])

            fig, ax1 = plt.subplots(1, 1)
            ax1.plot(np.arange(1, trials.stimOn_times.shape[0]+1), trials.probabilityLeft,
                     color=[0.6, 0.6, 0.6])
            ax1.set_ylabel('Probability of left trial', color='tab:gray')
            ax1.set(xlabel='Trials relative to block switch')
            ax2 = ax1.twinx()
            ax2.plot(trial_win_centers, avg_rate, 'r')
            ax2.set_ylabel('Mean spike rate (spk/s)', color='tab:red')
            plt.tight_layout()
            plt.savefig(join(FIG_PATH, FRONTAL_CONTROL,
                             '%s_%s' % (sessions.loc[i, 'subject'], sessions.loc[i, 'date']),
                             'p%s_d%s_n%s' % (sessions.loc[i, 'probe'],
                                              int(clusters.depths[
                                                      clusters.metrics.cluster_id == cluster][0]),
                                              cluster)))
            plt.close(fig)
