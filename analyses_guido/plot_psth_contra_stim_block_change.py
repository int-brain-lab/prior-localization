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
import scipy.stats as stats
import brainbox as bb
import seaborn as sns
import numpy as np
from functions_5HT import download_data, paths, sessions

download = False
overwrite = True
frontal_control = 'Frontal'
INCL_UNITS = 'blocks'  # blocks or contrastim
MIN_CONTRAST = 0.1
FIRST_TRIALS = [10, 20, 30]
TRIAL_WIN = 10
PLOT_TIME_WIN = [0.5, 1]
TEST_TIME_WIN = [0, 0.3]
MIN_SPIKE_COUNT = 20

if frontal_control == 'Frontal':
    sessions, _ = sessions()
elif frontal_control == 'Control':
    _, sessions = sessions()

DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'PSTH', 'ContraStim_BlockChange')
colors = sns.cubehelix_palette(len(FIRST_TRIALS), start=0.5, rot=-0.75, dark=0.05, reverse=True)

for i in range(sessions.shape[0]):
    # Download data if required
    if download is True:
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

    # Get block switches
    switch_to_l = [i for i, x in enumerate(np.diff(trials.probabilityLeft) > 0.3) if x]
    switch_to_r = [i for i, x in enumerate(np.diff(trials.probabilityLeft) < -0.3) if x]
    trial_id = np.arange(len(trials.stimOn_times))
    trial_ind = np.zeros(len(trials.stimOn_times))
    for t, trial in enumerate(FIRST_TRIALS):
        for s, switch in enumerate(switch_to_l):
            trial_ind[((trial_id >= switch+(trial-TRIAL_WIN)) & (trial_id <= switch+trial)
                       & (trials.contrastLeft > MIN_CONTRAST))] = t+1

    # Get significant units
    if INCL_UNITS == 'contrastim':
        spike_counts = []
        for t, trial in enumerate(FIRST_TRIALS):
            times = np.column_stack(((trials.stimOn_times[trial_ind == t+1] - TEST_TIME_WIN[0]),
                                     (trials.stimOn_times[trial_ind == t+1] + TEST_TIME_WIN[1])))
            counts, cluster_ids = bb.task._get_spike_counts_in_bins(spikes.times,
                                                                    spikes.clusters, times)
            spike_counts.append(counts)
        p_values = np.zeros(len(cluster_ids))
        for j in range(len(cluster_ids)):
            this_spikes = []
            for t in range(len(spike_counts)):
                this_spikes.append(spike_counts[t][j, :])
            if np.sum([np.sum(el) for el in this_spikes]) < MIN_SPIKE_COUNT:
                p_values[j] = 1
            else:
                _, p_values[j] = stats.kruskal(*this_spikes)
        sig_units = cluster_ids[p_values < 0.05]
    elif INCL_UNITS == 'blocks':
        trial_times = trials.stimOn_times[
                            ((trials.probabilityLeft > 0.55)
                             | (trials.probabilityLeft < 0.45))]
        trial_blocks = (trials.probabilityLeft[
                (((trials.probabilityLeft > 0.55)
                  | (trials.probabilityLeft < 0.45)))] > 0.55).astype(int)
        sig_units = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                trial_times, trial_blocks,
                                                pre_time=1, post_time=0,
                                                alpha=0.01)[0]

    # Make directories
    if (isdir(join(FIG_PATH, frontal_control, '%s_%s' % (sessions.loc[i, 'subject'],
                                                         sessions.loc[i, 'date'])))
            and (overwrite is True)):
        shutil.rmtree(join(FIG_PATH, frontal_control, '%s_%s' % (sessions.loc[i, 'subject'],
                                                                 sessions.loc[i, 'date'])))
    if not isdir(join(FIG_PATH, frontal_control, '%s_%s' % (sessions.loc[i, 'subject'],
                                                            sessions.loc[i, 'date']))):
        mkdir(join(FIG_PATH, frontal_control, '%s_%s' % (sessions.loc[i, 'subject'],
                                                         sessions.loc[i, 'date'])))

        # Get block switches
        for c, cluster in enumerate(sig_units):
            f, ax = plt.subplots(1, 1)
            y_lim_max = np.zeros(len(FIRST_TRIALS))
            for t, trial in enumerate(FIRST_TRIALS):
                bb.plot.peri_event_time_histogram(
                    spikes.times, spikes.clusters, trials.stimOn_times[trial_ind == t+1],
                    cluster, t_before=PLOT_TIME_WIN[0], t_after=PLOT_TIME_WIN[1], error_bars='sem',
                    pethline_kwargs={'color': colors[t], 'lw': 2},
                    errbar_kwargs={'color': colors[t], 'alpha': 0.5}, ax=ax)
                y_lim_max[t] = ax.get_ylim()[1]
            ax.set(ylim=[0, np.max(y_lim_max)])
            plt.savefig(join(FIG_PATH, frontal_control, '%s_%s' % (sessions.loc[i, 'subject'],
                                                                   sessions.loc[i, 'date']),
                             'p%s_d%s_n%s' % (sessions.loc[i, 'probe'], int(clusters.depths[
                                 clusters.metrics.cluster_id == cluster][0]),
                                             cluster)))
            plt.close(f)
