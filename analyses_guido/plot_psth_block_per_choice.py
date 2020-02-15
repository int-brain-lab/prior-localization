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
import numpy as np
from scipy import stats
from functions_5HT import download_data, paths, sessions

download = False
overwrite = True
frontal_control = 'Frontal'

if frontal_control == 'Frontal':
    sessions, _ = sessions()
elif frontal_control == 'Control':
    _, sessions = sessions()

DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'PSTH', 'Blocks')
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

        # Calculate whether neuron discriminates for left choice
        trial_times = trials.goCue_times[
                            ((trials.probabilityLeft > 0.55)
                             | (trials.probabilityLeft < 0.55)) & (trials.choice == -1)]
        trial_blocks = (trials.probabilityLeft[
                (((trials.probabilityLeft > 0.55)
                  | (trials.probabilityLeft < 0.55))) & (trials.choice == -1)] > 0.55).astype(int)

        diff_units = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                 trial_times, trial_blocks,
                                                 pre_time=0.5, post_time=0, alpha=0.01)[0]

        print('%d out of %d neurons differentiate blocks for left choice' % (
                                        len(diff_units), len(np.unique(spikes.clusters))))

        for n, cluster in enumerate(diff_units):
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                              trials.stimOn_times[
                                                  ((trials.probabilityLeft > 0.55)
                                                   & (trials.choice == -1))],
                                              cluster, t_before=1, t_after=2,
                                              error_bars='sem', ax=ax)
            y_lim_1 = ax.get_ylim()
            bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                              trials.stimOn_times[((trials.probabilityLeft < 0.45)
                                                                   & (trials.choice == -1))],
                                              cluster, t_before=1, t_after=2, error_bars='sem',
                                              pethline_kwargs={'color': 'red', 'lw': 2},
                                              errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
            y_lim_2 = ax.get_ylim()
            if y_lim_1[1] > y_lim_2[1]:
                ax.set(ylim=y_lim_1)
            plt.legend(['Left block', 'Right block'])
            plt.title('Leftward choices')
            plt.savefig(join(FIG_PATH, frontal_control,
                             '%s_%s' % (sessions.loc[i, 'subject'], sessions.loc[i, 'date']),
                             'left_p%s_d%s_n%s' % (sessions.loc[i, 'probe'],
                                                   int(clusters.depths[
                                                       clusters.metrics.cluster_id == cluster][0]),
                                                   cluster)))
            plt.close(fig)

        # Calculate whether neuron discriminates for right choice
        trial_times = trials.goCue_times[
                            ((trials.probabilityLeft > 0.55)
                             | (trials.probabilityLeft < 0.55)) & (trials.choice == 1)]
        trial_blocks = (trials.probabilityLeft[
                (((trials.probabilityLeft > 0.55)
                  | (trials.probabilityLeft < 0.55))) & (trials.choice == 1)] > 0.55).astype(int)

        diff_units = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                 trial_times, trial_blocks,
                                                 pre_time=0.5, post_time=0, alpha=0.01)[0]

        print('%d out of %d neurons differentiate blocks for right choice' % (
                                        len(diff_units), len(np.unique(spikes.clusters))))

        for n, cluster in enumerate(diff_units):
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                              trials.stimOn_times[
                                                  ((trials.probabilityLeft > 0.55)
                                                   & (trials.choice == 1))],
                                              cluster, t_before=1, t_after=2,
                                              error_bars='sem', ax=ax)
            y_lim_1 = ax.get_ylim()
            bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                              trials.stimOn_times[((trials.probabilityLeft < 0.45)
                                                                   & (trials.choice == 1))],
                                              cluster, t_before=1, t_after=2, error_bars='sem',
                                              pethline_kwargs={'color': 'red', 'lw': 2},
                                              errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
            y_lim_2 = ax.get_ylim()
            if y_lim_1[1] > y_lim_2[1]:
                ax.set(ylim=y_lim_1)
            plt.legend(['Left block', 'Right block'])
            plt.title('Rightward choices')
            plt.savefig(join(FIG_PATH, frontal_control,
                             '%s_%s' % (sessions.loc[i, 'subject'], sessions.loc[i, 'date']),
                             'right_p%s_d%s_n%s' % (sessions.loc[i, 'probe'],
                                                    int(clusters.depths[
                                                      clusters.metrics.cluster_id == cluster][0]),
                                                    cluster)))
            plt.close(fig)
