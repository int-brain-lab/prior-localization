# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import listdir, mkdir
from os.path import join, isdir
import alf.io as ioalf
import matplotlib.pyplot as plt
import brainbox as bb
import numpy as np
import shutil
from functions_5HT import download_data, paths, sessions

download = False
overwrite = True
frontal_control = 'Frontal'

if frontal_control == 'Frontal':
    sessions, _ = sessions()
elif frontal_control == 'Control':
    _, sessions = sessions()

DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'PSTH', 'ContraStim')
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

    # Get trial indices
    r_in_l_block = trials.stimOn_times[((trials.probabilityLeft > 0.55)
                                        & (trials.contrastRight > 0.1))]
    r_in_r_block = trials.stimOn_times[((trials.probabilityLeft < 0.45)
                                        & (trials.contrastRight > 0.1))]
    l_in_r_block = trials.stimOn_times[((trials.probabilityLeft < 0.45)
                                        & (trials.contrastLeft > 0.1))]
    l_in_l_block = trials.stimOn_times[((trials.probabilityLeft > 0.55)
                                        & (trials.contrastLeft > 0.1))]

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

        # Right stim
        sig_units = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                np.append(r_in_l_block,
                                                          r_in_r_block),
                                                np.append(np.zeros(len(r_in_l_block)),
                                                          np.ones(len(r_in_r_block))),
                                                pre_time=0, post_time=0.5,
                                                test='ranksums', alpha=0.05)[0]
        for n, cluster in enumerate(sig_units):
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters, r_in_r_block,
                                              cluster, t_before=1, t_after=2,
                                              error_bars='sem', ax=ax)
            y_lim_1 = ax.get_ylim()
            bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters, r_in_l_block,
                                              cluster, t_before=1, t_after=2, error_bars='sem',
                                              pethline_kwargs={'color': 'red', 'lw': 2},
                                              errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
            y_lim_2 = ax.get_ylim()
            if y_lim_1[1] > y_lim_2[1]:
                ax.set(ylim=y_lim_1)
            plt.legend(['Consistent', 'Inconsistent'])
            plt.title('Stimulus Onset (right side)')
            plt.savefig(join(FIG_PATH, frontal_control,
                             '%s_%s' % (sessions.loc[i, 'subject'], sessions.loc[i, 'date']),
                             'p%s_c%s_n%s' % (sessions.loc[i, 'probe'],
                                              clusters.channels[n], cluster)))
            plt.close(fig)

        # Left stim
        sig_units = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                np.append(l_in_r_block,
                                                          l_in_l_block),
                                                np.append(np.zeros(len(l_in_r_block)),
                                                          np.ones(len(l_in_l_block))),
                                                pre_time=0, post_time=0.5,
                                                test='ranksums', alpha=0.05)[0]
        for n, cluster in enumerate(sig_units):
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters, l_in_l_block,
                                              cluster, t_before=1, t_after=2,
                                              error_bars='sem', ax=ax)
            y_lim_1 = ax.get_ylim()
            bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters, l_in_r_block,
                                              cluster, t_before=1, t_after=2, error_bars='sem',
                                              pethline_kwargs={'color': 'red', 'lw': 2},
                                              errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
            y_lim_2 = ax.get_ylim()
            if y_lim_1[1] > y_lim_2[1]:
                ax.set(ylim=y_lim_1)
            plt.legend(['Consistent', 'Inconsistent'])
            plt.title('Stimulus Onset (left side)')
            plt.savefig(join(FIG_PATH, frontal_control,
                             '%s_%s' % (sessions.loc[i, 'subject'], sessions.loc[i, 'date']),
                             'p%s_c%s_n%s' % (sessions.loc[i, 'probe'],
                                              clusters.channels[n], cluster)))
            plt.close(fig)
