#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020

Get LDA score between blocks of all recording done by Guido

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import alf.io
import seaborn as sns
import brainbox as bb
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from functions_5HT import paths, plot_settings, one_session_path
from oneibl.one import ONE
one = ONE()

# Settings
PRE_TIME = 1
POST_TIME = -0.5
MIN_TRIALS = 200
MIN_NEURONS = 20
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# Get list of recordings
eids, ses_info = one.search(user='guido', dataset_types='spikes.times', details=True)

lda_result = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    print('Processing session %d of %d' % (i+1, len(eids)))
    session_path = one_session_path(eid)
    trials = one.load_object(eid, 'trials')
    probes = one.load_object(eid, 'probes', download_only=False)
    if (not hasattr(trials, 'stimOn_times')
            or (trials.stimOn_times.shape[0] != trials.probabilityLeft.shape[0])
            or (not hasattr(probes, 'trajectory'))):
        print('Invalid data, skipping recording')
        continue
    for p in range(len(probes['trajectory'])):
        # Select shallow penetrations
        # if (probes['trajectory'][p]['theta'] == 15)and (probes['trajectory'][p]['depth'] < 4500):
        probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
        try:
            spikes = alf.io.load_object(probe_path, object='spikes')
            clusters = alf.io.load_object(probe_path, object='clusters')
        except Exception:
            print('Could not load spikes or clusters Bunch, skipping recording')
            continue

        # Only use good single units
        clusters_to_use = clusters.metrics.ks2_label == 'good'
        if clusters_to_use.sum() < MIN_NEURONS:
            continue
        spikes.times = spikes.times[
                np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
        spikes.clusters = spikes.clusters[
                np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
        clusters.depths = clusters.depths[clusters_to_use]
        cluster_ids = clusters.metrics.cluster_id[clusters_to_use]

        # Get spike counts for all trials
        trial_times = trials.stimOn_times[(trials.probabilityLeft > 0.55)
                                          | (trials.probabilityLeft < 0.45)]
        if trial_times.shape[0] < MIN_TRIALS:
            continue
        times = np.column_stack(((trial_times - PRE_TIME), (trial_times + POST_TIME)))
        spike_counts, cluster_ids = bb.task._get_spike_counts_in_bins(spikes.times,
                                                                      spikes.clusters, times)
        trial_blocks = (trials.probabilityLeft[
                                (((trials.probabilityLeft > 0.55)
                                  | (trials.probabilityLeft < 0.45)))] > 0.55).astype(int)

        # Transform to LDA
        lda = LDA(n_components=1)
        lda_transform = lda.fit_transform(np.rot90(spike_counts), trial_blocks)

        # Correlate probability left with lda score
        r = stats.pearsonr(np.rot90(lda_transform)[0],
                           trials.probabilityLeft[
                               (trials.probabilityLeft > 0.55)
                               | (trials.probabilityLeft < 0.45)])[0]

        # Add to dataframe
        nickname = ses_info[i]['subject']
        ses_date = ses_info[i]['start_time'][:10]
        lda_result = lda_result.append(pd.DataFrame(
            index=[0], data={'subject': nickname, 'date': ses_date, 'eid': eid,
                             'r': r,
                             'ML': probes.trajectory[p]['x'],
                             'AP': probes.trajectory[p]['y'],
                             'DV': probes.trajectory[p]['z'],
                             'phi': probes.trajectory[p]['phi'],
                             'theta': probes.trajectory[p]['theta'],
                             'depth': probes.trajectory[p]['depth']}))

lda_result.to_csv(join(DATA_PATH, 'lda_block_all_my_recordings'))

# Plot
Y_LIM = [-6000, 4000]
X_LIM = [-5000, 5000]

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
plot_h = sns.scatterplot(x='ML', y='AP', data=lda_result, hue='r', palette='YlOrRd', size='r',
                         sizes=(50, 100), ax=ax1)

# Fix legend
leg = plot_h.legend(loc=(1.05, 0.5))
leg.texts[0].set_text('Corr. coef. (r)')
leg.texts[1].set_text('0.4')
leg.texts[2].set_text('0.6')
leg.texts[3].set_text('0.8')
leg.texts[4].set_text('1')

plot_settings()
plt.savefig(join(FIG_PATH, 'LDA_all_my_recordings'))
