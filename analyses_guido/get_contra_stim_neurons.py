"""

Get percentage of neurons that significantly differentiate stimuli that are consistent or
inconsistent with the block probability.

"""


from os import mkdir
from os.path import join, isdir
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
import alf.io
import pandas as pd
import brainbox as bb
import seaborn as sns
import shutil
from scipy import stats
import numpy as np
from functions_5HT import paths, one_session_path
from oneibl.one import ONE
one = ONE()

# Get list of recordings
eids, ses_info = one.search(user='guido', dataset_types='spikes.times', details=True)

# Set path to save plots
DATA_PATH, FIG_PATH, DATA_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

resp = pd.DataFrame()
for i, eid in enumerate(eids):
    # Load in data
    print('Processing recording %d of %d' % (i+1, len(eids)))
    session_path = one_session_path(eid)
    spikes = one.load_object(eid, 'spikes', download_only=True)
    trials = one.load_object(eid, 'trials')
    if (len(spikes) != 0) & (hasattr(trials, 'stimOn_times')):
        probes = one.load_object(eid, 'probes', download_only=False)
        for p in range(len(probes['trajectory'])):
            probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
            try:
                spikes = alf.io.load_object(probe_path, object='spikes')
                clusters = alf.io.load_object(probe_path, object='clusters')
            except Exception:
                continue

            # Only use single units
            spikes.times = spikes.times[np.isin(
                    spikes.clusters, clusters.metrics.cluster_id[
                        clusters.metrics.ks2_label == 'good'])]
            spikes.clusters = spikes.clusters[np.isin(
                    spikes.clusters, clusters.metrics.cluster_id[
                        clusters.metrics.ks2_label == 'good'])]

            # Get session info
            nickname = ses_info[i]['subject']
            ses_date = ses_info[i]['start_time'][:10]

            # Get trial indices
            r_in_l_block = trials.stimOn_times[((trials.probabilityLeft > 0.55)
                                                & (trials.contrastRight > 0.1))]
            r_in_r_block = trials.stimOn_times[((trials.probabilityLeft < 0.45)
                                                & (trials.contrastRight > 0.1))]
            l_in_r_block = trials.stimOn_times[((trials.probabilityLeft < 0.45)
                                                & (trials.contrastLeft > 0.1))]
            l_in_l_block = trials.stimOn_times[((trials.probabilityLeft > 0.55)
                                                & (trials.contrastLeft > 0.1))]

            # Get significant units
            r_units = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                  np.append(r_in_l_block,
                                                            r_in_r_block),
                                                  np.append(np.zeros(len(r_in_l_block)),
                                                            np.ones(len(r_in_r_block))),
                                                  pre_time=0, post_time=0.5,
                                                  test='ranksums', alpha=0.05)[0]
            l_units = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                  np.append(l_in_l_block,
                                                            l_in_r_block),
                                                  np.append(np.zeros(len(l_in_l_block)),
                                                            np.ones(len(l_in_r_block))),
                                                  pre_time=0, post_time=0.5,
                                                  test='ranksums', alpha=0.05)[0]
            sig_units = np.unique(np.concatenate((l_units, r_units)))
            print('%d out of %d neurons are responsive to contra stim' % (
                                    sig_units.shape[0], len(np.unique(spikes.clusters))))
            resp = resp.append(pd.DataFrame(index=[0],
                                            data={'subject': nickname,
                                                  'date': ses_date,
                                                  'eid': eid,
                                                  'n_neurons': len(np.unique(spikes.clusters)),
                                                  'contra': (sig_units.shape[0]
                                                             / len(np.unique(spikes.clusters))),
                                                  'ML': probes.trajectory[p]['x'],
                                                  'AP': probes.trajectory[p]['y'],
                                                  'DV': probes.trajectory[p]['z'],
                                                  'phi': probes.trajectory[p]['phi'],
                                                  'theta': probes.trajectory[p]['theta'],
                                                  'depth': probes.trajectory[p]['depth']}))
resp.to_csv(join(DATA_PATH, 'contra_stim_units_map'))

# %% Plot

Y_LIM = [-6000, 5000]
X_LIM = [-5000, 4000]

fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=(10, 8))
plot_h = sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='contra',
                         palette='YlOrRd', sizes=(100, 300), hue_norm=(0, 0.05), ax=ax1)
ax1.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Choice')

# Fix legend
ax1.legend(loc=(1.05, 0.5))

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'contra_stim_responsive_unit_map'))
