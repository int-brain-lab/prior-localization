# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

from oneibl.one import ONE
from os.path import expanduser, join
import pandas as pd
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score, confusion_matrix


def paths():
    if expanduser('~') == '/home/guido':
        data_path = '/media/guido/data/Flatiron/'
    else:
        data_path = join(expanduser('~'), 'Downloads', 'FlatIron')
    fig_path = join(expanduser('~'), 'Figures', '5HT', 'ephys')
    save_path = join(expanduser('~'), 'Data', '5HT')
    return data_path, fig_path, save_path


def plot_settings():
    plt.tight_layout()
    sns.despine(trim=True)
    sns.set(style="ticks", context="paper", font_scale=1.4)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


def download_data(nickname, date):
    one = ONE()
    eid = one.search(subject=nickname, date_range=[date, date])
    assert len(eid) == 1
    dtypes = ['_iblrig_taskSettings.raw',
              'spikes.times',
              'spikes.clusters',
              'clusters.channels',
              'clusters.metrics',
              'clusters.depths',
              'clusters.probes',
              'probes.trajectory',
              'trials.choice',
              'trials.intervals',
              'trials.contrastLeft',
              'trials.contrastRight',
              'trials.feedback_times',
              'trials.goCue_times',
              'trials.feedbackType',
              'trials.probabilityLeft',
              'trials.response_times',
              'trials.stimOn_times']
    one.load(eid[0], dataset_types=dtypes, download_only=True)


def sessions():
    frontal_sessions = pd.DataFrame(data={'lab': ['mainenlab',
                                                  'mainenlab',
                                                  'mainenlab',
                                                  'danlab',
                                                  'mainenlab'],
                                          'subject': ['ZM_2240',
                                                      'ZM_2240',
                                                      'ZM_2240',
                                                      'DY_011',
                                                      'ZM_2241'],
                                          'date': ['2020-01-23',
                                                   '2020-01-22',
                                                   '2020-01-21',
                                                   '2020-01-30',
                                                   '2020-01-27'],
                                          'probe': ['00',
                                                    '00',
                                                    '00',
                                                    '00',
                                                    '00']})
    control_sessions = pd.DataFrame(data={'lab': ['mainenlab',
                                                  'mainenlab',
                                                  'mainenlab'],
                                          'subject': ['ZM_2240',
                                                      'ZM_2240',
                                                      'ZM_2241'],
                                          'date': ['2020-01-22',
                                                   '2020-01-24',
                                                   '2020-01-28'],
                                          'probe': ['01',
                                                    '00',
                                                    '00']})
    return frontal_sessions, control_sessions


def one_session_path(eid):
    one = ONE()
    ses = one.alyx.rest('sessions', 'read', id=eid)
    return Path(one._par.CACHE_DIR, ses['lab'], 'Subjects', ses['subject'],
                ses['start_time'][:10], str(ses['number']).zfill(3))


def decoding(resp, labels, clf, num_splits):
    """
    Parameters
    ----------
    resp : TxN matrix
        Neuronal responses of N neurons in T trials
    labels : 1D array
        Class labels for T trials
    clf : object
        sklearn decoder object
    NUM_SPLITS : int
        The n in n-fold cross validation

    Returns
    -------
    f1 : float
        The F1-score of the classification
    cm : 2D matrix
        The normalized confusion matrix

    """
    assert resp.shape[0] == labels.shape[0]

    kf = KFold(n_splits=num_splits, shuffle=True)
    y_pred = np.array([])
    y_true = np.array([])
    y_auroc = np.array([])
    for train_index, test_index in kf.split(resp):
        train_resp = resp[train_index]
        test_resp = resp[test_index]
        clf.fit(train_resp, [labels[j] for j in train_index])
        y_pred = np.append(y_pred, clf.predict(test_resp))
        y_true = np.append(y_true, [labels[j] for j in test_index])
        probs = clf.predict_proba(test_resp)
        probs = probs[:, 1]  # keep positive only
        y_auroc = np.append(y_auroc, roc_auc_score([labels[j] for j in test_index], probs))
    f1 = f1_score(y_true, y_pred)
    auroc = np.mean(y_auroc)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    return f1, auroc, cm


def get_spike_counts_in_bins(spike_times, spike_clusters, intervals):
    """
    Return the number of spikes in a sequence of time intervals, for each neuron.

    Parameters
    ----------
    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    intervals : 2D array of shape (n_events, 2)
        the start and end times of the events

    Returns
    ---------
    counts : 2D array of shape (n_neurons, n_events)
        the spike counts of all neurons ffrom scipy.stats import sem, tor all events
        value (i, j) is the number of spikes of neuron `neurons[i]` in interval #j
    cluster_ids : 1D array
        list of cluster ids
    """

    # Check input
    assert intervals.ndim == 2
    assert intervals.shape[1] == 2

    # For each neuron and each interval, the number of spikes in the interval.
    cluster_ids = np.unique(spike_clusters)
    n_neurons = len(cluster_ids)
    n_intervals = intervals.shape[0]
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    for j in range(n_intervals):
        t0, t1 = intervals[j, :]
        # Count the number of spikes in the window, for each neuron.
        x = np.bincount(
            spike_clusters[(t0 <= spike_times) & (spike_times < t1)],
            minlength=cluster_ids.max() + 1)
        counts[:, j] = x[cluster_ids]
    return counts, cluster_ids
