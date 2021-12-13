import numpy as np
import pandas as pd
import brainbox.io.one as bbone
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate, correlation_lags
from sklearn.preprocessing import Normalizer
from brainbox.processing import bincount2D
from one.api import ONE

# which sesssion and probe to look at, bin size
eid = '5157810e-0fff-4bcf-b19d-32d4e39c7dfc'
probe = 'probe00'
binwidth = 0.02
corr_wind = (-0.2, 0.8)  # seconds

# Do some data loading
one = ONE()

spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, probe=probe, one=one)
trialsdf = bbone.load_trials_df(eid, one=one, addtl_types=['firstMovement_times'])

# Get information about the details of our session such as start time etc
t_start = 0
t_end = trialsdf['trial_end'].max()

events = {
    'leftstim': trialsdf[trialsdf.contrastLeft.notna()].stimOn_times,
    'rightstim': trialsdf[trialsdf.contrastRight.notna()].stimOn_times,
    'gocue': trialsdf.goCue_times,
    'movement': trialsdf.firstMovement_times,
    'correct': trialsdf[trialsdf.feedbackType == 1].feedback_times,
    'incorrect': trialsdf[trialsdf.feedbackType == -1].feedback_times,
}


# Build a basic vector to work with and also bin spikes
def binf(t):
    return np.ceil(t / binwidth).astype(int)


vecshape = binf(t_end + binwidth) - binf(t_start)
tmask = spikes[probe].times < t_end  # Only get spikes in interval
binned = bincount2D(spikes[probe].times[tmask], spikes[probe].clusters[tmask],
                    xlim=[t_start, t_end],
                    xbin=binwidth)[0]
n_corr = binf(corr_wind[1] - corr_wind[0])  # number of actual observations we're going to store
lags = correlation_lags(vecshape, binned.shape[1]) * binwidth  # Value of correlation lags
# Where to start and end
start, end = np.searchsorted(lags, corr_wind[0]), np.searchsorted(lags, corr_wind[1]) + 1
lagvals = lags[start:end]  # Per-step values of the lag


# Iterate through each timing event for regression
eventcorrs = {}
for name, event in events.items():
    evec = np.zeros(vecshape)
    evinds = event.dropna().apply(binf)
    evec[evinds] = 1
    corrarr = np.zeros((binned.shape[0], end - start))
    for i in range(binned.shape[0]):
        corrarr[i] = correlate(evec, binned[i])[start:end]
    eventcorrs[name] = corrarr

normer = Normalizer()
for event, corrarr in eventcorrs.items():
    keepinds = np.argwhere(np.max(corrarr, axis=1) >= 10).flatten()
    normarr = normer.transform(corrarr[keepinds])
    sortinds = np.argsort(normarr.argmax(axis=1))
    sns.heatmap(pd.DataFrame(normarr[sortinds], columns=lagvals))
    plt.title(' '.join([event, 'cross-correlations']))
    plt.figure()
