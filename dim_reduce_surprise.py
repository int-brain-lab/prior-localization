import numpy as np
import pandas as pd
from sklearn.manifold import Isomap
import brainbox as bb
from oneibl import one
from brainbox.core import TimeSeries
from cuml.manifold import UMAP
from cuml import PCA
import plotly.express as px
import plotly.io as pio

one = one.ONE()

BINSIZE = 0.15
SURPRISE_WINDOW = 0.6
METHOD = 'umap'

ids = one.search(subject='ZM_2240', dataset_types=['spikes.clusters'],
                 date_range=['2020-01-21', '2020-01-30'])
spikes, clus = one.load(ids[3], ['spikes.times', 'spikes.clusters'])
trialdata = one.load_object(ids[3], 'trials')

spikeseries = TimeSeries(spikes, clus, columns=['clusters'])
stimons = TimeSeries(trialdata.stimOn_times, np.vstack([trialdata.contrastLeft,
                                                        trialdata.contrastRight,
                                                        trialdata.probabilityLeft]).T,
                     columns=['contrastLeft', 'contrastRight', 'pLeft'])

surprises = stimons.times[((stimons.contrastRight > 0) & (stimons.pLeft > 0.5) |
                           (stimons.contrastLeft > 0) & (stimons.pLeft < 0.5))]
endsurprises = surprises + SURPRISE_WINDOW
surprises = np.vstack([surprises, endsurprises]).T
unsurprises = stimons.times[((stimons.contrastRight > 0) & (stimons.pLeft < 0.5) |
                             (stimons.contrastLeft > 0) & (stimons.pLeft > 0.5))]
endunsurprises = unsurprises + SURPRISE_WINDOW
unsurprises = np.vstack([unsurprises, endunsurprises]).T
binnedspikes = bb.processing.bin_spikes(spikeseries, BINSIZE)

surp_steps = np.vstack((np.ones(surprises.shape[0]), np.zeros(surprises.shape[0]))).T
unsurp_steps = np.vstack((-np.ones(unsurprises.shape[0]), np.zeros(unsurprises.shape[0]))).T

mergesurp = np.concatenate((surprises, unsurprises), axis=0)
mergesteps = np.concatenate((surp_steps, unsurp_steps), axis=0)
sortinds = mergesurp[:, 0].argsort()
allsurp = mergesurp[sortinds]
allstep = mergesteps[sortinds]

surpseries = TimeSeries(allsurp.flatten(), allstep.flatten(), columns=['surprise'])
surprisedata = bb.processing.sync(BINSIZE, timeseries=(binnedspikes, surpseries),
                                  interp='previous', fillval='extrapolate')
popnorms = np.linalg.norm(surprisedata.values[:, :-1], axis=1)
normfilter = popnorms < np.percentile(popnorms, 99)  # Exclude very high pop rates (artifacts?)

if METHOD is 'umap':
    embeddings = UMAP(n_components=3).fit_transform(surprisedata.values[normfilter, :-1])
elif METHOD is 'PCA':
    embeddings = PCA(n_components=3).fit_transform(surprisedata.values[normfilter, :-1]).as_matrix()

labeledembed = np.column_stack((embeddings,
                                surprisedata.surprise[normfilter]))
plotdf = pd.DataFrame(data=labeledembed, columns=['x', 'y', 'z', 'surprise'])
fig = px.scatter_3d(plotdf, x='x', y='y', z='z', color='surprise', opacity=0.5)
pio.write_html(fig, 'test.html')