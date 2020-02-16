import numpy as np
import pandas as pd
from sklearn.manifold import Isomap
import brainbox as bb
from oneibl import one
from brainbox.core import TimeSeries
from cuml.manifold import UMAP, TSNE
from cuml import PCA
import plotly.express as px
import plotly.io as pio
# import plotly.graph_objects as go

one = one.ONE()


def dim_red(sessid, filename, binsize=0.15, surpwind=0.6, method='tnse'):
    """Reduce dimension of all trials in a session using the specified method, then save the
    plot of the embeddings in a plotly html file specified. Can use umap, tsne, pca, or Isomap.
    Requires Nvidia RAPIDS framework installed along with rapids cuML, a library for GPU
    accelerated data analysis. Isomap is not in the cuML library so will be extremely slow to run
    for small binsizes or spike arrays with more than 10k data entries.

    Parameters
    ----------
    sessid : str
        Sesssion UUID taken from datajoint or ONE
    filename : str
        filename of HTML file to save. Should end in .html
    binsize : float, optional
        Width of the non-overlapping temporal bins for spikes, by default 0.15
    surpwind : float, optional
        Window about stimulus onset for which to apply 'surprise' or
        'unsurprise' labels to spike bins, by default 0.6
    method : str, optional
        Dimensionality reduction method, valid options are 'tsne', 'umap', 'PCA', or
        'isomap', by default 'tnse'
    """
    try:
        spikes, clus = one.load(sessid, ['spikes.times', 'spikes.clusters'])
    except ValueError:
        raise RuntimeError('Session has two probes which have not been merged into a file, '
                           'Please try another session.')
    trialdata = one.load_object(sessid, 'trials')

    spikeseries = TimeSeries(spikes, clus, columns=['clusters'])
    stimons = TimeSeries(trialdata.stimOn_times, np.vstack([trialdata.contrastLeft,
                                                            trialdata.contrastRight,
                                                            trialdata.probabilityLeft]).T,
                         columns=['contrastLeft', 'contrastRight', 'pLeft'])

    with np.errstate(invalid='ignore'):
        surprises = stimons.times[((stimons.contrastRight > 0) & (stimons.pLeft > 0.5) |
                                  (stimons.contrastLeft > 0) & (stimons.pLeft < 0.5))]
        endsurprises = surprises + surpwind
        surprises = np.vstack([surprises, endsurprises]).T
        unsurprises = stimons.times[((stimons.contrastRight > 0) & (stimons.pLeft < 0.5) |
                                    (stimons.contrastLeft > 0) & (stimons.pLeft > 0.5))]
        endunsurprises = unsurprises + surpwind
    unsurprises = np.vstack([unsurprises, endunsurprises]).T
    binnedspikes = bb.processing.bin_spikes(spikeseries, binsize)

    surp_steps = np.vstack((np.ones(surprises.shape[0]), np.zeros(surprises.shape[0]))).T
    unsurp_steps = np.vstack((2 * np.ones(unsurprises.shape[0]), np.zeros(unsurprises.shape[0]))).T

    mergesurp = np.concatenate((surprises, unsurprises), axis=0)
    mergesteps = np.concatenate((surp_steps, unsurp_steps), axis=0)
    sortinds = mergesurp[:, 0].argsort()
    allsurp = mergesurp[sortinds]
    allstep = mergesteps[sortinds]

    surpseries = TimeSeries(allsurp.flatten(), allstep.flatten(), columns=['surprise'])
    surpdata = bb.processing.sync(binsize, timeseries=(binnedspikes, surpseries),
                                  interp='previous', fillval='extrapolate')
    popnorms = np.linalg.norm(surpdata.values[:, :-1], axis=1)
    normfilter = popnorms < np.percentile(popnorms, 99)  # Exclude very high pop rates (artifacts?)

    if method == 'umap':
        embeddings = UMAP(n_components=3).fit_transform(surpdata.values[normfilter, :-1])
    elif method == 'PCA':
        embeddings = PCA(n_components=3).fit_transform(surpdata.values[normfilter, :-1])
        embeddings = embeddings.as_matrix()
    elif method == 'isomap':
        embeddings = Isomap(n_components=3).fit_transform(surpdata.values[normfilter, :-1])
    elif method == 'tsne':
        embeddings = TSNE(n_components=3).fit_transform(surpdata.values[normfilter, :-1])

    labeledembed = np.column_stack((embeddings,
                                    surpdata.surprise[normfilter]))
    catnames = {0: 'No stimulus', 1: 'Surprise', 2: 'No surprise'}
    cmap = {'No stimulus': 'rgb(178, 34, 34)',
            'Surprise': 'rgb(255, 129, 0)',
            'No surprise': 'rgb(0, 0, 128)'}
    plotdf = pd.DataFrame(data=labeledembed, columns=['x', 'y', 'z', 'surprise'])
    plotdf.surprise.replace(catnames, inplace=True)
    fig = px.scatter_3d(plotdf, x='x', y='y', z='z', color='surprise', opacity=0.3,
                        color_discrete_map=cmap)
    # fig = go.Figure(data=go.Scatter3d(x=plotdf.x, y=plotdf.y, z=plotdf.z,
    #                                   mode='markers',
    #                                   marker=dict(size=10,
    #                                               color=pointcolors,
    #                                               line=dict(width=0, color='rgba(0, 0, 0, 0)'),
    #                                   opacity=0.99)))
    pio.write_html(fig, filename)
