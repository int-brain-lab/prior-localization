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
from warnings import warn
# import plotly.graph_objects as go

one = one.ONE()


def dim_red(sessid, filename, binsize=0.15, surpwind=0.6, method='umap', probe_idx=None,
            depth_range=None):
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
        Dimensionality reduction method, valid options are 'umap', 'PCA', or
        'isomap', by default 'tnse'
    probe_idx : int, optional
        Which probe to use, if a session has multiple probes, by default None and will use probe 1.
    depth_range : list of ints, optional,
        Range of depths on the probe from which to accept spikes. By default set to None, but if
        set to a 2-element list or array will serve as the start and end depths.
    """
    try:
        spikes, clus, depths = one.load(sessid, ['spikes.times',
                                                 'spikes.clusters',
                                                 'spikes.depths'])
    except ValueError:
        if probe_idx is None:
            warn('Session has two probes. Defaulting to first probe. If a specific probe is '
                 'desired, pass int as argument to probe_idx')
            probe_idx = 0
        spikes = one.load(sessid, ['spikes.times'])[probe_idx]
        clus = one.load(sessid, ['spikes.clusters'])[probe_idx]
        depths = one.load(sessid, ['spikes.depths'])[probe_idx]

    if depth_range is not None:
        filt = (depths <= depth_range[1]) & (depths >= depth_range[0])
        spikes = spikes[filt]
        clus = clus[filt]
        depths = depths[filt]

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


if __name__ == "__main__":
    from ibl_pipeline import subject, ephys
    METHOD = 'tsne'
    sessions = subject.Subject * subject.SubjectProject *\
        ephys.acquisition.Session * ephys.ProbeTrajectory()
    bwm_sess = sessions & 'subject_project = "ibl_neuropixel_brainwide_01"' & \
        'task_protocol = "_iblrig_tasks_ephysChoiceWorld6.2.5"'
    for s in bwm_sess:
        sess_id = str(s['session_uuid'])
        filename = s['subject_nickname'] + '_' + str(s['session_start_time'].date()) +\
            METHOD + '.html'
        fullpath = './data/' + filename
        try:
            dim_red(sess_id, fullpath, surpwind=0.8, method=METHOD, probe_idx=0)
            print(filename + ' session succeeded')
        except TypeError:
            print(filename + ' session failed because no spike data.')
        except ValueError:
            print(filename + ' session failed. Possible mismatch of probe indices in ONE.')
