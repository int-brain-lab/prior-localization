# Standard library
import hashlib
import logging
import os
import pickle
import re
from datetime import datetime as dt
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd

# IBL libraries
import brainbox.io.one as bbone
import brainbox.metrics.single_units as bbqc
from one.api import ONE

# Brainwide repo imports
from brainwide.params import CACHE_PATH

_logger = logging.getLogger('brainwide')


def load_primaries(session_id,
                   probes,
                   max_len=2.,
                   t_before=0.,
                   t_after=0.,
                   binwidth=0.02,
                   abswheel=False,
                   ret_qc=False,
                   one=None):
    one = ONE() if one is None else one

    trialsdf = bbone.load_trials_df(session_id,
                                    maxlen=max_len,
                                    t_before=t_before,
                                    t_after=t_after,
                                    wheel_binsize=binwidth,
                                    ret_abswheel=abswheel,
                                    ret_wheel=not abswheel,
                                    addtl_types=['firstMovement_times'],
                                    one=one)

    spikes, clusters, cludfs = {}, {}, []
    clumax = 0
    for pid in probes:
        ssl = bbone.SpikeSortingLoader(one=one, pid=pid)
        spikes[pid], tmpclu, channels = ssl.load_spike_sorting()
        if 'metrics' not in tmpclu:
            tmpclu['metrics'] = np.ones(tmpclu['channels'].size)
        clusters[pid] = ssl.merge_clusters(spikes[pid], tmpclu, channels)
        clusters_df = pd.DataFrame(clusters[pid]).set_index(['cluster_id'])
        clusters_df.index += clumax
        clusters_df['pid'] = pid
        cludfs.append(clusters_df)
        clumax = clusters_df.index.max()
    allcludf = pd.concat(cludfs)

    allspikes, allclu, allreg, allamps, alldepths = [], [], [], [], []
    clumax = 0
    for pid in probes:
        allspikes.append(spikes[pid].times)
        allclu.append(spikes[pid].clusters + clumax)
        allreg.append(clusters[pid].acronym)
        allamps.append(spikes[pid].amps)
        alldepths.append(spikes[pid].depths)
        clumax += np.max(spikes[pid].clusters) + 1

    allspikes, allclu, allamps, alldepths = [
        np.hstack(x) for x in (allspikes, allclu, allamps, alldepths)
    ]
    sortinds = np.argsort(allspikes)
    spk_times = allspikes[sortinds]
    spk_clu = allclu[sortinds]
    spk_amps = allamps[sortinds]
    spk_depths = alldepths[sortinds]
    clu_regions = np.hstack(allreg)
    if not ret_qc:
        return trialsdf, spk_times, spk_clu, clu_regions, allcludf

    clu_qc = bbqc.quick_unit_metrics(spk_clu,
                                     spk_times,
                                     spk_amps,
                                     spk_depths,
                                     cluster_ids=np.arange(clu_regions.size))
    return trialsdf, spk_times, spk_clu, clu_regions, clu_qc, allcludf


def cache_primaries(subject, session_id, probes, regressor_params, trialsdf, spk_times, spk_clu,
                    clu_regions, clu_qc, clu_df):
    """
    Take outputs of load_primaries() and cache them to disk in the folder defined in the params.py
    file in this repository, using a nested subject -> session folder structure.

    If an existing file in the directory already contains identical data, will not write a new file
    and instead return the existing filenames.

    Returns the metadata filename and regressors filename.
    """
    subpath = Path(CACHE_PATH).joinpath(subject)
    if not subpath.exists():
        os.mkdir(subpath)
    sesspath = subpath.joinpath(session_id)
    if not sesspath.exists():
        os.mkdir(sesspath)
    curr_t = dt.now()
    fnbase = str(curr_t.date())
    metadata_fn = sesspath.joinpath(fnbase + '_metadata.pkl')
    data_fn = sesspath.joinpath(fnbase + '_regressors.pkl')
    regressors = {
        'trialsdf': trialsdf,
        'spk_times': spk_times,
        'spk_clu': spk_clu,
        'clu_regions': clu_regions,
        'clu_qc': clu_qc,
        'clu_df': clu_df,
    }
    reghash = _hash_dict(regressors)
    metadata = {
        'subject': subject,
        'session_id': session_id,
        'probes': probes,
        'regressor_hash': reghash,
        **regressor_params
    }
    prevdata = [
        sesspath.joinpath(f) for f in os.listdir(sesspath) if re.match(r'.*_metadata\.pkl', f)
    ]
    matchfile = False
    for f in prevdata:
        with open(f, 'rb') as fr:
            frdata = pickle.load(fr)
            if metadata == frdata:
                matchfile = True
        if matchfile:
            _logger.info(f'Existing cache file found for {subject}: {session_id}, '
                         'not writing data.')
            old_data_fn = sesspath.joinpath(f.name.split('_')[0] + '_regressors.pkl')
            return f, old_data_fn
    # If you've reached here, there's no matching file
    with open(metadata_fn, 'wb') as fw:
        pickle.dump(metadata, fw)
    with open(data_fn, 'wb') as fw:
        pickle.dump(regressors, fw)
    return metadata_fn, data_fn


def _hash_dict(d):
    hasher = hashlib.md5()
    sortkeys = sorted(d.keys())
    for k in sortkeys:
        v = d[k]
        if type(v) == np.ndarray:
            hasher.update(v)
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            hasher.update(v.to_string().encode())
        else:
            try:
                hasher.update(v)
            except Exception:
                _logger.warning(f'Key {k} was not able to be hashed. May lead to failure to update'
                                ' in cached files if something was changed.')
    return hasher.hexdigest()

