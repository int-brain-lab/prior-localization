# Standard library
from collections import defaultdict

# Third party libraries
import numpy as np
import pandas as pd
import xarray as xr

# IBL libraries
from ibllib.atlas import BrainRegions

# Brainwide repo imports
from brainwide.utils import get_id, remap


def generate_da_dict(
        filename,
        n_impostors,
        n_folds,
        region_mapper=lambda x: x,
        temporal_kernels=['wheel', 'stimonL', 'stimonR', 'fmove', 'correct', 'incorrect'],
        gain_kernels=['pLeft', 'pLeft_tr']):
    datafile = pd.read_pickle(filename)
    eid = filename.parts[-2]  # Ugly hack because I forgot to keep eid value in files
    darrays = process_temporal_kernels(datafile, eid, temporal_kernels, n_impostors, n_folds,
                                       region_mapper)

    darrays_gain = process_gain_kernels(datafile, eid, gain_kernels, n_impostors, n_folds,
                                        region_mapper)
    null_dist = {**darrays, **darrays_gain}

    return null_dist


def process_gain_kernels(datafile, eid, kern_names, n_impostors, n_folds, reg_map):
    darrays = defaultdict(list)
    for kernel in kern_names:
        for i in range(-1, n_impostors):
            fold_da = []
            for j in range(n_folds):
                if i < 0:  # this references the actual data fit
                    df = datafile['fitdata']['weights'][j][kernel].copy()
                else:
                    df = datafile['nullfits'][i]['weights'][j][kernel].copy()
                mi = make_multiindex(datafile, eid, reg_map, df)
                df.index = mi
                tmp_da = xr.DataArray(
                    df.values,
                    dims=['unit'],
                    coords={'unit': df.index},
                )
                fold_da.append(tmp_da)
            tmp_da = xr.concat(fold_da, pd.Index(np.arange(n_folds), name='fold'))
            darrays[kernel].append(tmp_da)
    darrays_cat = {
        k: xr.concat(darrays[k], pd.Index(np.arange(-1, n_impostors), name='run')) for k in darrays
    }
    return darrays_cat


def process_temporal_kernels(datafile, eid, kern_names, n_impostors, n_folds, reg_map):
    darrays = defaultdict(list)
    for kernel in kern_names:
        for i in range(-1, n_impostors):
            fold_da = []
            for j in range(n_folds):
                if i < 0:  # this references the actual data fit
                    df = datafile['fitdata']['weights'][j][kernel].copy()
                else:
                    df = datafile['nullfits'][i]['weights'][j][kernel].copy()
                mi = make_multiindex(datafile, eid, reg_map, df)
                df.index = mi
                tmp_da = xr.DataArray(
                    df.values,
                    dims=['unit', 'time'],
                    coords={
                        'unit': df.index,
                        'time': df.columns
                    },
                )
                fold_da.append(tmp_da)
            tmp_da = xr.concat(fold_da, pd.Index(np.arange(n_folds), name='fold'))
            darrays[kernel].append(tmp_da)
    darrays_cat = {
        k: xr.concat(darrays[k], pd.Index(np.arange(-1, n_impostors), name='run')) for k in darrays
    }
    return darrays_cat


def make_multiindex(datafile, eid, reg_map, df):
    mi = pd.MultiIndex.from_arrays(
        [
            np.array([eid] * len(df)),
            df.index,
            reg_map(datafile['clu_regions'][df.index]),
        ],
        names=['eid', 'clu_id', 'region'],
    )

    return mi


if __name__ == '__main__':
    # Standard library
    import os
    import pickle
    from pathlib import Path

    # Brainwide repo imports
    from brainwide.params import GLM_CACHE, GLM_FIT_PATH

    FITDATE = '2022-02-21'  # Date on which fit was run

    parpath = Path(GLM_FIT_PATH).joinpath(f'{FITDATE}_glm_fit_pars.pkl')
    with open(parpath, 'rb') as fo:
        params = pickle.load(fo)
    datapath = Path(GLM_CACHE).joinpath(params['dataset_fn'])
    with open(datapath, 'rb') as fo:
        dataset = pickle.load(fo)

    n_folds = params['n_folds'] if 'n_folds' in params else 5
    n_impostors = params['n_impostors']

    filenames = []
    for subj in os.listdir(Path(GLM_FIT_PATH)):
        subjdir = Path(GLM_FIT_PATH).joinpath(subj)
        if not os.path.isdir(subjdir):
            continue
        for sess in os.listdir(subjdir):
            sessdir = subjdir.joinpath(sess)
            for file in os.listdir(sessdir):
                filepath = sessdir.joinpath(file)
                if os.path.isfile(filepath) and filepath.match(f'*{FITDATE}*impostor*'):
                    filenames.append(filepath)

    br = BrainRegions()

    def regmap(acr):
        ids = get_id(acr)
        return remap(ids, br=br)

    fdata = defaultdict(list)
    for filename in filenames:
        darrays = generate_da_dict(filename, n_impostors, n_folds, regmap)
        for k in darrays:
            fdata[k].extend(darrays[k])
    fdata['params'] = params
    fdata['dataset'] = dataset

    with open(Path(GLM_FIT_PATH).joinpath(f'{FITDATE}_impostor_fit.pkl'), 'wb') as fw:
        pickle.dump(fdata, fw)
