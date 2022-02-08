"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

import pickle
import os
import sys
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from brainbox.task.closed_loop import generate_pseudo_session
from brainwide_models.encoding.design import fit_stepwise, generate_design
from params import BEH_MOD_PATH, GLM_FIT_PATH, GLM_CACHE
sys.path.append(Path(__file__).joinpath('decoding'))
from decoding.functions.utils import compute_target, query_sessions  # noqa
from ibllib.atlas import BrainRegions
from iblutil.numerical import ismember


def filter_nan(trialsdf):
    target_cols = ['stimOn_times', 'feedback_times', 'firstMovement_times']
    mask = ~np.any(np.isnan(trialsdf[target_cols]), axis=1)
    return trialsdf[mask]


def get_cached_regressors(fpath):
    with open(fpath, 'rb') as fo:
        d = pickle.load(fo)
    return d['trialsdf'], d['spk_times'], d['spk_clu'], d['clu_regions'], d['clu_qc']


def save_pseudo(subject, session_id, fitout, params):
    # TODO: Make this work
    raise NotImplementedError
    # sesspath = _create_sub_sess_path(GLM_FIT_PATH, subject, session_id)
    # fn = sesspath.joinpath()
    # subjfilepath = os.path.abspath(filename)
    # fw = open(subjfilepath, 'wb')
    # pickle.dump(outdict, fw)
    # fw.close()
    # return True


def save_stepwise(subject, session_id, fitout, params, probes, input_fn, clu_reg, clu_qc, fitdate):
    sesspath = _create_sub_sess_path(GLM_FIT_PATH, subject, session_id)
    fn = sesspath.joinpath(f'{fitdate}_stepwise_regression.pkl')
    outdict = {
        'params': params,
        'probes': probes,
        'model_input_fn': input_fn,
        'clu_regions': clu_reg,
        'clu_qc': clu_qc,
    }
    outdict.update(fitout)
    with open(fn, 'wb') as fw:
        pickle.dump(outdict, fw)
    return fn


def compute_deltas(scores):
    outdf = pd.DataFrame(np.zeros_like(scores), index=scores.index, columns=scores.columns)
    for i in scores.columns:  # Change this for diff num covs
        if i >= 1:
            diff = scores[i] - scores[i - 1]
        else:
            diff = scores[i]
        outdf [i] = diff
    return outdf


def colrename(cname, suffix):
    return str(cname + 1) + 'cov' + suffix


def _create_sub_sess_path(parent, subject, session):
    subpath = Path(parent).joinpath(subject)
    if not subpath.exists():
        os.mkdir(subpath)
    sesspath = subpath.joinpath(session)
    if not sesspath.exists():
        os.mkdir(sesspath)
    return sesspath


def remap(ids, source='Allen', dest='Beryl', output='acronym', br=BrainRegions()):
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == 'id':
        return br.id[br.mappings[dest][inds]]
    elif output == 'acronym':
        return br.get(br.id[br.mappings[dest][inds]])['acronym']


def get_id(acronym, brainregions=BrainRegions()):
    return brainregions.id[np.argwhere(brainregions.acronym == acronym)[0, 0]]


def get_name(acronym, brainregions=BrainRegions()):
    if acronym == 'void':
        return acronym
    reg_idxs = np.argwhere(brainregions.acronym == acronym).flat
    return brainregions.name[reg_idxs[0]]


def label_cerebellum(acronym, brainregions=BrainRegions()):
    regid = brainregions.id[np.argwhere(brainregions.acronym == acronym).flat][0]
    ancestors = brainregions.ancestors(regid)
    if 'Cerebellum' in ancestors.name or 'Medulla' in ancestors.name:
        return True
    else:
        return False


import argparse
import pickle
import bbglm_cluster as bc
from pathlib import Path
from params import BEH_MOD_PATH


def fit_save_inputs(subject, eid, probes, eidfn, subjeids, params, t_before, fitdate):
    stdf, sspkt, sspkclu, sclureg, scluqc = bc.get_cached_regressors(eidfn)
    stdf_nona = bc.filter_nan(stdf)
    sessfullprior = bc.compute_target('pLeft', subject, subjeids, eid, Path(BEH_MOD_PATH))
    sessprior = sessfullprior[stdf_nona.index]
    sessdesign = bc.generate_design(stdf_nona, sessprior, t_before, **params)
    sessfit = bc.fit_stepwise(sessdesign, sspkt, sspkclu, **params)
    outputfn = bc.save_stepwise(subject, eid, sessfit, params, probes, eidfn, sclureg, scluqc,
                                fitdate)
    return outputfn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster GLM fitter')
    parser.add_argument('datafile', type=Path, help='Input file (parquet pandas df) \
                        containing inputs to each worker')
    parser.add_argument('paramsfile', type=Path, help='Parameters for model fitting for worker')
    parser.add_argument('index', type=int, help='Index in inputfile for this worker to \
                        process/save')
    parser.add_argument('fitdate', help='Date of fit for output file')
    args = parser.parse_args()

    with open(args.datafile, 'rb') as fo:
        dataset = pickle.load(fo)
    with open(args.paramsfile, 'rb') as fo:
        params = pickle.load(fo)
    t_before = dataset['params']['t_before']
    dataset_fns = dataset['dataset_filenames']

    subject, eid, probes, metafn, eidfn = dataset_fns.loc[args.index]
    subjeids = list(dataset_fns[dataset_fns.subject == subject].eid.unique())

    outputfn = fit_save_inputs(subject, eid, probes, eidfn, subjeids, params, t_before,
                               args.fitdate)
    print('Fitting completed successfully!')
    print(outputfn)