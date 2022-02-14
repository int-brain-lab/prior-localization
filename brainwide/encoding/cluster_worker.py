"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

# Standard library
import argparse
import os
import pickle
from pathlib import Path

# Third party libraries
import numpy as np

# Brainwide repo imports
from brainwide.decoding.functions.utils import compute_target
from brainwide.encoding.design import generate_design
from brainwide.encoding.fit import fit_stepwise
from brainwide.params import BEH_MOD_PATH, GLM_FIT_PATH


def filter_nan(trialsdf):
    target_cols = ['stimOn_times', 'feedback_times', 'firstMovement_times']
    mask = ~np.any(np.isnan(trialsdf[target_cols]), axis=1)
    return trialsdf[mask]


def get_cached_regressors(fpath):
    with open(fpath, 'rb') as fo:
        d = pickle.load(fo)
    return d['trialsdf'], d['spk_times'], d['spk_clu'], d['clu_regions'], d['clu_qc']


def _create_sub_sess_path(parent, subject, session):
    subpath = Path(parent).joinpath(subject)
    if not subpath.exists():
        os.mkdir(subpath)
    sesspath = subpath.joinpath(session)
    if not sesspath.exists():
        os.mkdir(sesspath)
    return sesspath


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


def fit_save_inputs(subject,
                    eid,
                    probes,
                    eidfn,
                    subjeids,
                    params,
                    t_before,
                    fitdate,
                    prior_estimate=False):
    stdf, sspkt, sspkclu, sclureg, scluqc = get_cached_regressors(eidfn)
    stdf_nona = filter_nan(stdf)
    if prior_estimate:
        sessfullprior = compute_target('pLeft', subject, subjeids, eid, Path(BEH_MOD_PATH))
        sessprior = sessfullprior[stdf_nona.index]
    else:
        sessprior = stdf_nona['probablityLeft']
    sessdesign = generate_design(stdf_nona, sessprior, t_before, **params)
    sessfit = fit_stepwise(sessdesign, sspkt, sspkclu, **params)
    outputfn = save_stepwise(subject, eid, sessfit, params, probes, eidfn, sclureg, scluqc,
                             fitdate)
    return outputfn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster GLM fitter')
    parser.add_argument('datafile',
                        type=Path,
                        help='Input file (parquet pandas df) \
                        containing inputs to each worker')
    parser.add_argument('paramsfile', type=Path, help='Parameters for model fitting for worker')
    parser.add_argument('index',
                        type=int,
                        help='Index in inputfile for this worker to '
                        'process/save')
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
                               args.fitdate, prior_estimate=params['prior_estimate'])
    print('Fitting completed successfully!')
    print(outputfn)
