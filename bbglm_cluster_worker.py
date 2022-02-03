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
    parser.add_argument('datafile', dtype=Path, help='Input file (parquet pandas df) \
                        containing inputs to each worker')
    parser.add_argument('paramsfile', dtype=Path, help='Parameters for model fitting for worker')
    parser.add_argument('index', dtype=int, help='Index in inputfile for this worker to \
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
