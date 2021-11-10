import os
import pickle
import numpy as np
import pandas as pd
import brainbox.behavior.pyschofit as pfit
import brainbox.io.one as bbone
from pathlib import Path
from one.api import ONE
from tqdm import tqdm
from scipy.stats import zscore, percentileofscore


def get_target_df(target, pred, test_idxs, trialsdf, one):
    """
    Get two arrays needed for the psychofit method for fitting an erf.

    Parameters
    ----------
    eid : str
        uuid of session
    target : numpy.ndarray
        vector of all targets for the given session
    pred : numpy.ndarray
        vector of all predictions for the given session and region
    test_idxs : numpy.ndarray
        indices of the used in the test set and scoring computation

    Returns
    -------
    tuple of numpy.ndarrays
        two 3 x 9 arrays, the first of which is for P(Left) = 0.2 ,the latter for 0.8, with
        the specifications required for mle_fit_psycho.
    """
    offset = 1 - target.max()
    test_target = target[test_idxs]
    test_pred = pred[test_idxs]
    corr_test = test_target + offset
    corr_pred = test_pred + offset
    pred_signs = np.sign(corr_pred)
    df = pd.DataFrame({'stimuli': corr_test,
                       'predictions': corr_pred,
                       'sign': pred_signs,
                       'blockprob': trialsdf.loc[test_idxs, 'probabilityLeft']},
                      index=test_idxs)
    grpby = df.groupby(['blockprob', 'stimuli'])
    grpbyagg = grpby.agg({'sign': [('num_trials', 'count'),
                                   ('prop_L', lambda x: (x == 1).sum() / len(x))]})
    return grpbyagg.loc[0.2].reset_index().values.T, grpbyagg.loc[0.8].reset_index().values.T


def fit_get_shift_range(lowprob_arr, highprob_arr):
    """
    Fit psychometric functions with erf with two gammas for lapse rate, and return the parameters,
    traces, and the slope and shift.

    Parameters
    ----------
    lowprobarr : numpy.ndarray
        3 x 9 array for function fitting (see get_target_df), from low probability left block
    highprob_arr : numpy.ndarray
        Same as above, for high probability Left
    """
    low_pars, low_L = pfit.mle_fit_psycho(lowprob_arr, P_model='erf_psycho_2gammas')
    low_fit_trace = pfit.erf_psycho_2gammas(low_pars, lowprob_arr[0, :])
    low_slope = low_pars[1]
    high_pars, high_L = pfit.mle_fit_psycho(highprob_arr, P_model='erf_psycho_2gammas')
    high_fit_trace = pfit.erf_psycho_2gammas(high_pars, highprob_arr[0, :])
    high_slope = high_pars[1]
    high_zerind = np.argwhere(np.isclose(highprob_arr[0, :], 0)).flat[0]
    low_zerind = np.argwhere(np.isclose(lowprob_arr[0, :], 0)).flat[0]
    shift = high_fit_trace[high_zerind] - low_fit_trace[low_zerind]
    params = {'low_pars': low_pars, 'low_likelihood': low_L,
              'high_pars': high_pars, 'high_likelihood': high_L,
              'low_fit_trace': low_fit_trace, 'high_fit_trace': high_fit_trace}
    return params, low_slope, high_slope, shift


def fit_file(file, overwrite=False):
    one = ONE()
    filedata = np.load(file, allow_pickle=True)
    parentfolder = file.parent
    outparts = list(file.name.partition('.'))
    outparts.insert(1, '_neurometric_fits')
    outfn = parentfolder.joinpath(''.join(outparts))
    if not overwrite:
        if os.path.exists(outfn):
            cached_results = np.load(outfn, allow_pickle=True)
            return cached_results['fileresults']
    target = filedata['fit']['target']
    pred = filedata['fit']['prediction']
    test_idxs = filedata['fit']['idxes_test']
    trialsdf = bbone.load_trials_df(filedata['eid'], one=one)
    fileresults = {x: filedata[x] for x in filedata if x in ['subject', 'eid', 'probe', 'region']}

    lowprob_arr, highprob_arr = get_target_df(target, pred, test_idxs, trialsdf, one)
    params, low_slope, high_slope, shift = fit_get_shift_range(lowprob_arr, highprob_arr)
    fileresults['low_pLeft_slope'] = low_slope
    fileresults['high_pLeft_slope'] = high_slope
    fileresults['shift'] = shift
    fileresults['low_bias'] = params['low_pars'][0]
    fileresults['high_bias'] = params['high_pars'][0]
    fileresults['low_gamma1'] = params['low_pars'][2]
    fileresults['high_gamma1'] = params['high_pars'][2]
    fileresults['low_gamma2'] = params['low_pars'][3]
    fileresults['high_gamma2'] = params['high_pars'][3]
    fileresults['low_L'] = params['low_likelihood']
    fileresults['high_L'] = params['high_likelihood']
    fileresults['low_range'] = params['low_fit_trace'][-1] - params['low_fit_trace'][0]
    fileresults['high_range'] = params['high_fit_trace'][-1] - params['high_fit_trace'][0]
    fileresults['mean_range'] = np.mean([fileresults['low_range'], fileresults['high_range']])

    input_arrs = {'lowprob_arr': lowprob_arr, 'highprob_arr': highprob_arr}
    fit_pars = params
    null_pars = []
    for i in tqdm(range(len(filedata['pseudosessions'])), 'Pseudo num:', leave=False):
        keystr = f'run{i}_'
        fit = filedata['pseudosessions'][i]
        target = fit['target']
        pred = fit['prediction']
        test_idxs = fit['idxes_test']
        lowprob_arr, highprob_arr = get_target_df(target, pred, test_idxs, trialsdf, one)
        try:
            params, low_slope, high_slope, shift = fit_get_shift_range(lowprob_arr, highprob_arr)
            null_pars.append(params)
        except IndexError:
            fileresults[keystr + 'high_slope'] = np.nan
            fileresults[keystr + 'low_slope'] = np.nan
            fileresults[keystr + 'range'] = np.nan
            fileresults[keystr + 'shift'] = np.nan
            null_pars.append(np.nan)
            continue
        fileresults[keystr + 'low_slope'] = low_slope
        fileresults[keystr + 'high_slope'] = high_slope
        ltrace = params['low_fit_trace']
        htrace = params['high_fit_trace']
        fileresults[keystr + 'range'] = np.mean([ltrace[-1] - ltrace[0],
                                                 htrace[-1] - htrace[0]])
        fileresults[keystr + 'shift'] = shift
    outdict = {'fileresults': fileresults, 'fit_params': fit_pars, 'input_arrs': input_arrs,
               'null_pars': null_pars}
    fw = open(outfn, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return fileresults


if __name__ == "__main__":
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    fitpath = Path('/home/gercek/scratch/results/decoding/')
    fitdate = '2021-11-08'
    meta_f = fitdate + \
        '_decode_signcont_task_Lasso_align_stimOn_times_200_pseudosessions.metadata.pkl'
    fit_metadata = pd.read_pickle(meta_f)

    filenames = []
    ignorestrs = ['void', 'root', 'neurometric']
    for path in fitpath.rglob(f'{fitdate}*.pkl'):
        pathstr = str(path)
        if not any(pathstr.find(x) > -1 for x in ignorestrs):
            filenames.append(path)

    N_CORES = 1
    cluster = SLURMCluster(cores=N_CORES, memory='8GB', processes=1, queue="shared-cpu",
                           walltime="00:45:00", log_directory='/home/gercek/dask-worker-logs',
                           interface='ib0',
                           extra=["--lifetime", "45m"],
                           job_cpu=N_CORES, env_extra=[f'export OMP_NUM_THREADS={N_CORES}',
                                                       f'export MKL_NUM_THREADS={N_CORES}',
                                                       f'export OPENBLAS_NUM_THREADS={N_CORES}'])
    cluster.adapt(minimum_jobs=0, maximum_jobs=500)
    client = Client(cluster)

    fitresults = []
    for file in tqdm(filenames, 'Region rec:'):
        fitresults.append(client.submit(fit_file, file, pure=False, overwrite=True))

    columnorder = [*[f'run{i}_shift' for i in range(200)],
                   *[f'run{i}_low_slope' for i in range(200)],
                   *[f'run{i}_high_slope' for i in range(200)],
                   *[f'run{i}_range' for i in range(200)]]

    resultsdf = pd.DataFrame([x.result() for x in fitresults if x.status == 'finished'])
    resultsdf = resultsdf.set_index(['eid', 'probe', 'region'])
    resultsdf = resultsdf.reindex(columns=['subject',
                                           'low_pLeft_slope',
                                           'high_pLeft_slope',
                                           'shift',
                                           'low_bias',
                                           'high_bias'
                                           'low_range',
                                           'high_range',
                                           'mean_range',
                                           'low_gamma1',
                                           'high_gamma1',
                                           'low_gamma2',
                                           'high_gamma2',
                                           'low_L',
                                           'high_L',
                                           *columnorder])

    def compute_perc(row, target):
        mapper = {'high_pLeft_slope': 'high_slope',
                  'low_pLeft_slope': 'low_slope',
                  'shift': 'shift'}
        base = row[target]
        null = row['run0_' + mapper[target]:'run199_' + mapper[target]]
        return percentileofscore(null, base)

    def compute_zsc(row, target):
        mapper = {'high_pLeft_slope': 'high_slope',
                  'low_pLeft_slope': 'low_slope',
                  'shift': 'shift'}
        base = row.loc[target]
        null = row.loc['run0_' + mapper[target]:'run199_' + mapper[target]]
        return zscore(np.append(null.astype(float).values, base))[-1]

    targets = ['high_pLeft_slope', 'low_pLeft_slope', 'shift']
    for target in targets:
        for name, metric in {'perc': compute_perc, 'zsc': compute_zsc}.items():
            colname = f'{target}_{name}'
            resultsdf[colname] = resultsdf.apply(metric, target=target, axis=1)

    resultsdf['mean_shift_null'] = resultsdf.apply(lambda x:
                                                   np.mean(x.loc['run0_shift':'run199_shift']),
                                                   axis=1)
    resultsdf['mean_slope'] = resultsdf.apply(lambda x:
                                              np.mean([x.low_pLeft_slope, x.high_pLeft_slope]),
                                              axis=1)
    resultsdf['mean_slope_null'] = resultsdf.apply(
        lambda x: np.mean(
            np.vstack([x.loc['run0_high_slope':'run199_high_slope'].astype(float).values,
                       x.loc['run0_low_slope':'run199_low_slope'].astype(float).values])),
        axis=1)

    grpbyagg = resultsdf.groupby('region').agg({'mean_range': 'mean', 'shift': 'mean',

                                                'shift_zsc': 'mean', 'shift_perc': 'mean',
                                                'run0_range': 'mean',
                                                'run0_shift': 'mean'})
    slopedf = grpbyagg['mean_range'].to_frame()
    slopedf['pseudo'] = False
    slopedf = slopedf.set_index('pseudo', append=True)
    null_slopedf = grpbyagg['run0_range'].to_frame()
    null_slopedf = null_slopedf.rename(columns={'run0_range': 'mean_range'})
    null_slopedf['pseudo'] = True
    null_slopedf = null_slopedf.set_index('pseudo', append=True)
    full_slopes = slopedf.append(null_slopedf)

    shiftdf = grpbyagg['shift'].to_frame()
    shiftdf['pseudo'] = False
    shiftdf = shiftdf.set_index('pseudo', append=True)
    null_shiftdf = grpbyagg['run0_shift'].to_frame()
    null_shiftdf = null_shiftdf.rename(columns={'run0_shift': 'shift'})
    null_shiftdf['pseudo'] = True
    null_shiftdf = null_shiftdf.set_index('pseudo', append=True)
    full_shifts = shiftdf.append(null_shiftdf)
    slope_shift_arr = full_slopes.join(full_shifts)
