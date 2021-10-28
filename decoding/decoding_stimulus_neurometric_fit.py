import numpy as np
import pandas as pd
import brainbox.behavior.pyschofit as pfit
import brainbox.io.one as bbone
from pathlib import Path
from one.api import ONE
from tqdm import tqdm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from scipy.stats import zscore, percentileofscore

fitpath = Path('/home/gercek/scratch/results/decoding/')
fitdate = '2021-10-27'
meta_f = f'{fitdate}_decode_signcont_task_Lasso_align_stimOn_times_200_pseudosessions.metadata.pkl'
fit_metadata = pd.read_pickle(meta_f)


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


def fit_file(file):
    one = ONE()
    filedata = np.load(file, allow_pickle=True)
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

    for i in tqdm(range(fit_metadata.n_pseudo), 'Pseudo num:', leave=False):
        keystr = f'run{i}_'
        fit = filedata['pseudosessions'][i]
        target = fit['target']
        pred = fit['prediction']
        test_idxs = fit['idxes_test']
        lowprob_arr, highprob_arr = get_target_df(target, pred, test_idxs, trialsdf, one)
        params, low_slope, high_slope, shift = fit_get_shift_range(lowprob_arr, highprob_arr)
        fileresults[keystr + 'low_slope'] = low_slope
        fileresults[keystr + 'high_slope'] = high_slope
        fileresults[keystr + 'shift'] = shift
    return fileresults


filenames = []
for path in Path('/home/gercek/scratch/results/decoding/').rglob(f'{fitdate}*.pkl'):
    pathstr = str(path)
    if not pathstr.find('root') > -1 and not pathstr.find('void') > -1:
        filenames.append(path)

N_CORES = 1
cluster = SLURMCluster(cores=N_CORES, memory='16GB', processes=1, queue="shared-cpu",
                       walltime="00:15:00", log_directory='/home/gercek/dask-worker-logs',
                       interface='ib0',
                       extra=["--lifetime", "15m"],
                       job_cpu=N_CORES, env_extra=[f'export OMP_NUM_THREADS={N_CORES}',
                                                   f'export MKL_NUM_THREADS={N_CORES}',
                                                   f'export OPENBLAS_NUM_THREADS={N_CORES}'])
cluster.adapt(minimum_jobs=0, maximum_jobs=200)
client = Client(cluster)

fitresults = []
for file in tqdm(filenames, 'Region rec:'):
    fitresults.append(client.submit(fit_file, file))

columnorder = [*[f'run{i}_shift' for i in range(200)], *[f'run{i}_low_slope' for i in range(200)],
               *[f'run{i}_high_slope' for i in range(200)]]

resultsdf = pd.DataFrame([x.result() for x in fitresults if x.status == 'finished'])
resultsdf = resultsdf.set_index(['eid', 'probe', 'region'])
resultsdf = resultsdf.reindex(columns=['subject', 'low_pLeft_slope',
                                       'high_pLeft_slope', 'shift', *columnorder])


def compute_perc(row, target):
    mapper = {'high_pLeft_slope': 'high_slope', 'low_pLeft_slope': 'low_slope', 'shift': 'shift'}
    base = row[target]
    null = row['run0_' + mapper[target]:'run199_' + mapper[target]]
    return percentileofscore(null, base)


def compute_zsc(row, target):
    mapper = {'high_pLeft_slope': 'high_slope', 'low_pLeft_slope': 'low_slope', 'shift': 'shift'}
    base = row.loc[target]
    null = row.loc['run0_' + mapper[target]:'run199_' + mapper[target]]
    return zscore(np.append(null.astype(float).values, base))[-1]


targets = ['high_pLeft_slope', 'low_pLeft_slope', 'shift']
for target in targets:
    for name, metric in {'perc': compute_perc, 'zsc': compute_zsc}.items():
        colname = f'{target}_{name}'
        resultsdf[colname] = resultsdf.apply(metric, target=target, axis=1)
