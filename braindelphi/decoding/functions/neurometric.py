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

'''
trialsdf_neurometric = nb_trialsdf.reset_index() if (pseudo_id == -1) else \
    pseudosess[pseudomask].reset_index()
if kwargs['model'] is not None:
    blockprob_neurometric = dut.compute_target(
        'pLeft',
        subject,
        subjeids,
        eid,
        kwargs['modelfit_path'],
        binarization_value=kwargs['binarization_value'],
        modeltype=kwargs['model'],
        beh_data_test=trialsdf if pseudo_id == -1 else pseudosess,
        behavior_data_train=behavior_data_train,
        one=one)

    trialsdf_neurometric['blockprob_neurometric'] = np.stack([
        np.greater_equal(
            blockprob_neurometric[(mask & (trialsdf.choice != 0) if pseudo_id
                                                                    == -1 else pseudomask)], border).astype(int)
        for border in kwargs['border_quantiles_neurometric']
    ]).sum(0)

else:
    blockprob_neurometric = trialsdf_neurometric['probabilityLeft'].replace(
        0.2, 0).replace(0.8, 1)
    trialsdf_neurometric['blockprob_neurometric'] = blockprob_neurometric
'''

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
                       'blockprob': trialsdf.loc[test_idxs, 'blockprob_neurometric']},
                      index=test_idxs)
    grpby = df.groupby(['blockprob', 'stimuli'])
    grpbyagg = grpby.agg({'sign': [('num_trials', 'count'),
                                   ('prop_L', lambda x: ((x == 1).sum() + (x == 0).sum() / 2.) / len(x))]})
    return [grpbyagg.loc[k].reset_index().values.T for k in
            grpbyagg.index.get_level_values('blockprob').unique().sort_values()]


def get_neurometric_parameters_(prob_arr, possible_contrasts, force_positive_neuro_slopes=False, nfits=100):
    if force_positive_neuro_slopes:
        pars, L = pfit.mle_fit_psycho(prob_arr,
                                      P_model='erf_psycho_2gammas',
                                      nfits=nfits)
    else:
        pars, L = pfit.mle_fit_psycho(prob_arr,
                                      P_model='erf_psycho_2gammas',
                                      nfits=nfits,
                                      parmin=np.array([-1., -10., 0., 0.]),
                                      parmax=np.array([1., 10., 0.4, 0.4]))
    contrasts = prob_arr[0, :] if possible_contrasts is None else possible_contrasts
    fit_trace = pfit.erf_psycho_2gammas(pars, contrasts)
    range = fit_trace[np.argwhere(np.isclose(contrasts, 1)).flat[0]] - \
            fit_trace[np.argwhere(np.isclose(contrasts, -1)).flat[0]]
    slope = pars[1]
    zerind = np.argwhere(np.isclose(contrasts, 0)).flat[0]
    return {'range': range, 'slope': slope, 'pars': pars, 'L': L, 'fit_trace': fit_trace, 'zerind': zerind}

def fit_get_shift_range(prob_arrs,
                        force_positive_neuro_slopes=False,
                        seed_=None,
                        possible_contrasts=np.array([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1]),
                        nfits=100,
                        ):
    """
    Fit psychometric functions with erf with two gammas for lapse rate, and return the parameters,
    traces, and the slope and shift.
    Parameters
    ----------
    lowprob_arr
    seed_
    possible_contrasts
    lowprob_arr : numpy.ndarray
        3 x 9 array for function fitting (see get_target_df), from low probability left block
    highprob_arr : numpy.ndarray
        Same as above, for high probability Left
    """
    # pLeft = 0.2 blocks
    if seed_ is not None:
        np.random.seed(seed_)
    lows = get_neurometric_parameters_(prob_arrs[0], possible_contrasts,
                                       force_positive_neuro_slopes=force_positive_neuro_slopes, nfits=nfits)
    # pLeft = 0.8 blocks
    if seed_ is not None:
        np.random.seed(seed_)
    highs = get_neurometric_parameters_(prob_arrs[-1], possible_contrasts,
                                        force_positive_neuro_slopes=force_positive_neuro_slopes, nfits=nfits)

    # compute shift
    shift = highs['fit_trace'][highs['zerind']] - lows['fit_trace'][lows['zerind']]
    params = {'low_pars': lows['pars'], 'low_likelihood': lows['L'],
              'high_pars': highs['pars'], 'high_likelihood': highs['L'],
              'low_fit_trace': lows['fit_trace'], 'high_fit_trace': highs['fit_trace'],
              'low_slope': lows['slope'], 'high_slope': highs['slope'], 'low_range': lows['range'],
              'high_range': highs['range'], 'shift': shift,
              'mean_range': (lows['range'] + highs['range'] ) /2.,
              'mean_slope': (lows['slope'] + highs['slope'] ) /2.}

    params = {**params, **{'NB_QUANTILES': len(prob_arrs)}}
    for (out, k) in zip([lows, highs], [0, len(prob_arrs) - 1]):
        params = {**params, **{'quantile_%i_0contrastLevel' % k: out['fit_trace'][out['zerind']],
                               'quantile_%i_pars' % k: out['pars'],
                               'quantile_%i_likelihood' % k: out['L'],
                               'quantile_%i_fit_trace' % k: out['fit_trace'],
                               'quantile_%i_slope' % k: out['slope'],
                               'quantile_%i_range' % k: out['range']}}

    if len(prob_arrs) > 2:
        for k in range(1, len(prob_arrs) - 1):
            mediums = get_neurometric_parameters_(prob_arrs[k], possible_contrasts,
                                                  force_positive_neuro_slopes=force_positive_neuro_slopes)
            params = {**params, **{'quantile_%i_0contrastLevel' % k: mediums['fit_trace'][mediums['zerind']],
                                   'quantile_%i_pars' % k: mediums['pars'],
                                   'quantile_%i_likelihood' % k: mediums['L'],
                                   'quantile_%i_fit_trace' % k: mediums['fit_trace'],
                                   'quantile_%i_slope' % k: mediums['slope'],
                                   'quantile_%i_range' % k: mediums['range']}}
    return params

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

def get_neurometric_parameters(fit_result, trialsdf, one, compute_on_each_fold, force_positive_neuro_slopes):
    # fold-wise neurometric curve
    if compute_on_each_fold:
        raise NotImplementedError('Sorry, this is not up to date to perform computation on each folds. Ask Charles F.')
    if compute_on_each_fold:
        try:
            prob_arrays = [get_target_df(fit_result['target'],
                                         fit_result['predictions'][k],
                                         fit_result['idxes_test'][k],
                                         trialsdf, one)
                           for k in range(fit_result['nFolds'])]

            fold_neurometric = [fit_get_shift_range(prob_arrays[k][0], prob_arrays[k][1], force_positive_neuro_slopes)
                                for k in range(fit_result['nFolds'])]
        except KeyError:
            fold_neurometric = None
    else:
        fold_neurometric = None

    # full neurometric curve
    full_test_prediction = np.zeros(len(fit_result['target']))
    for k in range(fit_result['nFolds']):
        full_test_prediction[fit_result['idxes_test'][k]] = fit_result['predictions_test'][k]

    prob_arrs = get_target_df(fit_result['target'], full_test_prediction,
                              np.arange(len(fit_result['target'])), trialsdf, one)
    full_neurometric = fit_get_shift_range(prob_arrs, force_positive_neuro_slopes)

    return full_neurometric, fold_neurometric


if __name__ == "__main__":
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    fitpath = Path('/home/gercek/scratch/results/decoding/')
    fitdate = '2021-11-08'
    meta_f = fitdate + \
             '_decode_signcont_task_Lasso_align_stimOn_times_2_pseudosessions.metadata.pkl'
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
