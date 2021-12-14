import os
import pickle
import logging
import numpy as np
import pandas as pd
import decoding_utils as dut
import brainbox.io.one as bbone
import sklearn.linear_model as sklm
import models.utils as mut
from pathlib import Path
from datetime import date
from one.api import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction
# from brainbox.singlecell import calculate_peths
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import generate_pseudo_session
from brainbox.metrics.single_units import quick_unit_metrics
from decoding_stimulus_neurometric_fit import get_neurometric_parameters

try:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
except:
    import warnings
    warnings.warn('dask import failed')
    pass
from tqdm import tqdm
from ibllib.atlas import AllenAtlas

logger = logging.getLogger('ibllib')
logger.disabled = True

strlut = {sklm.Lasso: 'Lasso',
          sklm.LassoCV: 'LassoCV',
          sklm.Ridge: 'Ridge',
          sklm.RidgeCV: 'RidgeCV',
          sklm.LinearRegression: 'PureLinear',
          sklm.LogisticRegression: 'Logistic'}

# %% Run param definitions

# aligned -> histology was performed by one experimenter
# resolved -> histology was performed by 2-3 experiments
SESS_CRITERION = 'aligned-behavior'  # aligned and behavior
TARGET = 'signcont'
MODEL = expSmoothing_prevAction
#MODELFIT_PATH = '/home/users/f/findling/ibl/prior-localization/decoding/results/behavior/'
#OUTPUT_PATH = '/home/users/f/findling/ibl/prior-localization/decoding/results/decoding/'
MODELFIT_PATH = '/Users/csmfindling/Documents/Postdoc-Geneva/IBL/behavior/prior-localization/decoding/results/behavior/'
OUTPUT_PATH = '/Users/csmfindling/Documents/Postdoc-Geneva/IBL/behavior/prior-localization/decoding/results/decoding/'
ALIGN_TIME = 'goCue_times'
TIME_WINDOW = (-0.4, -0.1)
ESTIMATOR = sklm.Lasso  # Must be in keys of strlut above
ESTIMATOR_KWARGS = {'tol': 0.0001, 'max_iter': 10000, 'fit_intercept': True}
N_PSEUDO = 2
MIN_UNITS = 10
MIN_BEHAV_TRIAS = 200
MIN_RT = 0.08  # Float (s) or None
NO_UNBIAS = True
DATE = str(date.today())
SHUFFLE = False
# Basically, quality metric on the stability of a single unit. Should have 1 metric per neuron
QC_CRITERIA = 3 / 3  # In {None, 1/3, 2/3, 3/3}
SAVE_BINNED = False  # Debugging parameter, not usually necessary

HPARAM_GRID = {'alpha': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100])}
# HPARAM_GRID = [0.001, 0.01, 0.1, 1, 10, 100] # None  # For GridSearchCV, set to None if using a CV estimator

fit_metadata = {
    'criterion': SESS_CRITERION,
    'target': TARGET,
    'model_type': dut.modeldispatcher[MODEL],
    'modelfit_path': MODELFIT_PATH,
    'output_path': OUTPUT_PATH,
    'align_time': ALIGN_TIME,
    'time_window': TIME_WINDOW,
    'estimator': strlut[ESTIMATOR],
    'n_pseudo': N_PSEUDO,
    'min_units': MIN_UNITS,
    'min_behav_trials': MIN_BEHAV_TRIAS,
    'qc_criteria': QC_CRITERIA,
    'date': DATE,
    'shuffle': SHUFFLE,
    'no_unbias': NO_UNBIAS,
    'hyperparameter_grid': HPARAM_GRID,
    'save_binned': SAVE_BINNED,
}


# %% Define helper functions for dask workers to use
def save_region_results(fit_result, pseudo_results, subject, eid, probe, region, N):
    subjectfolder = Path(OUTPUT_PATH).joinpath(subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    for folder in [subjectfolder, eidfolder, probefolder]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    start_tw, end_tw = TIME_WINDOW
    fn = '_'.join([DATE, region, 'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_')]) + '.pkl'
    fw = open(probefolder.joinpath(fn), 'wb')
    outdict = {'fit': fit_result, 'pseudosessions': pseudo_results,
               'subject': subject, 'eid': eid, 'probe': probe, 'region': region, 'N_units': N}
    pickle.dump(outdict, fw)
    fw.close()
    return probefolder.joinpath(fn)


def fit_eid(eid, sessdf):
    one = ONE()
    atlas = AllenAtlas()

    estimator = ESTIMATOR(**ESTIMATOR_KWARGS)

    subject = sessdf.xs(eid, level='eid').index[0]
    subjeids = sessdf.xs(subject, level='subject').index.unique()
    brainreg = dut.BrainRegions()
    behavior_data = mut.load_session(eid, one=one)
    try:
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, MODELFIT_PATH,
                                  modeltype=MODEL, beh_data=behavior_data,
                                  one=one)
    except ValueError:
        print('Model not fit.')
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, MODELFIT_PATH,
                                  modeltype=MODEL, one=one)

    try:
        trialsdf = bbone.load_trials_df(eid, one=one, addtl_types=['firstMovement_times'])
        if len(trialsdf) != len(tvec):
            raise IndexError
    except IndexError:
        raise IndexError('Problem in the dimensions of dataframe of session')
    trialsdf['react_times'] = trialsdf['firstMovement_times'] - trialsdf[ALIGN_TIME]
    mask = trialsdf[ALIGN_TIME].notna()
    if NO_UNBIAS:
        mask = mask & (trialsdf.probabilityLeft != 0.5).values
    if MIN_RT is not None:
        mask = mask & (~(trialsdf.react_times < MIN_RT)).values

    nb_trialsdf = trialsdf[mask]
    msub_tvec = tvec[mask]

    filenames = []
    if len(msub_tvec) <= MIN_BEHAV_TRIAS:
        return filenames

    print(f'Working on eid : {eid}')
    for probe in tqdm(sessdf.loc[subject, eid, :].probe, desc='Probe: ', leave=False):
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid,
                                                                    one=one,
                                                                    probe=probe,
                                                                    brain_atlas=atlas,
                                                                    dataset_types=['spikes.depths', 'spikes.amps'],
                                                                    aligned=True)

        beryl_reg = dut.remap_region(clusters[probe].atlas_id, br=brainreg)
        if QC_CRITERIA:
            metrics = pd.DataFrame.from_dict(quick_unit_metrics(spikes[probe].clusters,
                                                                spikes[probe].times,
                                                                spikes[probe].amps,
                                                                spikes[probe].depths,
                                                                cluster_ids=np.arange(len(beryl_reg))))
            try:
                metrics_verif = clusters[probe].metrics
                if beryl_reg.shape[0] == len(metrics_verif):
                    if not np.all(((metrics_verif.label - metrics.label) < 1e-10) + metrics_verif.label.isna()):
                        raise ValueError('there is a problem in the metric computations')
            except AttributeError:
                pass
            qc_pass = (metrics.label >= QC_CRITERIA)
            if (beryl_reg.shape[0] - 1) != qc_pass.index.max():
                raise IndexError('Shapes of metrics and number of clusters '
                                 'in regions don\'t match')
        else:
            qc_pass = np.ones_like(beryl_reg, dtype=bool)
        regions = np.unique(beryl_reg)
        # warnings.filterwarnings('ignore')
        for region in tqdm(regions, desc='Region: ', leave=False):
            reg_mask = beryl_reg == region
            reg_clu_ids = np.argwhere(reg_mask & qc_pass.values).flatten()
            N_units = len(reg_clu_ids)
            if N_units < MIN_UNITS:
                continue
            # or get_spike_count_in_bins
            if np.any(np.isnan(nb_trialsdf[ALIGN_TIME])):
                # if this happens, verify scrub of NaN values in all aign times before get_spike_counts_in_bins
                raise ValueError('this should not happen')
            intervals = np.vstack([nb_trialsdf[ALIGN_TIME] + TIME_WINDOW[0],
                                   nb_trialsdf[ALIGN_TIME] + TIME_WINDOW[1]]).T
            spikemask = np.isin(spikes[probe].clusters, reg_clu_ids)
            regspikes = spikes[probe].times[spikemask]
            regclu = spikes[probe].clusters[spikemask]
            binned, _ = get_spike_counts_in_bins(regspikes, regclu,
                                                 intervals)
            msub_binned = binned.T.astype(int)

            if len(msub_binned.shape) > 2:
                raise ValueError('Multiple bins are being calculated per trial,'
                                 'may be due to floating point representation error.'
                                 'Check window.')
            fit_result = dut.regress_target(msub_tvec, msub_binned, estimator,
                                            hyperparam_grid=HPARAM_GRID,
                                            save_binned=SAVE_BINNED, shuffle=SHUFFLE)

            # neurometric curve
            fit_result['full_neurometric'], fit_result['fold_neurometric'] = \
                get_neurometric_parameters(fit_result, nb_trialsdf.reset_index(), one)

            pseudo_results = []
            for _ in tqdm(range(N_PSEUDO), desc='Pseudo num: ', leave=False):
                pseudosess = generate_pseudo_session(trialsdf)
                msub_pseudo_tvec = dut.compute_target(TARGET, subject, subjeids, eid,
                                                      MODELFIT_PATH, modeltype=MODEL,
                                                      beh_data=pseudosess, one=one)[mask]
                pseudo_result = dut.regress_target(msub_pseudo_tvec, msub_binned, estimator,
                                                   hyperparam_grid=HPARAM_GRID, shuffle=SHUFFLE)

                # neurometric curve
                pseudo_result['full_neurometric'], pseudo_result['fold_neurometric'] = \
                    get_neurometric_parameters(pseudo_result, pseudosess[mask].reset_index(), one)

                pseudo_results.append(pseudo_result)
            filenames.append(save_region_results(fit_result, pseudo_results, subject,
                                                 eid, probe, region, N_units))

    return filenames


if __name__ == '__main__':
    from decode_prior import fit_eid

    # Generate cluster interface and map eids to workers via dask.distributed.Client
    sessdf = dut.query_sessions(selection=SESS_CRITERION)
    sessdf = sessdf.sort_values('subject').set_index(['subject', 'eid'])

    N_CORES = 2
    cluster = SLURMCluster(cores=N_CORES, memory='12GB', processes=1, queue="shared-cpu",
                           walltime="01:15:00",
                           log_directory='/home/users/f/findling/ibl/prior-localization/decoding/dask-worker-logs',
                           interface='ib0',
                           extra=["--lifetime", "70m", "--lifetime-stagger", "4m"],
                           job_cpu=N_CORES, env_extra=[f'export OMP_NUM_THREADS={N_CORES}',
                                                       f'export MKL_NUM_THREADS={N_CORES}',
                                                       f'export OPENBLAS_NUM_THREADS={N_CORES}'])
    cluster.adapt(minimum_jobs=0, maximum_jobs=200)
    client = Client(cluster)

    import time
    filenames = []
    for eid in sessdf.index.unique(level='eid'):
        fns = client.submit(fit_eid, eid, sessdf)
        filenames.append(fns)
    # WAIT FOR COMPUTATION TO FINISH BEFORE MOVING ON
    # %% Collate results into master dataframe and save
    tmp = [x.result() for x in filenames if x.status == 'finished']
    finished = []
    for fns in tmp:
        finished.extend(fns)

    indexers = ['subject', 'eid', 'probe', 'region']
    indexers_neurometric = ['low_slope', 'high_slope', 'low_range', 'high_range', 'shift']
    resultslist = []
    for fn in finished:
        fo = open(fn, 'rb')
        result = pickle.load(fo)
        fo.close()
        tmpdict = {**{x: result[x] for x in indexers},
                   'fold': -1,
                   **{idx_neuro: result['fit']['full_neurometric'][idx_neuro] for idx_neuro in indexers_neurometric},
                   **{'Rsquared_test': result['fit']['Rsquared_test_full']},
                   **{f'Rsquared_test_run{i}': result['pseudosessions'][i]['Rsquared_test_full']
                      for i in range(N_PSEUDO)},
                   **{str(idx_neuro) + f'_run{i}': result['pseudosessions'][i]['full_neurometric'][idx_neuro]
                      for i in range(N_PSEUDO) for idx_neuro in indexers_neurometric}}
        resultslist.append(tmpdict)
        for kfold in range(result['fit']['nFolds']):
            tmpdict = {**{x: result[x] for x in indexers},
                       'fold': kfold,
                       'Rsquared_test': result['fit']['Rsquareds_test'][kfold],
                       **{idx_neuro: result['fit']['fold_neurometric'][kfold][idx_neuro] for idx_neuro in
                          indexers_neurometric},
                       **{f'Rsquared_test_run{i}': result['pseudosessions'][i]['Rsquareds_test'][kfold]
                          for i in range(N_PSEUDO)},
                       **{str(idx_neuro) + f'_run{i}': result['pseudosessions'][i]['fold_neurometric'][kfold][idx_neuro]
                          for i in range(N_PSEUDO) for idx_neuro in indexers_neurometric}
                       }
            resultslist.append(tmpdict)
    resultsdf = pd.DataFrame(resultslist).set_index(indexers)

    estimatorstr = strlut[ESTIMATOR]
    start_tw, end_tw = TIME_WINDOW
    fn = OUTPUT_PATH + '_'.join([DATE, 'decode', TARGET,
                                 dut.modeldispatcher[MODEL] if TARGET in ['prior', 'prederr'] else 'task',
                                 estimatorstr, 'align', ALIGN_TIME, str(N_PSEUDO), 'pseudosessions',
                                 'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_')]) + \
         '.parquet'
    metadata_df = pd.Series({'filename': fn, **fit_metadata})
    metadata_fn = '.'.join([fn.split('.')[0], 'metadata', 'pkl'])
    resultsdf.to_parquet(fn)
    metadata_df.to_pickle(metadata_fn)

# command to close the ongoing placeholder
# client.close(); cluster.close()

# If you want to get the errors per-failure in the run:
"""
failures = [(i, x) for i, x in enumerate(filenames) if x.status == 'error']
for i, failure in failures:
    print(i, failure.exception(), failure.key)
print(len(failures))
"""
# You can also get the traceback from failure.traceback and print via `import traceback` and
# traceback.print_tb()
