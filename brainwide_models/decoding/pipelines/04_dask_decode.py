from one.api import ONE
import pandas as pd
try:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, LocalCluster
    import dask
except:
    import warnings

    warnings.warn('dask import failed')
    pass

from settings.settings import *
from functions.decoding import fit_eid
import pickle

# import cached data
insdf = pd.read_parquet(DECODING_PATH.joinpath('insertions.pqt'))
insdf = insdf[insdf.spike_sorting != '']
eids = insdf['eid'].unique()

# create necessary empty directories if not existing
DECODING_PATH.joinpath('results').mkdir(exist_ok=True)
DECODING_PATH.joinpath('results', 'behavioral').mkdir(exist_ok=True)
DECODING_PATH.joinpath('results', 'neural').mkdir(exist_ok=True)
DECODING_PATH.joinpath('logs').mkdir(exist_ok=True)
DECODING_PATH.joinpath('logs', 'dask-worker-logs').mkdir(exist_ok=True)

# Generate cluster interface and map eids to workers via dask.distributed.Client
if LOCAL:
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
else:
    N_CORES = 12
    cluster = SLURMCluster(cores=N_CORES, memory='18GB', processes=1, queue="shared-cpu",
                           walltime="12:00:00",
                           log_directory=DECODING_PATH.joinpath('logs', 'dask-worker-logs'),
                           interface='ib0',
                           extra=["--lifetime", "20h", "--lifetime-stagger", "120m"],
                           job_cpu=N_CORES, env_extra=[f'export OMP_NUM_THREADS={N_CORES}',
                                                       f'export MKL_NUM_THREADS={N_CORES}',
                                                       f'export OPENBLAS_NUM_THREADS={N_CORES}'])
    cluster.adapt(minimum_jobs=len(eids), maximum_jobs=len(eids) * N_JOBS_PER_SESSION // 2)
    #cluster.scale(len(eids) * N_JOBS_PER_SESSION)
client = Client(cluster)
# verify you have at least 1 process before continuing (if not, you must wait that there is a process in cluster
# and then relaunch client = Client(cluster)
if USE_IMPOSTER_SESSION:
    imposterdf = pd.read_parquet(DECODING_PATH.joinpath('imposterSessions_beforeRecordings.pqt'))
    imposterdf_future = client.scatter(imposterdf)
else:
    imposterdf_future = None

one = ONE(mode='local')
one_future = client.scatter(one)

kwargs = {'imposterdf': None, 'nb_runs': N_RUNS, 'single_region': SINGLE_REGION, 'merged_probes': MERGED_PROBES,
          'modelfit_path': DECODING_PATH.joinpath('results', 'behavioral'),
          'output_path': DECODING_PATH.joinpath('results', 'neural'), 'one': None,
          'estimator_kwargs': ESTIMATOR_KWARGS, 'hyperparam_grid': HPARAM_GRID,
          'save_binned': SAVE_BINNED, 'shuffle': SHUFFLE, 'balanced_weight': BALANCED_WEIGHT,
          'normalize_input': NORMALIZE_INPUT, 'normalize_output': NORMALIZE_OUTPUT,
          'compute_on_each_fold': COMPUTE_NEURO_ON_EACH_FOLD,
          'force_positive_neuro_slopes': FORCE_POSITIVE_NEURO_SLOPES,
          'estimator': ESTIMATOR, 'target': TARGET, 'model': MODEL, 'align_time': ALIGN_TIME,
          'no_unbias': NO_UNBIAS, 'min_rt': MIN_RT, 'min_behav_trials': MIN_BEHAV_TRIAS,
          'qc_criteria': QC_CRITERIA, 'min_units': MIN_UNITS, 'time_window': TIME_WINDOW,
          'use_imposter_session': USE_IMPOSTER_SESSION, 'compute_neurometric': COMPUTE_NEUROMETRIC,
          'border_quantiles_neurometric': BORDER_QUANTILES_NEUROMETRIC, 'today': DATE
          }

filenames = []
for i, eid in enumerate(eids[:1]):
    if (eid in excludes or np.any(insdf[insdf['eid'] == eid]['spike_sorting'] == "")):
        print(f"dud {eid}")
        continue
    print(f"{i}, session: {eid}")
    for job_id in range(N_JOBS_PER_SESSION):
        pseudo_ids = np.arange(job_id * N_PSEUDO_PER_JOB, (job_id + 1) * N_PSEUDO_PER_JOB) + 1
        if 1 in pseudo_ids:
            pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype('int64')
        fns = client.submit(fit_eid, eid=eid, sessdf=insdf, pseudo_ids=pseudo_ids, **kwargs)
        filenames.append(fns)

# WAIT FOR COMPUTATION TO FINISH BEFORE MOVING ON
# %% Collate results into master dataframe and save
tmp = [x.result() for x in filenames if x.status == 'finished']

finished = []
for fns in tmp:
    finished.extend(fns)

indexers = ['subject', 'eid', 'probe', 'region']
indexers_neurometric = ['low_slope', 'high_slope', 'low_range', 'high_range', 'shift', 'mean_range', 'mean_slope']
resultslist = []
for fn in finished:
    fo = open(fn, 'rb')
    result = pickle.load(fo)
    fo.close()
    for i_run in range(len(result['fit'])):
        tmpdict = {**{x: result[x] for x in indexers},
                   'fold': -1,
                   'pseudo_id': result['pseudo_id'],
                   'N_units': result['N_units'],
                   'run_id': i_run + 1,
                   'mask': ''.join([str(item) for item in list(result['fit'][i_run]['mask'].values * 1)]),
                   'R2_test': result['fit'][i_run]['Rsquared_test_full']}
        if result['fit'][i_run]['full_neurometric'] is not None:
            tmpdict = {**tmpdict,
                       **{idx_neuro: result['fit'][i_run]['full_neurometric'][idx_neuro]
                          for idx_neuro in indexers_neurometric}}
        resultslist.append(tmpdict)
        for kfold in range(result['fit'][i_run]['nFolds']):
            tmpdict = {**{x: result[x] for x in indexers},
                       'fold': kfold,
                       'pseudo_id': result['pseudo_id'],
                       'N_units': result['N_units'],
                       'run_id': i_run + 1,
                       'R2_test': result['fit'][i_run]['Rsquareds_test'][kfold],
                       'Best_regulCoef': result['fit'][i_run]['best_params'][kfold],
                       }
            if result['fit'][i_run]['fold_neurometric'] is not None:
                tmpdict = {**tmpdict,
                           **{idx_neuro: result['fit'][i_run]['fold_neurometric'][kfold][idx_neuro]
                              for idx_neuro in indexers_neurometric}}
            resultslist.append(tmpdict)
resultsdf = pd.DataFrame(resultslist).set_index(indexers)

estimatorstr = strlut[ESTIMATOR]
start_tw, end_tw = TIME_WINDOW
fn = str(DECODING_PATH.joinpath('results', 'neural', '_'.join([DATE, 'decode', TARGET,
                                                               dut.modeldispatcher[MODEL] if TARGET in ['prior',
                                                                                                        'prederr'] else 'task',
                                                               estimatorstr, 'align', ALIGN_TIME, str(N_PSEUDO),
                                                               'pseudosessions',
                                                               'regionWise' if SINGLE_REGION else 'allProbes',
                                                               'timeWindow', str(start_tw).replace('.', '_'),
                                                               str(end_tw).replace('.', '_')])))
if COMPUTE_NEUROMETRIC:
    fn = fn + '_'.join(['', 'neurometricPLeft', dut.modeldispatcher[MODEL]])

if ADD_TO_SAVING_PATH != '':
    fn = fn + '_' + ADD_TO_SAVING_PATH

fn = fn + '.parquet'

metadata_df = pd.Series({'filename': fn, **fit_metadata})
metadata_fn = '.'.join([fn.split('.')[0], 'metadata', 'pkl'])
resultsdf.to_parquet(fn)
metadata_df.to_pickle(metadata_fn)

# save weights
weightsdict = {}
for fn in finished:
    fo = open(fn, 'rb')
    result = pickle.load(fo)
    fo.close()
    for i_run in range(len(result['fit'])):
        weightsdict = {**weightsdict, **{(tuple(result[x] for x in indexers)
                                          + ('pseudo_id_{}'.format(result['pseudo_id']),
                                             'run_id_{}'.format(i_run + 1)))
                                         : np.vstack(result['fit'][i_run]['weights'])}}

with open(metadata_fn.split('.metadata.pkl')[0] + '.weights.pkl', 'wb') as f:
    pickle.dump(weightsdict, f)

# command to close the ongoing placeholder
# client.close(); cluster.close()

# If you want to get the errors per-failure in the run:
"""
failures = [(i, x) for i, x in enumerate(filenames) if x.status == 'error']
for i, failure in failures:
    print(i, failure.exception(), failure.key)
print(len(failures))
print(np.array(failures)[:,0])
import traceback
tb = failure.traceback()
traceback.print_tb(tb)
print(len([(i, x) for i, x in enumerate(filenames) if x.status == 'cancelled']))
print(len([(i, x) for i, x in enumerate(filenames) if x.status == 'error']))
print(len([(i, x) for i, x in enumerate(filenames) if x.status == 'lost']))
print(len([(i, x) for i, x in enumerate(filenames) if x.status == 'finished']))
"""
# You can also get the traceback from failure.traceback and print via `import traceback` and
# traceback.print_tb()


'''
custom static plot
fo = open(finished[418], 'rb') # 416, 418
result = pickle.load(fo)
fo.close()

low_trace = np.vstack([result["fit"][k]['full_neurometric']['low_fit_trace'] for k in range(len(result["fit"]))])
high_trace = np.vstack([result["fit"][k]['full_neurometric']['high_fit_trace'] for k in range(len(result["fit"]))])
'''
