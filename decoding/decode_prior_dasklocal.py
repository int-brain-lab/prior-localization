import os
import pickle
import logging
import warnings
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
from brainbox.singlecell import calculate_peths
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import generate_pseudo_session
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from tqdm import tqdm


logger = logging.getLogger('ibllib')
logger.disabled = True

strlut = {sklm.Lasso: 'Lasso',
          sklm.LassoCV: 'LassoCV',
          sklm.Ridge: 'Ridge',
          sklm.RidgeCV: 'RidgeCV',
          sklm.LinearRegression: 'PureLinear',
          sklm.LogisticRegression: 'Logistic'}

# %% Run param definitions

SESS_CRITERION = 'aligned-behavior'
TARGET = 'signcont'
MODEL = expSmoothing_prevAction
MODELFIT_PATH = '/home/gercek/Projects/prior-localization/results/inference/'
OUTPUT_PATH = '/home/gercek/scratch/results/decoding/'
ALIGN_TIME = 'goCue_times'
TIME_WINDOW = (-0.6, -0.2)
ESTIMATOR = sklm.LassoCV  # Must be in keys of strlut above
ESTIMATOR_KWARGS = {'tol': 0.0001, 'max_iter': 10000, 'fit_intercept': False}
N_PSEUDO = 2
MIN_UNITS = 10
MIN_RT = 0.08  # Float (s) or None
NO_UNBIAS = True
DATE = str(date.today())
QC_CRITERIA = 3/3  # In {None, 1/3, 2/3, 3/3}
SAVE_BINNED = False  # Debugging parameter, not usually necessary

HPARAM_GRID = None  # For GridSearchCV, set to None if using a CV estimator

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
    'qc_criteria': QC_CRITERIA,
    'date': DATE,
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
    fn = '_'.join([DATE, region]) + '.pkl'
    fw = open(probefolder.joinpath(fn), 'wb')
    outdict = {'fit': fit_result, 'pseudosessions': pseudo_results,
               'subject': subject, 'eid': eid, 'probe': probe, 'region': region, 'N_units': N}
    pickle.dump(outdict, fw)
    fw.close()
    return probefolder.joinpath(fn)


def fit_eid(eid):
    one = ONE()

    estimator = ESTIMATOR(**ESTIMATOR_KWARGS)

    subject = sessdf.xs(eid, level='eid').index[0]
    subjeids = sessdf.xs(subject, level='subject').index.unique()
    brainreg = dut.BrainRegions()
    behavior_data = mut.load_session(eid, one=one)
    try:
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, MODELFIT_PATH,
                                  modeltype=MODEL, beh_data=behavior_data, no_unbias=NO_UNBIAS,
                                  one=one)
    except ValueError:
        print('Model not fit.')
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, MODELFIT_PATH,
                                  modeltype=MODEL, no_unbias=NO_UNBIAS, one=one)

    msub_tvec = tvec # - np.mean(tvec)
    trialsdf = bbone.load_trials_df(eid, one=one, addtl_types=['firstMovement_times'])
    trialsdf = trialsdf[trialsdf[ALIGN_TIME].notna()]
    trialsdf['react_times'] = trialsdf['firstMovement_times'] - trialsdf['goCue_times']
    if NO_UNBIAS:
        nb_trialsdf = trialsdf[trialsdf.probabilityLeft != 0.5]
    else:
        nb_trialsdf = trialsdf.copy()
    
    if MIN_RT is not None:
        mask = ~(nb_trialsdf['react_times'] < MIN_RT)
        nb_trialsdf = nb_trialsdf[mask]
        msub_tvec = msub_tvec[mask]

    filenames = []
    print(f'Working on eid : {eid}')
    for probe in tqdm(sessdf.loc[subject, eid, :].probe, desc='Probe: ', leave=False):
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid,
                                                                    one=one,
                                                                    probe=probe,
                                                                    aligned=True)
        beryl_reg = dut.remap_region(clusters[probe].atlas_id, br=brainreg)
        if QC_CRITERIA:
            try:
                metrics = clusters[probe].metrics
            except AttributeError:
                raise AttributeError('Session has no QC metrics')
            qc_pass = metrics.label >= QC_CRITERIA
            if (beryl_reg.shape[0] - 1) != qc_pass.index.max():
                raise IndexError('Shapes of metrics and number of clusters '
                                 'in regions don\'t match')
        else:
            qc_pass = np.ones_like(beryl_reg, dtype=bool)
        regions = np.unique(beryl_reg)
        # warnings.filterwarnings('ignore')
        for region in tqdm(regions, desc='Region: ', leave=False):
            reg_mask = beryl_reg == region
            reg_clu_ids = np.argwhere((reg_mask & qc_pass).values).flatten()
            N_units = len(reg_clu_ids)
            if N_units < MIN_UNITS:
                continue
            # _, binned = calculate_peths(spikes[probe].times, spikes[probe].clusters, reg_clu_ids,
            #                             nb_trialsdf[ALIGN_TIME] + TIME_WINDOW[0], pre_time=0,
            #                             post_time=TIME_WINDOW[1] - TIME_WINDOW[0],
            #                             bin_size=TIME_WINDOW[1] - TIME_WINDOW[0], smoothing=0,
            #                             return_fr=False)
            # binned = binned.squeeze()
            intervals = np.vstack([nb_trialsdf[ALIGN_TIME] + TIME_WINDOW[0],
                                   nb_trialsdf[ALIGN_TIME] + TIME_WINDOW[1]]).T
            spikemask = np.isin(spikes[probe].clusters, reg_clu_ids)
            regspikes = spikes[probe].times[spikemask]
            regclu = spikes[probe].clusters[spikemask]
            binned, _ = get_spike_counts_in_bins(regspikes, regclu,
                                                 intervals)
            binned = binned.T.astype(int)

            if len(binned.shape) > 2:
                raise ValueError('Multiple bins are being calculated per trial,'
                                 'may be due to floating point representation error.'
                                 'Check window.')
            msub_binned = binned - np.mean(binned, axis=0)
            fit_result = dut.regress_target(msub_tvec, msub_binned, estimator,
                                            hyperparam_grid=HPARAM_GRID,
                                            save_binned=SAVE_BINNED)
            pseudo_results = []
            for _ in tqdm(range(N_PSEUDO), desc='Pseudo num: ', leave=False):
                pseudosess = generate_pseudo_session(trialsdf)
                pseudo_tvec = dut.compute_target(TARGET, subject, subjeids, eid,
                                                 MODELFIT_PATH,
                                                 modeltype=MODEL, beh_data=pseudosess,
                                                 no_unbias=NO_UNBIAS, one=one)[mask]
                msub_pseudo_tvec = pseudo_tvec - np.mean(pseudo_tvec)
                pseudo_result = dut.regress_target(msub_pseudo_tvec, msub_binned, estimator,
                                                   hyperparam_grid=HPARAM_GRID)
                pseudo_results.append(pseudo_result)
            filenames.append(save_region_results(fit_result, pseudo_results, subject,
                                                 eid, probe, region, N_units))

    return filenames


# %% Generate cluster interface and map eids to workers via dask.distributed.Client
sessdf = dut.query_sessions(selection=SESS_CRITERION)
sessdf = sessdf.sort_values('subject').set_index(['subject', 'eid'])

N_CORES = 2
cluster = SLURMCluster(cores=N_CORES, memory='12GB', processes=1, queue="shared-cpu",
                       walltime="01:15:00", log_directory='/home/gercek/dask-worker-logs',
                       interface='ib0',
                       extra=["--lifetime", "70m", "--lifetime-stagger", "4m"],
                       job_cpu=N_CORES, env_extra=[f'export OMP_NUM_THREADS={N_CORES}',
                                                   f'export MKL_NUM_THREADS={N_CORES}',
                                                   f'export OPENBLAS_NUM_THREADS={N_CORES}'])
cluster.adapt(minimum_jobs=0, maximum_jobs=200)
client = Client(cluster)


filenames = []
for eid in sessdf.index.unique(level='eid'):
    fns = client.submit(fit_eid, eid)
    filenames.append(fns)
# WAIT FOR COMPUTATION TO FINISH BEFORE MOVING ON
# %% Collate results into master dataframe and save
tmp = [x.result() for x in filenames if x.status == 'finished']
finished = []
for fns in tmp:
    finished.extend(fns)

indexers = ['subject', 'eid', 'probe', 'region']
resultslist = []
for fn in finished:
    fo = open(fn, 'rb')
    result = pickle.load(fo)
    fo.close()
    tmpdict = {**{x: result[x] for x in indexers},
               'baseline': result['fit']['Rsquared_test'],
               **{f'run{i}': result['pseudosessions'][i]['Rsquared_test']
                  for i in range(N_PSEUDO)}}
    resultslist.append(tmpdict)
resultsdf = pd.DataFrame(resultslist).set_index(indexers)

estimatorstr = strlut[ESTIMATOR]
fn = '_'.join([DATE, 'decode', TARGET,
              dut.modeldispatcher[MODEL] if TARGET in ['prior', 'prederr'] else 'task',
              estimatorstr, 'align', ALIGN_TIME, str(N_PSEUDO), 'pseudosessions']) + '.parquet'
metadata_df = pd.Series({'filename': fn, **fit_metadata})
metadata_fn = '.'.join([fn.split('.')[0], 'metadata', 'pkl'])
resultsdf.to_parquet(fn)
metadata_df.to_pickle(metadata_fn)
