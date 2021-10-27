import os
import pickle
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
from brainbox.task.closed_loop import generate_pseudo_session
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

one = ONE()

strlut = {sklm.Lasso: 'Lasso',
          sklm.Ridge: 'Ridge',
          sklm.LinearRegression: 'PureLinear',
          sklm.LogisticRegression: 'Logistic'}

# %% Run param definitions

SESS_CRITERION = 'aligned-behavior'
TARGET = 'signcont'
MODEL = expSmoothing_prevAction
MODELFIT_PATH = '/home/gercek/Projects/prior-localization/results/inference/'
OUTPUT_PATH = '/home/gercek/scratch/results/decoding/'
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (0, 0.1)
ESTIMATOR = sklm.Lasso
N_PSEUDO = 200
MIN_UNITS = 5
DATE = str(date.today())

HPARAM_GRID = {'alpha': np.array([0.001, 0.01, 0.1])}

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
    'date': DATE,
    'hyperparameter_grid': HPARAM_GRID,
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
    subject = sessdf.xs(eid, level='eid').index[0]
    subjeids = sessdf.xs(subject, level='subject').index.unique()
    brainreg = dut.BrainRegions()
    behavior_data = mut.load_session(eid, one=one)
    try:
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, MODELFIT_PATH,
                                  modeltype=MODEL, beh_data=behavior_data, one=one)
    except ValueError:
        print('Model not fit.')
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, MODELFIT_PATH,
                                  modeltype=MODEL, one=one)

    msub_tvec = tvec - np.mean(tvec)
    trialsdf = bbone.load_trials_df(eid, one=one)
    filenames = []
    for probe in sessdf.loc[subject, eid, :].probe:
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid,
                                                                    one=one,
                                                                    probe=probe,
                                                                    aligned=True)
        beryl_reg = dut.remap_region(clusters[probe].atlas_id, br=brainreg)
        regions = np.unique(beryl_reg)
        for region in regions:
            reg_clu = np.argwhere(beryl_reg == region).flatten()
            N_units = len(reg_clu)
            if N_units < MIN_UNITS:
                continue
            _, binned = calculate_peths(spikes[probe].times, spikes[probe].clusters, reg_clu,
                                        trialsdf[ALIGN_TIME], pre_time=TIME_WINDOW[0],
                                        post_time=TIME_WINDOW[1],
                                        bin_size=TIME_WINDOW[1] - TIME_WINDOW[0], smoothing=0,
                                        return_fr=False)
            binned = binned.squeeze()
            if len(binned.shape) > 2:
                raise ValueError('Multiple bins are being calculated per trial,'
                                 'may be due to floating point representation error.'
                                 'Check window.')
            msub_binned = binned - np.mean(binned, axis=0)
            fit_result = dut.regress_target(msub_tvec, msub_binned, ESTIMATOR(),
                                            hyperparam_grid=HPARAM_GRID)
            pseudo_results = []
            for _ in range(N_PSEUDO):
                pseudosess = generate_pseudo_session(trialsdf)
                pseudo_tvec = dut.compute_target(TARGET, subject, subjeids, eid,
                                                 MODELFIT_PATH,
                                                 modeltype=MODEL, beh_data=pseudosess, one=one)
                msub_pseudo_tvec = pseudo_tvec - np.mean(pseudo_tvec)
                pseudo_result = dut.regress_target(msub_pseudo_tvec, msub_binned, ESTIMATOR(),
                                                   hyperparam_grid=HPARAM_GRID)
                pseudo_results.append(pseudo_result)
            filenames.append(save_region_results(fit_result, pseudo_results, subject,
                                                 eid, probe, region, N_units))

    return filenames


# %% Generate cluster interface and map eids to workers via dask.distributed.Client
sessdf = dut.query_sessions(selection=SESS_CRITERION)
sessdf = sessdf.sort_values('subject').set_index(['subject', 'eid'])

N_CORES = 2
cluster = SLURMCluster(cores=N_CORES, memory='32GB', processes=1, queue="shared-cpu",
                       walltime="03:00:00", log_directory='/home/gercek/dask-worker-logs',
                       interface='ib0',
                       extra=["--lifetime", "3h", "--lifetime-stagger", "4m"],
                       job_cpu=N_CORES, env_extra=[f'export OMP_NUM_THREADS={N_CORES}',
                                                   f'export MKL_NUM_THREADS={N_CORES}',
                                                   f'export OPENBLAS_NUM_THREADS={N_CORES}'])
cluster.adapt(minimum_jobs=0, maximum_jobs=200)
client = Client(cluster)


filenames = []
for eid in sessdf.index.unique(level='eid'):
    fns = client.submit(fit_eid, eid)
    filenames.append(fns)

# %% Collate results into master dataframe and save
indexers = ['subject', 'eid', 'probe', 'region']
resultslist = []
for fn in filenames:
    fo = open(fn, 'rb')
    result = pickle.load(fo)
    fo.close()
    tmpdict = {**{x: result[x] for x in indexers},
               'baseline': result['fit']['r2'],
               **{f'run{i}': result['pseudosessions'][i]['score'] for i in range(N_PSEUDO)}}
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
