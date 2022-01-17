import os
import pickle
import logging
import numpy as np
import pandas as pd
import decoding_utils as dut
import brainbox.io.one as bbone
import sklearn.linear_model as sklm
import models.utils as mut
from datetime import date
from pathlib import Path
from models.expSmoothing_prevAction import expSmoothing_prevAction

from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import generate_pseudo_session
import one.alf.io as alfio

from decoding_stimulus_neurometric_fit import get_neurometric_parameters

try:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, LocalCluster
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
LOCAL = False
if LOCAL:
    DECODING_PATH = Path("/Users/csmfindling/Documents/Postdoc-Geneva/IBL/behavior/prior-localization/decoding")
else:
    DECODING_PATH = Path("/home/users/f/findling/ibl/prior-localization/decoding")

# aligned -> histology was performed by one experimenter
# resolved -> histology was performed by 2-3 experiments
SESS_CRITERION = 'aligned-behavior'  # aligned and behavior
DATE = str(date.today())
ALIGN_TIME = 'goCue_times'
TARGET = 'signcont'  # 'signcont' or 'pLeft'
# NB: if TARGET='signcont', MODEL with define how the neurometric curves will be generated. else MODEL computes TARGET
MODEL = None  # expSmoothing_prevAction  # or dut.modeldispatcher.
TIME_WINDOW = (-0.6, -0.1)  # (0, 0.1)  #
ESTIMATOR = sklm.Lasso  # Must be in keys of strlut above
ESTIMATOR_KWARGS = {'tol': 0.0001, 'max_iter': 10000, 'fit_intercept': True}
N_PSEUDO = 2
N_RUNS = 10
MIN_UNITS = 10
MIN_BEHAV_TRIAS = 400  # default BWM setting
MIN_RT = 0.08  # 0.08  # Float (s) or None
SINGLE_REGION = True  # perform decoding on region-wise or whole brain analysis
NO_UNBIAS = False
SHUFFLE = True
COMPUTE_NEUROMETRIC = True if TARGET == 'signcont' else False
FORCE_POSITIVE_NEURO_SLOPES = False
# NEUROMETRIC_PRIOR_MODEL = expSmoothing_prevAction #'oracle'
# Basically, quality metric on the stability of a single unit. Should have 1 metric per neuron
QC_CRITERIA = 3 / 3  # 3 / 3  # In {None, 1/3, 2/3, 3/3}
NORMALIZE_INPUT = False  # take out mean of the neural activity per unit across trials
NORMALIZE_OUTPUT = False  # take out mean of output to predict
if NORMALIZE_INPUT or NORMALIZE_OUTPUT:
    warnings.warn('This feature has not been tested')
USE_IMPOSTER_SESSION = True  # if false, it uses pseudosessions

BALANCED_WEIGHT = False  # seems to work better with BALANCED_WEIGHT=False
HPARAM_GRID = {'alpha': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
SAVE_BINNED = False  # Debugging parameter, not usually necessary
COMPUTE_NEURO_ON_EACH_FOLD = False  # if True, expect a script that is 5 times slower
ADD_TO_SAVING_PATH = 'imposter_v5'

# session to be excluded (by Olivier Winter)
excludes = [
    'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',  # inconsistent trials object: relaunched task on 31-12-2021
    'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',  # same same
    '7a887357-850a-4378-bd2a-b5bc8bdd3aac',  # same same
    '56b57c38-2699-4091-90a8-aba35103155e',  # load object pickle error
    '09394481-8dd2-4d5c-9327-f2753ede92d7',  # same same
]

# ValueErrors and NotImplementedErrors
if TARGET not in ['signcont', 'pLeft']:
    raise NotImplementedError('this TARGET is not supported yet')

if MODEL not in list(dut.modeldispatcher.keys()):
    raise NotImplementedError('this MODEL is not supported yet')

if COMPUTE_NEUROMETRIC and TARGET != 'signcont':
    raise ValueError('the target should be signcont to compute neurometric curves')

fit_metadata = {
    'criterion': SESS_CRITERION,
    'target': TARGET,
    'model_type': dut.modeldispatcher[MODEL],
    'decoding_path': DECODING_PATH,
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
    'balanced_weight': BALANCED_WEIGHT,
    'force_positive_neuro_slopes': FORCE_POSITIVE_NEURO_SLOPES,
    'compute_neurometric': COMPUTE_NEUROMETRIC,
    'n_runs': N_RUNS,
    'normalize_output': NORMALIZE_OUTPUT,
    'normalize_input': NORMALIZE_INPUT,
    'single_region': SINGLE_REGION,
    'use_imposter_session': USE_IMPOSTER_SESSION
}


# %% Define helper functions for dask workers to use
def save_region_results(fit_result, pseudo_id, subject, eid, probe, region, N,
                        output_path=DECODING_PATH.joinpath('results', 'neural')):
    subjectfolder = Path(output_path).joinpath(subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    for folder in [subjectfolder, eidfolder, probefolder]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    start_tw, end_tw = TIME_WINDOW
    fn = '_'.join([DATE, region, 'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_'),
                   'pseudo_id', str(pseudo_id)]) + '.pkl'
    fw = open(probefolder.joinpath(fn), 'wb')
    outdict = {'fit': fit_result, 'pseudo_id': pseudo_id,
               'subject': subject, 'eid': eid, 'probe': probe, 'region': region, 'N_units': N}
    pickle.dump(outdict, fw)
    fw.close()
    return probefolder.joinpath(fn)


def fit_eid(eid, sessdf, imposterdf, pseudo_id=-1, nb_runs=10, single_region=SINGLE_REGION,
            modelfit_path=DECODING_PATH.joinpath('results', 'behavioral'),
            output_path=DECODING_PATH.joinpath('results', 'neural'), one=None):
    """
    Parameters
    ----------
    single_region: Bool, decoding using region wise or pulled over regions
    eid: eid of session
    sessdf: dataframe of session eid
    pseudo_id: whether to compute a pseudosession or not. if pseudo_id=-1, the true session is considered.
    can not be 0
    nb_runs: nb of independent runs performed. this was added after consequent variability was observed across runs.
    modelfit_path: outputs of behavioral fits
    output_path: outputs of decoding fits
    one: ONE object -- this is not to be used with dask, this option is given for debugging purposes
    """

    if pseudo_id == 0:
        raise ValueError('pseudo id can be -1 (actual session) or strictly greater than 0 (pseudo session)')

    one = ONE(mode='local') if one is None else one
    estimator = ESTIMATOR
    df_insertions = sessdf.loc[sessdf['eid'] == eid]
    subject = df_insertions['subject'].to_numpy()[0]
    subjeids = sessdf.loc[sessdf['subject'] == subject]['eid'].unique()

    brainreg = dut.BrainRegions()
    behavior_data = mut.load_session(eid, one=one)
    try:
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, modelfit_path,
                                  modeltype=MODEL, beh_data=behavior_data,
                                  one=one)
    except ValueError:
        print('Model not fit.')
        tvec = dut.compute_target(TARGET, subject, subjeids, eid, modelfit_path,
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
    mask = mask & (trialsdf.choice != 0)  # take out when mouse doesn't perform any action

    nb_trialsdf = trialsdf[mask]
    msub_tvec = tvec[mask]

    filenames = []
    if len(msub_tvec) <= MIN_BEHAV_TRIAS:
        return filenames

    print(f'Working on eid and on pseudo_id: {eid}, {pseudo_id}')
    across_probes = {'regions': [], 'clusters': [], 'times': [], 'qc_pass': []}
    for i_probe, (_, ins) in tqdm(enumerate(df_insertions.iterrows()), desc='Probe: ', leave=False):
        probe = ins['probe']
        spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
        spikes = alfio.load_object(spike_sorting_path, 'spikes')
        clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
        beryl_reg = dut.remap_region(clusters.atlas_id, br=brainreg)
        qc_pass = (clusters['label'] >= QC_CRITERIA).values
        across_probes['regions'].extend(beryl_reg)
        across_probes['clusters'].extend(spikes.clusters if i_probe == 0 else
                                         (spikes.clusters + max(across_probes['clusters']) + 1))
        across_probes['times'].extend(spikes.times)
        across_probes['qc_pass'].extend(qc_pass)
    across_probes = {k: np.array(v) for k, v in across_probes.items()}
    # warnings.filterwarnings('ignore')
    if single_region:
        regions = [[k] for k in np.unique(across_probes['regions'])]
    else:
        regions = [np.unique(across_probes['regions'])]
    for region in tqdm(regions, desc='Region: ', leave=False):
        reg_mask = np.isin(across_probes['regions'], region)
        reg_clu_ids = np.argwhere(reg_mask & across_probes['qc_pass']).flatten()
        N_units = len(reg_clu_ids)
        if N_units < MIN_UNITS:
            continue
        # or get_spike_count_in_bins
        if np.any(np.isnan(nb_trialsdf[ALIGN_TIME])):
            # if this happens, verify scrub of NaN values in all align times before get_spike_counts_in_bins
            raise ValueError('this should not happen')
        intervals = np.vstack([nb_trialsdf[ALIGN_TIME] + TIME_WINDOW[0],
                               nb_trialsdf[ALIGN_TIME] + TIME_WINDOW[1]]).T
        spikemask = np.isin(across_probes['clusters'], reg_clu_ids)
        regspikes = across_probes['times'][spikemask]
        regclu = across_probes['clusters'][spikemask]
        arg_sortedSpikeTimes = np.argsort(regspikes)
        binned, _ = get_spike_counts_in_bins(regspikes[arg_sortedSpikeTimes],
                                             regclu[arg_sortedSpikeTimes],
                                             intervals)
        msub_binned = binned.T

        if len(msub_binned.shape) > 2:
            raise ValueError('Multiple bins are being calculated per trial,'
                             'may be due to floating point representation error.'
                             'Check window.')

        if pseudo_id > 0:  # create pseudo session when necessary
            if USE_IMPOSTER_SESSION:
                pseudosess = dut.generate_imposter_session(imposterdf, eid, trialsdf)
            else:
                pseudosess = generate_pseudo_session(trialsdf)

            msub_pseudo_tvec = dut.compute_target(TARGET, subject, subjeids, eid,
                                                  modelfit_path, modeltype=MODEL,
                                                  beh_data=pseudosess, one=one)[mask]

        if COMPUTE_NEUROMETRIC:  # compute prior for neurometric curve
            trialsdf_neurometric = nb_trialsdf.reset_index() if (pseudo_id == -1) else \
                pseudosess[mask].reset_index()
            if MODEL is not None:
                blockprob_neurometric = dut.compute_target('pLeft', subject, subjeids, eid, modelfit_path,
                                                           modeltype=MODEL,
                                                           beh_data=trialsdf if pseudo_id == -1 else pseudosess,
                                                           one=one)
                trialsdf_neurometric['blockprob_neurometric'] = np.greater_equal(blockprob_neurometric[mask],
                                                                                 0.5).astype(int)
            else:
                blockprob_neurometric = trialsdf_neurometric['probabilityLeft'].replace(0.2, 0).replace(0.8, 1)
                trialsdf_neurometric['blockprob_neurometric'] = blockprob_neurometric

        fit_results = []
        for i_run in range(nb_runs):
            if pseudo_id == -1:
                fit_result = dut.regress_target(msub_tvec, msub_binned, estimator,
                                                estimator_kwargs=ESTIMATOR_KWARGS,
                                                hyperparam_grid=HPARAM_GRID,
                                                save_binned=SAVE_BINNED, shuffle=SHUFFLE,
                                                balanced_weight=BALANCED_WEIGHT,
                                                normalize_input=NORMALIZE_INPUT,
                                                normalize_output=NORMALIZE_OUTPUT)
            else:
                fit_result = dut.regress_target(msub_pseudo_tvec, msub_binned, estimator,
                                                estimator_kwargs=ESTIMATOR_KWARGS,
                                                hyperparam_grid=HPARAM_GRID,
                                                save_binned=SAVE_BINNED, shuffle=SHUFFLE,
                                                balanced_weight=BALANCED_WEIGHT,
                                                normalize_input=NORMALIZE_INPUT,
                                                normalize_output=NORMALIZE_OUTPUT)
            fit_result['mask'] = mask
            fit_result['pseudo_id'] = pseudo_id
            fit_result['run_id'] = i_run
            # neurometric curve
            if COMPUTE_NEUROMETRIC:
                fit_result['full_neurometric'], fit_result['fold_neurometric'] = \
                    get_neurometric_parameters(fit_result,
                                               trialsdf=trialsdf_neurometric,
                                               one=one,
                                               compute_on_each_fold=COMPUTE_NEURO_ON_EACH_FOLD,
                                               force_positive_neuro_slopes=FORCE_POSITIVE_NEURO_SLOPES)
            else:
                fit_result['full_neurometric'] = None
                fit_result['fold_neurometric'] = None
            fit_results.append(fit_result)

        filenames.append(save_region_results(fit_results, pseudo_id, subject,
                                             eid, 'pulledProbes',
                                             region[0] if single_region else 'allRegions',
                                             N_units, output_path=output_path))

    return filenames


if __name__ == '__main__':
    from decode_prior import fit_eid, save_region_results

    # import cached data
    insdf = pd.read_parquet(DECODING_PATH.joinpath('insertions.pqt'))
    insdf = insdf[insdf.spike_sorting != '']
    eids = insdf['eid'].unique()
    imposterdf = pd.read_parquet(DECODING_PATH.joinpath('imposterSessions.pqt'))

    # create necessary empty directories if not existing
    DECODING_PATH.joinpath('results').mkdir(exist_ok=True)
    DECODING_PATH.joinpath('results', 'behavioral').mkdir(exist_ok=True)
    DECODING_PATH.joinpath('results', 'neural').mkdir(exist_ok=True)

    # Generate cluster interface and map eids to workers via dask.distributed.Client
    if LOCAL:
        cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    else:
        N_CORES = 2
        cluster = SLURMCluster(cores=N_CORES, memory='16GB', processes=1, queue="shared-cpu",
                               walltime="01:15:00",
                               log_directory='/home/users/f/findling/ibl/prior-localization/decoding/dask-worker-logs',
                               interface='ib0',
                               extra=["--lifetime", "60m", "--lifetime-stagger", "10m"],
                               job_cpu=N_CORES, env_extra=[f'export OMP_NUM_THREADS={N_CORES}',
                                                           f'export MKL_NUM_THREADS={N_CORES}',
                                                           f'export OPENBLAS_NUM_THREADS={N_CORES}'])
        cluster.adapt(minimum_jobs=1, maximum_jobs=200)
    client = Client(cluster)
    # todo verify the behavior of scatter
    imposterdf_future = client.scatter(imposterdf)

    # debug
    IMIN = 0
    filenames = []
    for i, eid in enumerate(eids):
        if (i < IMIN or eid in excludes or np.any(insdf[insdf['eid'] == eid]['spike_sorting'] == "")) or \
                (USE_IMPOSTER_SESSION and eid not in imposterdf.eid.values):
            print(f"dud {eid}")
            continue
        print(f"{i}, session: {eid}")
        for pseudo_id in range(N_PSEUDO + 1):
            fns = client.submit(fit_eid,
                                eid=eid,
                                sessdf=insdf,
                                pseudo_id=-1 if pseudo_id == 0 else pseudo_id,
                                nb_runs=N_RUNS,
                                imposterdf=imposterdf_future)
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
print(len([(i, x) for i, x in enumerate(filenames) if x.status == 'error']))
import traceback
tb = failure.traceback()
traceback.print_tb(tb)
"""
# You can also get the traceback from failure.traceback and print via `import traceback` and
# traceback.print_tb()
