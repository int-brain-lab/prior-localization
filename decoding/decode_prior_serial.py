import os
import pickle
import logging
import numpy as np
import pandas as pd
import decoding_utils as dut
import models.utils as mut
import brainbox.io.one as bbone
import sklearn.linear_model as sklm
from pathlib import Path
from datetime import date
from one.api import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction
from brainbox.singlecell import calculate_peths
from brainbox.task.closed_loop import generate_pseudo_session
from tqdm import tqdm


one = ONE()
logger = logging.getLogger('ibllib')
logger.disabled = True
# %% Run param definitions

SESS_CRITERION = 'aligned-behavior'
TARGET = 'prior'
MODEL = expSmoothing_prevAction
MODELFIT_PATH = '/home/berk/Documents/Projects/prior-localization/results/inference/'
OUTPUT_PATH = '/home/berk/Documents/Projects/prior-localization/results/decoding/'
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (0, 0.1)
ESTIMATOR = sklm.Lasso
N_PSEUDO = 200
DATE = str(date.today())

HPARAM_GRID = {'alpha': np.array([0.001, 0.01, 0.1])}


# %% Define helper functions to make main loop readable
def save_region_results(fit_result, pseudo_results, subject, eid, probe, region):
    subjectfolder = Path(OUTPUT_PATH).joinpath(subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    for folder in [subjectfolder, eidfolder, probefolder]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    fn = '_'.join([DATE, region]) + '.pkl'
    fw = open(probefolder.joinpath(fn), 'wb')
    outdict = {'fit': fit_result, 'pseudosessions': pseudo_results,
               'subject': subject, 'eid': eid, 'probe': probe, 'region': region}
    pickle.dump(outdict, fw)
    fw.close()
    return probefolder.joinpath(fn)


# %% Check if fits have been run on a per-subject basis
sessdf = dut.query_sessions(selection=SESS_CRITERION)
sessdf = sessdf.sort_values('subject').set_index(['subject', 'eid'])

filenames = []
for eid in tqdm(sessdf.index.unique(level='eid'), desc='EID: '):
    subject = sessdf.xs(eid, level='eid').index[0]
    subjeids = sessdf.xs(subject, level='subject').index.unique()

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
    for probe in tqdm(sessdf.loc[subject, eid, :].probe, desc='Probe: ', leave=False):
        spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid,
                                                                    one=one,
                                                                    probe=probe,
                                                                    aligned=True)
        beryl_reg = dut.remap_region(clusters[probe].atlas_id)
        regions = np.unique(beryl_reg)
        for region in tqdm(regions, desc='Region: ', leave=False):
            reg_clu = np.argwhere(beryl_reg == region).flatten()
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
            for _ in tqdm(range(N_PSEUDO), desc='Pseudosess: ', leave=False):
                pseudosess = generate_pseudo_session(trialsdf)
                pseudo_tvec = dut.compute_target(TARGET, subject, subjeids, eid,
                                                 MODELFIT_PATH,
                                                 modeltype=MODEL, beh_data=pseudosess, one=one)
                msub_pseudo_tvec = pseudo_tvec - np.mean(pseudo_tvec)
                pseudo_result = dut.regress_target(msub_pseudo_tvec, msub_binned, ESTIMATOR(),
                                                   hyperparam_grid=HPARAM_GRID)
                pseudo_results.append(pseudo_result)
            filenames.append(save_region_results(fit_result, pseudo_results, subject,
                                                 eid, probe, region))

# %% Collate results into master dataframe and save
indexers = ['subject', 'eid', 'probe', 'region']
resultsdf = pd.DataFrame(index=indexers)
for fn in filenames:
    fo = open(fn, 'rb')
    result = pickle.load(fo)
    fo.close()
    tmpdict = {**result.fromkeys(['subject', 'eid', 'probe', 'region']),
               'baseline': result['fit']['score'],
               **{f'run{i}': result['pseudosessions'][i]['score'] for i in range(N_PSEUDO)}}
    tmpdf = pd.DataFrame(tmpdict, index=indexers)
    resultsdf = resultsdf.append(tmpdf)

strlut = {sklm.Lasso: 'Lasso',
          sklm.Ridge: 'Ridge',
          sklm.LinearRegression: 'PureLinear',
          sklm.LogisticRegression: 'Logistic'}
estimatorstr = strlut[ESTIMATOR]
fn = '_'.join([DATE, 'decode', TARGET,
              dut.modeldispatcher[MODEL] if TARGET in ['prior', 'prederr'] else 'task',
              estimatorstr, 'align', ALIGN_TIME, N_PSEUDO, 'pseudosessions']) + '.parquet'
resultsdf.to_parquet(fn)
