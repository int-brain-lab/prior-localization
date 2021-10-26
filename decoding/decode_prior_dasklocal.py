import os
import pickle
import numpy as np
import pandas as pd
import decoding_utils as dut
import brainbox.io.one as bbone
import sklearn.linear_model as sklm
from pathlib import Path
from datetime import date
from one.api import ONE
from models.expSmoothing_prevAction import expSmoothing_prevAction
from brainbox.singlecell import calculate_peths
from dask.delayed import delayed
from dask.distributed import Client

one = ONE()

# %% Run param definitions

SESS_CRITERION = 'aligned-behavior'
TARGET = 'prior'
MODEL = expSmoothing_prevAction
MODELFIT_PATH = Path('/home/berk/Documents/Projects/prior-localization/results/inference/')
OUTPUT_PATH = Path('/home/berk/Documents/Projects/prior_localization/results/decoding/')
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (0, 0.1)
ESTIMATOR = sklm.Lasso
N_PSEUDO = 200
DATE = str(date.today())


# %% Define helper functions to make main loop readable
def save_region_results(fit_result, pseudo_results, subject, eid, probe, region):
    subjectfolder = OUTPUT_PATH.joinpath(subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    for folder in [subjectfolder, eidfolder, probefolder]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    fn = '_'.join(DATE, region) + '.pkl'
    fw = open(fn, 'wb')
    outdict = {'fit': fit_result, 'pseudosessions': pseudo_results,
               'subject': subject, 'eid': eid, 'probe': probe, 'region': region}
    pickle.dump(outdict, fw)
    fw.close()
    return probefolder.joinpath(fn)


@delayed
def comp_targ_delayed(subject, subjeids, eid):
    return dut.compute_target(TARGET, subject, subjeids, eid, str(MODELFIT_PATH),
                              modeltype=MODEL, one=one)


@delayed
def trdf_delayed(eid):
    return bbone.load_trials_df(eid, one=one)


@delayed
def 
# %% Check if fits have been run on a per-subject basis
sessdf = dut.query_sessions(selection=SESS_CRITERION)
sessdf = sessdf.sort_values('subject').set_index(['subject', 'eid'])

client = Client()
filenames = []
for eid in sessdf.index.unique(level='eid'):
    subject = sessdf.xs(eid, level='eid').index[0]
    subjeids = sessdf.xs(subject, level='subject').index.unique()

    tvec = client.submit(dut.compute_target,
                         TARGET, subject, subjeids, eid, str(MODELFIT_PATH),
                         modeltype=MODEL, one=one)
    trialsdf = client.submit(bbone.load_trials_df, eid, one=one)
    for probe in sessdf.loc[subject, eid, :].probe:
        spikes, clusters, _ = client.submit(bbone.load_spike_sorting_with_channel, eid,
                                            one=one,
                                            probe=probe,
                                            aligned=True)
        beryl_reg = client.submit(dut.remap_region, clusters[probe].atlas_id)
        regions = client.submit(np.unique, beryl_reg)
        for region in regions:
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
            fit_result = dut.regress_target(msub_tvec, msub_binned, ESTIMATOR())
            pseudo_results = []
            for _ in range(N_PSEUDO):
                pseudo_tvec = dut.compute_target(TARGET, subject, subjeids, eid,
                                                 str(MODELFIT_PATH),
                                                 modeltype=MODEL, pseudo=True, one=one)
                msub_pseudo_tvec = pseudo_tvec - np.mean(pseudo_tvec)
                pseudo_result = dut.regress_target(msub_pseudo_tvec, msub_binned, ESTIMATOR())
                pseudo_results.append(pseudo_result)
            filenames.append(save_region_results(fit_result, pseudo_results, subject,
                                                 eid, probe, region))
            break
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
