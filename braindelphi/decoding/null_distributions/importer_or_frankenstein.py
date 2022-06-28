from braindelphi.decoding.functions.nulldistributions import generate_null_distribution_session
import pandas as pd
from braindelphi.decoding.settings import kwargs
from braindelphi.params import CACHE_PATH, IMPOSTER_SESSION_PATH
from braindelphi.decoding.functions.utils import load_metadata
import pickle
from behavior_models.models.utils import format_data as format_data_mut
from behavior_models.models.utils import format_input as format_input_mut
from braindelphi.decoding.functions.process_targets import optimal_Bayesian
from braindelphi.decoding.functions.process_targets import check_bhv_fit_exists
from brainbox.task.closed_loop import generate_pseudo_session
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import scipy
from braindelphi.params import out_dir


# import most recent cached data
bwmdf, _ = load_metadata(CACHE_PATH.joinpath('*_%s_metadata.pkl' % kwargs['neural_dtype']).as_posix())
unique_eids = bwmdf['dataset_filenames'].eid.unique()
nb_nulls = 100

def get_repBias(df):
    return (df.choice[1:].values == df.choice[:-1].values).mean()

def get_prevActionCorrelation(df):
    return pearsonr(df.choice[1:].values, df.choice[:-1].values)[0]

def acf(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, length)])

def monoExp(x, decay):
    return np.exp(-decay * x)

def get_timescale(df):
    result = acf(df.choice.values)
    params, cv = scipy.optimize.curve_fit(monoExp, np.arange(result.size), result, [0])
    return np.log(0.5) / np.log(params[0])

def get_perf_at_contrast(df, contrast):
    contrasts = np.where(np.isnan(df.contrastLeft), df.contrastRight, df.contrastLeft)
    return np.nanmean((df.feedbackType[contrasts == contrast] + 1) * 0.5)

# loop over eids
frankenstein_dfs_all = []
imposter_dfs_all = []
psychometric_dfs_all = []
for i_eid, eid in tqdm(enumerate(unique_eids)):

    pid = bwmdf['dataset_filenames'][bwmdf['dataset_filenames'].eid == eid]
    metadata = pickle.load(open(out_dir.joinpath('cache', pid.meta_file.iloc[0].as_posix().split('cache/')[-1]), 'rb'))
    regressors = pickle.load(open(out_dir.joinpath('cache', pid.reg_file.iloc[0].as_posix().split('cache/')[-1]), 'rb'))

    metadata['eids_train'] = ([metadata['eid']] if 'eids_train' not in metadata.keys()
                               else metadata['eids_train'])
    kwargs['model_parameters'] = None

    trials_df, neural_dict = regressors['trials_df'], regressors

    # train if does not existe
    if kwargs['model'] != optimal_Bayesian:
        side, stim, act, _ = format_data_mut(trials_df)
        stimuli, actions, stim_side = format_input_mut([stim], [act], [side])
        behmodel = kwargs['model'](kwargs['behfit_path'], np.array(metadata['eids_train']), metadata['subject'],
                                   actions, stimuli, stim_side)
        istrained, _ = check_bhv_fit_exists(metadata['subject'], kwargs['model'], metadata['eids_train'],
                                            kwargs['behfit_path'], modeldispatcher=kwargs['modeldispatcher'])
        if not istrained:
            behmodel.load_or_train(remove_old=False)

    frankenstein_dfs = []
    psychometric_dfs = []
    imposter_dfs = []

    for i_null in range(nb_nulls):
        # frankenstein df
        kwargs['use_imposter_session'] = False
        kwargs['imposterdf'] = None
        frankenstein_df = generate_null_distribution_session(trials_df, metadata, **kwargs)
        frankenstein_df['feedbackType'] = np.where(frankenstein_df.choice == frankenstein_df.stim_side, 1, -1)
        frankenstein_dfs.append(frankenstein_df)

        # importer df
        kwargs['use_imposter_session'] = True
        kwargs['imposterdf'] = pd.read_parquet(IMPOSTER_SESSION_PATH.joinpath('imposterSessions_beforeRecordings.pqt'))
        imposter_df = generate_null_distribution_session(trials_df, metadata, **kwargs)
        imposter_dfs.append(imposter_df)

        # psychometric df
        psychometric_df = generate_pseudo_session(trials_df, generate_choices=True)
        psychometric_df['feedbackType'] = np.where(psychometric_df.choice == psychometric_df.stim_side, 1, -1)
        psychometric_dfs.append(psychometric_df)

    frankenstein_dfs_all.append(frankenstein_dfs)
    imposter_dfs_all.append(imposter_dfs)
    psychometric_dfs_all.append(psychometric_dfs)

# compute metrics
performances_zero_contrast = np.zeros([unique_eids.size, nb_nulls + 1, 3])
performances_full_contrast = np.zeros([unique_eids.size, nb_nulls + 1, 3])
repetition_bias = np.zeros([unique_eids.size, nb_nulls + 1, 3])
action_correlation = np.zeros([unique_eids.size, nb_nulls + 1, 3])
timescale = np.zeros([unique_eids.size, nb_nulls + 1, 3])

for i_eid, eid in tqdm(enumerate(unique_eids)):

    pid = bwmdf['dataset_filenames'][bwmdf['dataset_filenames'].eid == eid]
    metadata = pickle.load(open(pid.meta_file.iloc[0], 'rb'))
    regressors = pickle.load(open(pid.reg_file.iloc[0], 'rb'))
    trials_df = regressors['trials_df']

    for i_null in range(nb_nulls):
        performances_zero_contrast[i_eid, i_null] = [get_perf_at_contrast(k[i_eid][i_null], contrast=0) for k in
                                                     [frankenstein_dfs_all, imposter_dfs_all, psychometric_dfs_all]]
        performances_full_contrast[i_eid, i_null] = [get_perf_at_contrast(k[i_eid][i_null], contrast=1) for k in
                                                     [frankenstein_dfs_all, imposter_dfs_all, psychometric_dfs_all]]
        repetition_bias[i_eid, i_null] = [get_repBias(k[i_eid][i_null]) for k in
                                                     [frankenstein_dfs_all, imposter_dfs_all, psychometric_dfs_all]]
        action_correlation[i_eid, i_null] = [get_prevActionCorrelation(k[i_eid][i_null])  for k in
                                                     [frankenstein_dfs_all, imposter_dfs_all, psychometric_dfs_all]]
        timescale[i_eid, i_null] = [get_timescale(k[i_eid][i_null]) for k in
                                                     [frankenstein_dfs_all, imposter_dfs_all, psychometric_dfs_all]]

    performances_full_contrast[i_eid, -1] = get_perf_at_contrast(trials_df, contrast=1)
    performances_zero_contrast[i_eid, -1] = get_perf_at_contrast(trials_df, contrast=0)
    repetition_bias[i_eid, -1] = get_repBias(trials_df)
    action_correlation[i_eid, -1] = get_prevActionCorrelation(trials_df)
    timescale[i_eid, -1] = get_timescale(trials_df)

pvalues_timescale = np.minimum(
    (timescale[:i_eid][:, -1:] > timescale[:i_eid][:, :-1]).mean(axis=1),
    (timescale[:i_eid][:, -1:] < timescale[:i_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_action_correlation = np.minimum(
    (action_correlation[:i_eid][:, -1:] > action_correlation[:i_eid][:, :-1]).mean(axis=1),
    (action_correlation[:i_eid][:, -1:] < action_correlation[:i_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_full_contrast = np.minimum(
    (performances_full_contrast[:i_eid][:, -1:] > performances_full_contrast[:i_eid][:, :-1]).mean(axis=1),
    (performances_full_contrast[:i_eid][:, -1:] < performances_full_contrast[:i_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_zero_contrast = np.minimum(
    (performances_zero_contrast[:i_eid][:, -1:] > performances_zero_contrast[:i_eid][:, :-1]).mean(axis=1),
    (performances_zero_contrast[:i_eid][:, -1:] < performances_zero_contrast[:i_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_rep_bias = np.minimum(
    (repetition_bias[:i_eid][:, -1:] > repetition_bias[:i_eid][:, :-1]).mean(axis=1),
    (repetition_bias[:i_eid][:, -1:] < repetition_bias[:i_eid][:, :-1]).mean(axis=1)
) * 2
