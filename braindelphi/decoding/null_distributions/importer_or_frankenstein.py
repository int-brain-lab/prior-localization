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
nb_nulls = 100

def get_repBias(df):
    return (df.choice[1:].values == df.choice[:-1].values).mean()

def get_prevActionCorrelation(df):
    return pearsonr(df.choice[1:].values, df.choice[:-1].values)[0]

def acf(x, length=100):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])

def monoExp(x, decay):
    return np.exp(-decay * x)

def get_timescale(df):
    result = acf(df.choice.values)
    params, cv = scipy.optimize.curve_fit(monoExp, np.arange(result.size), result, [0])
    return np.log(0.5) / np.log(params[0])

def get_pswitch(df):
    pswitch = (df.choice.values[1:] != df.choice.values[:-1])
    contrasts = np.where(np.isnan(df.contrastLeft), df.contrastRight, df.contrastLeft)
    return pswitch[(contrasts[1:] == 0) * (df.feedbackType[:-1] == -1)].mean()

def get_perf_at_contrast(df, contrast):
    contrasts = np.where(np.isnan(df.contrastLeft), df.contrastRight, df.contrastLeft)
    return np.nanmean((df.feedbackType[contrasts == contrast] + 1) * 0.5)

def get_perf_at_0contrast_afternegative(df):
    contrasts = np.where(np.isnan(df.contrastLeft), df.contrastRight, df.contrastLeft)
    return np.nanmean((df.feedbackType.values[1:][(contrasts[1:] == 0) * (df.feedbackType[:-1] == -1)] + 1) * 0.5)

def get_bias(df):
    contrasts = np.where(np.isnan(df.contrastLeft), df.contrastRight, df.contrastLeft)
    return np.nanmean(df.choice.values[contrasts == 0])

from scipy import fftpack
def get_frequency(df):
    x = df.choice.values
    X = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x))
    return np.mean(np.abs(X[freqs > 0])[np.argmin(np.abs(1./freqs[freqs > 0][None] - 100/1.15**np.arange(12)[:,None]), axis=1)])

from scipy.stats import entropy
def get_delta_entropy(df):
    choice = df.choice.values
    return entropy(pk = [(choice == -1).mean(), (choice == 1).mean()])

from sklearn.linear_model import LogisticRegression
def get_predictability(df):
    test_id = np.random.choice(np.arange(df.choice.size), size=int(df.choice.size/5), replace=False)
    train_id = np.delete(np.arange(df.choice.size), test_id)
    prediction = np.sign(np.mean(df.choice.values[train_id]))
    return (prediction == df.choice.values[test_id]).mean()

"""
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
"""

import dask
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

@dask.delayed
def delayed_null_distribution(trials_df, metadata, kwargs, type):
    null_distributions_df = []
    for _ in range(nb_nulls):
        if type == 0:
            # frankenstein df
            kwargs['use_imposter_session'] = False
            kwargs['imposterdf'] = None
            null_df = generate_null_distribution_session(trials_df, metadata, **kwargs)
            null_df['feedbackType'] = np.where(null_df.choice == null_df.stim_side, 1, -1)
            null_distributions_df.append(null_df)
        elif type == 1:
            kwargs['use_imposter_session'] = True
            kwargs['imposterdf'] = pd.read_parquet(
                IMPOSTER_SESSION_PATH.joinpath('imposterSessions_beforeRecordings.pqt'))
            null_df = generate_null_distribution_session(trials_df, metadata, **kwargs)
            null_distributions_df.append(null_df)
        else:
            null_df = generate_pseudo_session(trials_df, generate_choices=True)
            null_distributions_df.append(null_df)
    return null_distributions_df

unique_eids = bwmdf['dataset_filenames'].eid.unique()
frankenstein_dfs_all_futures = []
imposter_dfs_all_futures = []
psychometric_dfs_all_futures = []
for i_eid, eid in tqdm(enumerate(unique_eids)):
    pid = bwmdf['dataset_filenames'][bwmdf['dataset_filenames'].eid == eid]
    metadata = pickle.load(open(out_dir.joinpath('cache', pid.meta_file.iloc[0].as_posix().split('cache/')[-1]), 'rb'))
    regressors = pickle.load(open(out_dir.joinpath('cache', pid.reg_file.iloc[0].as_posix().split('cache/')[-1]), 'rb'))
    metadata['eids_train'] = ([metadata['eid']] if 'eids_train' not in metadata.keys()
                               else metadata['eids_train'])
    kwargs['model_parameters'] = None
    trials_df, neural_dict = regressors['trials_df'], regressors

    istrained, _ = check_bhv_fit_exists(metadata['subject'], kwargs['model'], metadata['eids_train'],
                                        kwargs['behfit_path'], modeldispatcher=kwargs['modeldispatcher'])

    if istrained:
        frankenstein_dfs_all_futures.append(delayed_null_distribution(trials_df, metadata, kwargs, 0))
        imposter_dfs_all_futures.append(delayed_null_distribution(trials_df, metadata, kwargs, 1))
        psychometric_dfs_all_futures.append(delayed_null_distribution(trials_df, metadata, kwargs, 2))


from dask.distributed import LocalCluster
cluster = LocalCluster()
cluster.scale(20)
client = Client(cluster)

frankenstein_dfs_all = [client.compute(future) for future in frankenstein_dfs_all_futures]
imposter_dfs_all = [client.compute(future) for future in imposter_dfs_all_futures]
psychometric_dfs_all = [client.compute(future) for future in psychometric_dfs_all_futures]

frankenstein_df_eids = [df.result() if df.status == "finished" else None for df in frankenstein_dfs_all]
imposter_df_eids = [df.result() if df.status == "finished" else None for df in imposter_dfs_all]
psychometric_df_eids = [df.result() if df.status == "finished" else None for df in psychometric_dfs_all]

frankenstein_df_status = [df.status for df in frankenstein_dfs_all]
imposter_df_status = [df.status for df in imposter_dfs_all]
psychometric_df_status = [df.status for df in psychometric_dfs_all]

import pickle
#pickle.dump([frankenstein_df_eids, imposter_df_eids, psychometric_df_eids], open('null_dist_eids.pkl', 'rb'))
#pickle.dump([frankenstein_df_status, imposter_df_status, psychometric_df_status], open('null_dist_statuses.pkl', 'rb'))

[frankenstein_df_eids, imposter_df_eids, psychometric_df_eids] = pickle.load(open('null_dist_eids.pkl', 'rb'))
[frankenstein_df_status, imposter_df_status, psychometric_df_status] = pickle.load(open('null_dist_statuses.pkl', 'rb'))

# compute metrics
performances_zero_contrast = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan
performances_full_contrast = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan
repetition_bias = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan
action_correlation = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan
p_switch = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan
timescale = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan
bias = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan
perf_0_contast_afternegative = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan
delta_entropy = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan
predictability = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan
frequencies = np.zeros([unique_eids.size, nb_nulls + 1, 3]) + np.nan

for i_eid, eid in tqdm(enumerate(unique_eids)):

    pid = bwmdf['dataset_filenames'][bwmdf['dataset_filenames'].eid == eid]
    metadata = pickle.load(open(out_dir.joinpath('cache', pid.meta_file.iloc[0].as_posix().split('cache/')[-1]), 'rb'))
    metadata['eids_train'] = ([metadata['eid']] if 'eids_train' not in metadata.keys()
                               else metadata['eids_train'])
    kwargs['model_parameters'] = None
    regressors = pickle.load(open(out_dir.joinpath('cache', pid.reg_file.iloc[0].as_posix().split('cache/')[-1]), 'rb'))
    trials_df = regressors['trials_df']

    istrained, _ = check_bhv_fit_exists(metadata['subject'], kwargs['model'], metadata['eids_train'],
                                        kwargs['behfit_path'], modeldispatcher=kwargs['modeldispatcher'])
    if not istrained and (frankenstein_df_status[i_eid] != 'finished' or imposter_df_status[i_eid] != 'finished'
    or psychometric_df_status[i_eid] != 'finished'):
        continue

    try:
        frankenstein_df_eid = frankenstein_df_eids[i_eid]
        imposter_df_eid = imposter_df_eids[i_eid]
        psychometric_df_eid = psychometric_df_eids[i_eid]

        for i_null in range(nb_nulls):
            frequencies[i_eid, i_null] = [get_frequency(k[i_null]) for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]

            """
            performances_full_contrast[i_eid, i_null] = [get_perf_at_contrast(k[i_null], contrast=1) for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]

            bias[i_eid, i_null] = [get_bias(k[i_null]) for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]
            delta_entropy[i_eid, i_null] = [get_delta_entropy(k[i_null]) for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]

            predictability[i_eid, i_null] = [get_predictability(k[i_null]) for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]

            performances_zero_contrast[i_eid, i_null] = [get_perf_at_contrast(k[i_null], contrast=0) for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]

            repetition_bias[i_eid, i_null] = [get_repBias(k[i_null]) for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]
            action_correlation[i_eid, i_null] = [get_prevActionCorrelation(k[i_null])  for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]
            timescale[i_eid, i_null] = [get_timescale(k[i_null]) for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]
            p_switch[i_eid, i_null] = [get_pswitch(k[i_null]) for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]
            perf_0_contast_afternegative[i_eid, i_null] = [get_perf_at_0contrast_afternegative(k[i_null]) for k in
                                                         [frankenstein_df_eid, imposter_df_eid, psychometric_df_eid]]
            """

        performances_full_contrast[i_eid, -1] = get_perf_at_contrast(trials_df, contrast=1)
        performances_zero_contrast[i_eid, -1] = get_perf_at_contrast(trials_df, contrast=0)
        repetition_bias[i_eid, -1] = get_repBias(trials_df)
        action_correlation[i_eid, -1] = get_prevActionCorrelation(trials_df)
        timescale[i_eid, -1] = get_timescale(trials_df)
        p_switch[i_eid, -1] = get_pswitch(trials_df)
        bias[i_eid, -1] = get_bias(trials_df)
        perf_0_contast_afternegative[i_eid, -1] = get_perf_at_0contrast_afternegative(trials_df)
        delta_entropy[i_eid, -1] = get_delta_entropy(trials_df)
        predictability[i_eid, -1] = get_predictability(trials_df)
        frequencies[i_eid, -1] = get_frequency(trials_df)
    except Exception as e:
        print(e)
        pass

not_nan_eid = ~np.any(np.isnan(frequencies), axis=(1,2))

pvalues_timescale = np.minimum(
    (timescale[not_nan_eid][:, -1:] >= timescale[not_nan_eid][:, :-1]).mean(axis=1),
    (timescale[not_nan_eid][:, -1:] <= timescale[not_nan_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_action_correlation = np.minimum(
    (action_correlation[not_nan_eid][:, -1:] >= action_correlation[not_nan_eid][:, :-1]).mean(axis=1),
    (action_correlation[not_nan_eid][:, -1:] <= action_correlation[not_nan_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_full_contrast = np.minimum(
    (performances_full_contrast[not_nan_eid][:, -1:] >= performances_full_contrast[not_nan_eid][:, :-1]).mean(axis=1),
    (performances_full_contrast[not_nan_eid][:, -1:] <= performances_full_contrast[not_nan_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_zero_contrast = np.minimum(
    (performances_zero_contrast[not_nan_eid][:, -1:] >= performances_zero_contrast[not_nan_eid][:, :-1]).mean(axis=1),
    (performances_zero_contrast[not_nan_eid][:, -1:] <= performances_zero_contrast[not_nan_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_rep_bias = np.minimum(
    (repetition_bias[not_nan_eid][:, -1:] >= repetition_bias[not_nan_eid][:, :-1]).mean(axis=1),
    (repetition_bias[not_nan_eid][:, -1:] <= repetition_bias[not_nan_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_pswitch = np.minimum(
    (p_switch[not_nan_eid][:, -1:] >= p_switch[not_nan_eid][:, :-1]).mean(axis=1),
    (p_switch[not_nan_eid][:, -1:] <= p_switch[not_nan_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_bias = np.minimum(
    (bias[not_nan_eid][:, -1:] >= bias[not_nan_eid][:, :-1]).mean(axis=1),
    (bias[not_nan_eid][:, -1:] <= bias[not_nan_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_perf_0_contast_afternegative = np.minimum(
    (perf_0_contast_afternegative[not_nan_eid][:, -1:] >= perf_0_contast_afternegative[not_nan_eid][:, :-1]).mean(axis=1),
    (perf_0_contast_afternegative[not_nan_eid][:, -1:] <= perf_0_contast_afternegative[not_nan_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_delta_entropy = np.minimum(
    (delta_entropy[not_nan_eid][:, -1:] >= delta_entropy[not_nan_eid][:, :-1]).mean(axis=1),
    (delta_entropy[not_nan_eid][:, -1:] <= delta_entropy[not_nan_eid][:, :-1]).mean(axis=1)
) * 2

pvalues_frequencies = np.minimum(
    (frequencies[not_nan_eid][:, -1:] >= frequencies[not_nan_eid][:, :-1]).mean(axis=1),
    (frequencies[not_nan_eid][:, -1:] <= frequencies[not_nan_eid][:, :-1]).mean(axis=1)
) * 2


plim = 0.015
print('timescale')
print((pvalues_timescale < plim).mean(0))
print('action correlation')
print((pvalues_action_correlation < plim).mean(0))
print('perf full contrast')
print((pvalues_full_contrast < plim).mean(0))
print('perf zero contrast')
print((pvalues_zero_contrast < plim).mean(0))
print('rep bias')
print((pvalues_rep_bias < plim).mean(0))
print('prob switch on 0 contrast after negative feedback')
print((pvalues_pswitch < plim).mean(0))
print('bias')
print((pvalues_bias < plim).mean(0))
print('perf_0_contast_afternegative')
print((pvalues_perf_0_contast_afternegative < plim).mean(0))
print('delta_entropy')
print((pvalues_delta_entropy < plim).mean(0))
print('frequencies')
print((pvalues_frequencies < plim).mean(0))

# predictability

quantile_predictability = np.quantile(predictability[not_nan_eid, :-1], q=0.95, axis=1)
print(np.median(quantile_predictability, axis=0))

import pickle
p = pickle.load(open('quantile_predictability.pkl', 'rb'))[:, :2]

from matplotlib import pyplot as plt
plt.figure()
plt.hist(p[:,::-1], label=['imposter', 'frankenstein'])
plt.legend()
plt.xlabel('95% quantile accuracy of null distribution')
plt.ylabel('unnormalized density')
plt.title('95% quantile accuracy of null distribution for N=140 sessions')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.savefig('figures/95_quantile_accuracy.pdf')
plt.show()

plt.figure()
plt.hist(predictability[not_nan_eid, :-1][0,:,1], label='imposter', alpha=0.5)
plt.legend()
plt.xlabel('test accuracy of action prediction')
plt.ylabel('unnormalized density')
plt.title('Action prediction: test accuracy')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.savefig('figures/action_prediction_example.pdf')
plt.show()

plt.figure()
plt.hist(predictability[not_nan_eid, :-1][0,:,1], label='imposter', alpha=0.5)
quantile = np.quantile(predictability[not_nan_eid, :-1][0,:,1], 0.95)
plt.plot([quantile, quantile], plt.gca().get_ylim())
plt.legend()
plt.xlabel('test accuracy of action prediction')
plt.ylabel('unnormalized density')
plt.title('Action prediction: test accuracy')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.savefig('figures/action_prediction_example_1.pdf')
plt.show()


plt.figure()
plt.hist(action_correlation[0, :-1, 0], alpha=0.5, label='frankenstein')
plt.plot([action_correlation[0, -1, 0], action_correlation[0, -1, 0]], plt.gca().get_ylim(),
         color='orange', linewidth=2)
plt.legend()
plt.xlabel('action correlations of N=100 frankenstein sessions')
plt.ylabel('unnormalized density')
plt.title('Null distribution of action correlations with frankenstein sessions')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.savefig('figures/action_correlation_frankenstein.pdf')
plt.show()
