# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020
Decode from all brain regions
@author: Guido Meijer
"""

from os.path import join, isfile
import numpy as np
from brainbox.task.closed_loop import generate_pseudo_session, generate_pseudo_stimuli
# , regress removed to access weights BB
from brainbox.population.decode import get_spike_counts_in_bins, sigtest_pseudosessions
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge
from my_functions_Guido import query_sessions, check_trials, combine_layers_cortex, load_trials
from my_functions_Brandon import check_probe
# from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
# from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
import brainbox.io.one as bbone
# from oneibl.one import ONE # changed per Miles' recommendation and Guido 210701
from one.api import ONE
from ibllib.atlas import BrainRegions
from sklearn.metrics import mean_squared_error, r2_score
# from brainbox.numerical import ismember

from decoding_helpers import regress, remap, get_incl_trials, get_target_from_model, getatlassummary

import matplotlib.pyplot as plt

import logging


one = ONE()
br = BrainRegions()

# Settings
TARGET = 'prior'  # 'stimulus' 'prior', 'reward', 'prederr'

TIMING = 'stimulus_onset'  # 'stimulus_onset' 'movement_onset', 'feedback'
PRE_TIME = .6  # .6
POST_TIME = -.2  # -.2

REMOVE_OLD_FIT = False
# 'contrast'#'model_prior_stimside' # must include 'model' to use Charles' behavior model
# with either 'prior' or 'prederr' and either 'stimside' or 'prevaction'
# If not model then signed contrast is used
MIN_NEURONS = 10  # min neurons per region
REGULARIZATION = 'CVL1'
DECODER = 'linear-regression-%s' % REGULARIZATION
VALIDATION = 'kfold'
INCL_NEURONS = 'pass-QC'  # all or pass-QC
# all, aligned, resolved, aligned-behavior or resolved-behavior
INCL_SESSIONS = 'aligned-behavior'
ATLAS = 'beryl-atlas'
NUM_SPLITS = 5
CHANCE_LEVEL = 'other-trials'
ITERATIONS = 100  # for null distribution estimation
MIN_RT = 0.08  # in seconds
EXCL_5050 = True
DATA_PATH = '../../../../../IntBrainLabData/'
SAVE_PATH = './'

# %% Initialize


# Query session list
eids, probes, subjects = query_sessions(
    selection=INCL_SESSIONS, return_subjects=True)


# %% MAIN
decoding_result = pd.DataFrame(
    columns=['subject', 'date', 'eid', 'probe', 'region'])

# Loop over subjects

for i, subject in enumerate(np.unique(subjects)):
    #     if i == 64 or i == 78:
    #         continue
    print('\nStarting subject %s [%d of %d]\n' %
          (subject, i + 1, len(np.unique(subjects))))

    #  load all session data for this subject
    spikes_arr, clusters_arr, channels_arr, trials_arr = [], [], [], []
    stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
    stim_probs_arr = []
    indices_to_skip = []
    for j, eid in enumerate(eids[subjects == subject]):
        # Load in trials vectors
        count = 0
        while count <= 0 and count > -5:  # WTF?
            try:
                spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
                    eid, aligned=True, one=one)
                print('loaded spiking sorting with count', count)
                count = 1

            except:
                logging.info((eid, '_', subject, '_withcount', count))
                count -= 1

        skip_subject = (count < 0)
        trials = load_trials(eid, invert_stimside=True, one=one)
        if skip_subject or (len(trials.keys()) == 0):
            indices_to_skip.append(j)
            continue

        trials = get_incl_trials(trials, TARGET, EXCL_5050, MIN_RT)

        # Check data integrity
#         if check_trials(trials) is False:
#             continue

        spikes_arr.append(spikes)
        clusters_arr.append(clusters)
        channels_arr.append(channels)
        trials_arr.append(trials)
        stimuli_arr.append(trials['signed_contrast'])
        actions_arr.append(trials['choice'])
        stim_probs_arr.append(trials['probabilityLeft'])
        stim_sides_arr.append(trials['stim_side'])
        session_uuids.append(eid)

    session_uuids = np.array(session_uuids)

    print(f'\nLoaded data from {len(session_uuids)} sessions')
    if len(session_uuids) == 0:
        continue

    if TIMING == 'stimulus_onset':
        timing = 'goCue_times'
    elif TIMING == 'movement_onset':
        timing = 'firstMovement_times'
    elif TIMING == 'feedback':
        timing = 'feedback_times'
    else:
        print('no valid timing')
        assert False

    # Target: define functions to retrieve targets, Corresponding trial times
    if ('prior' in TARGET) or ('prederr' in TARGET):
        target, params = get_target_from_model(TARGET, SAVE_PATH, subject,
                                               stimuli_arr, actions_arr, stim_sides_arr, session_uuids,
                                               REMOVE_OLD_FIT)
        target = target - 0.5
        if len(session_uuids) == 1:
            target = np.reshape(target, (1, len(target)))
        if 'prior' in TARGET:
            def get_current_target_and_times(j): return (target[j, :len(trials_arr[j][timing])], np.column_stack(((trials_arr[j][timing] - PRE_TIME),
                                                                                                                  (trials_arr[j][timing] + POST_TIME))))
#             sample_null_distribution = lambda this_target, population_activity:

        elif 'prederr' in TARGET:
            def get_current_target_and_times(j): return (target[j, :len(trials_arr[j][timing])], np.column_stack(((trials_arr[j][timing] - PRE_TIME),
                                                                                                                  (trials_arr[j][timing] + POST_TIME))))

    elif 'stimulus' in TARGET:
        params = [np.nan]

        def get_current_target_and_times(j): return (np.array(trials_arr[j]['signed_contrast']), np.column_stack(((trials_arr[j][timing] - PRE_TIME),
                                                                                                                  (trials_arr[j][timing] + POST_TIME))))
    elif 'reward' in TARGET:
        # Assuming signed contrast
        params = [np.nan]
        def get_current_target_and_times(j): return (np.array(trials_arr[j]['feedbackType']), np.column_stack(((trials_arr[j][timing] - PRE_TIME),
                                                                                                               (trials_arr[j][timing] + POST_TIME))))
    else:
        print('no such TARGET')
        assert False

    # Now that we have the priors from the model fit, loop over sessions and decode
    for j, eid in enumerate(session_uuids):
        if j in indices_to_skip:
            print('\nSkipping session %d of %d' % (j+1, len(session_uuids)))
            continue
        print('\nProcessing session %d of %d' % (j+1, len(session_uuids)))

        spikes, clusters, channels, trials = spikes_arr[j], clusters_arr[j], channels_arr[j], trials_arr[j]
        choice, stim_probLeft = actions_arr[j], stim_probs_arr[j]

        # Extract session data
        ses_info = one.get_details(eid)
        subject = ses_info['subject']
        date = ses_info['start_time'][:10]
        probes_to_use = probes[np.where(eids == eid)[0][0]]

        # Decode per brain region
        for p, probe in enumerate(probes_to_use):
            print('Processing %s (%d of %d)' %
                  (probe, p + 1, len(probes_to_use)))

            if not check_probe(probe, clusters, spikes):
                print('bad probe')
                continue

            # Get list of brain regions
            if ATLAS == 'beryl-atlas':
                mapped_br = br.get(ids=remap(clusters[probe]['atlas_id']))
                clusters_regions = mapped_br['acronym']

            elif ATLAS == 'allen-atlas':
                clusters_regions = combine_layers_cortex(
                    clusters[probe]['acronym'])

            # Get list of neurons that pass QC
            if INCL_NEURONS == 'pass-QC':
                clusters_pass = np.where(
                    clusters[probe]['metrics']['label'] == 1)[0]
            elif INCL_NEURONS == 'all':
                clusters_pass = np.arange(clusters[probe]['metrics'].shape[0])

            # Decode per brain region
            for r, region in enumerate(np.unique(clusters_regions)):

                # Skip region if any of these conditions apply
                if region.islower():
                    continue

                print('Decoding region %s (%d of %d)' %
                      (region, r + 1, len(np.unique(clusters_regions))))

                # Get target, trial times, and population activity for all trials
                # Select spikes and clusters in this brain region
                clusters_in_region = [x for x, y in enumerate(clusters_regions)
                                      if (region == y) and (x in clusters_pass)]
                spks_region = spikes[probe].times[np.isin(
                    spikes[probe].clusters, clusters_in_region)]
                clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                             clusters_in_region)]

                # Check if there are enough neurons in this brain region
                if np.unique(clus_region).shape[0] < MIN_NEURONS:
                    continue

                # prepare target and population_activity
                this_target, times = get_current_target_and_times(j)
                population_activity, cluster_ids = get_spike_counts_in_bins(spks_region,
                                                                            clus_region, times)

                population_activity = np.array(
                    population_activity, dtype=np.float32)
                population_activity_mean = np.mean(population_activity, axis=1)
                population_activity = population_activity - \
                    np.reshape(population_activity_mean,
                               (population_activity.shape[0], 1))
                population_activity = population_activity.T

                # diagnostics
                xyz_region = (np.array(clusters[probe].x[np.isin(clusters[probe]['metrics']['cluster_id'],
                                                                 cluster_ids)]),
                              np.array(clusters[probe].y[np.isin(clusters[probe]['metrics']['cluster_id'],
                                                                 cluster_ids)]),
                              np.array(clusters[probe].z[np.isin(clusters[probe]['metrics']['cluster_id'],
                                                                 cluster_ids)]))
                atlascent, atlasradi = getatlassummary(xyz_region)

                # Initialize cross-validation
                if VALIDATION == 'kfold-interleaved':
                    cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
                elif VALIDATION == 'kfold':
                    cv = KFold(n_splits=NUM_SPLITS, shuffle=False)
                elif VALIDATION == 'none':
                    cv = None

                if REGULARIZATION == 'none':
                    reg = LinearRegression()
                elif REGULARIZATION == 'L1':
                    reg = Lasso(alpha=0.01)
                elif REGULARIZATION == 'CVL1':
                    reg = LassoCV(tol=0.0001, max_iter=10000,
                                  fit_intercept=False, cv=cv)
                    cv = None
                elif REGULARIZATION == 'L2':
                    reg = Ridge()

                # Convert to encoding matrix for post-regression analysis
                convert_to_encoding = np.matmul(
                    population_activity.T, population_activity)/np.dot(this_target, this_target)
                # Decode selected trials
                pred_target, pred_target_train, r2_target, coefs, intercepts = regress(population_activity,
                                                                                       this_target, cross_validation=cv,
                                                                                       return_training_r2_weights=True,
                                                                                       reg=reg)

                r_target = pearsonr(this_target, pred_target)[0]
                mse_target = mean_squared_error(this_target, pred_target)
                r_target_train = pearsonr(this_target, pred_target_train)[0]
                mse_target_train = mean_squared_error(
                    this_target, pred_target_train)

                # Add to dataframe
                decoding_result = decoding_result.append(pd.DataFrame(
                    index=[decoding_result.shape[0] + 1], data={'subject': subject,
                                                                'date': date,
                                                                'eid': eid,
                                                                'probe': probe,
                                                                'region': region,
                                                                'atlas_posx': atlascent[0],
                                                                'atlas_posy': atlascent[1],
                                                                'atlas_posz': atlascent[2],
                                                                'atlas_radius': atlasradi,
                                                                'target': [this_target],
                                                                'population_activity': [population_activity],
                                                                'pred_target': [pred_target],
                                                                'stim_probLeft': [stim_probLeft],
                                                                'choice': [choice],
                                                                'r': r_target,
                                                                'r2': r2_target,
                                                                'mse': mse_target,
                                                                'r_train': r_target_train,
                                                                'mse_train': mse_target_train,
                                                                'tau': 1 / params[0],
                                                                'n_trials': trials['probabilityLeft'].shape[0],
                                                                'n_neurons': np.unique(clus_region).shape[0],
                                                                'cluster_ids': [cluster_ids],
                                                                'reg_coefs_convert_to_encoding': [convert_to_encoding],
                                                                'reg_coefs_encoding': [np.matmul(convert_to_encoding, coefs)],
                                                                'reg_coefs': [coefs],
                                                                'reg_intercepts': [intercepts]}), sort=False)

        decoding_result.to_pickle(join(SAVE_PATH, 'Decoding', DECODER,
                                       ('%s_%s_%s_%s_%s_cells_%s_%s_%s-%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                                                INCL_SESSIONS, INCL_NEURONS, ATLAS, TIMING,
                                                                                int(PRE_TIME*1000), int(POST_TIME*1000)))))

#########################################################################################################
# compute p-values using imposter sessions

decoding_result['r_null'] = np.nan
decoding_result['mse_null'] = np.nan
decoding_result['r_train_null'] = np.nan
decoding_result['mse_train_null'] = np.nan
decoding_result['p_value_r'] = np.nan
decoding_result['p_value_r2'] = np.nan
decoding_result['p_value_mse'] = np.nan

all_trials = decoding_result
_, ind = np.unique(all_trials['eid'], return_index=True)
ind = np.sort(ind)
unique_targets = np.array(all_trials['target'])[ind]
unique_eids = np.array(all_trials['eid'])[ind]
# Shuffle
x = np.arange(len(unique_eids))
np.random.shuffle(x)
unique_targets = unique_targets[x]
unique_eids = unique_eids[x]

len_decoding = len(decoding_result['eid'])
for i in range(len_decoding):
    ind = i+1
    print('Computing p-value for entry %s of %s' % (ind, len_decoding))
    current_decoding_result = decoding_result.loc[ind]
    eid = current_decoding_result['eid']

    null_population_activity = current_decoding_result['population_activity']
    this_target = current_decoding_result['target']

    use_eids = (unique_eids == eid)
    assert len(np.nonzero(use_eids)[0]) <= 1
    franken_trials = np.concatenate(unique_targets[~use_eids])
    length_of_targets = len(this_target)
    length_of_franken_trials = len(franken_trials)

    def get_franken_trials_at_index(
        x): return franken_trials[x: x+length_of_targets]

    def get_null_target(): return get_franken_trials_at_index(
        np.random.randint(length_of_franken_trials-length_of_targets))

    r2_null = np.empty(ITERATIONS)
    r_null = np.empty(ITERATIONS)
    r_train_null = np.empty(ITERATIONS)
    mse_null = np.empty(ITERATIONS)
    mse_train_null = np.empty(ITERATIONS)
    for k in range(ITERATIONS):

        null_target = get_null_target()

        # Decode prior of null trials
        null_pred, null_pred_train, null_r2, _, _ = regress(null_population_activity, null_target,
                                                            cross_validation=cv, return_training_r2_weights=True,
                                                            reg=reg)
        r2_null[k] = null_r2
        r_null[k] = pearsonr(null_target, null_pred)[0]
        mse_null[k] = mean_squared_error(null_target, null_pred)
        r_train_null[k] = pearsonr(null_target, null_pred_train)[0]
        mse_train_null[k] = mean_squared_error(null_target, null_pred_train)

    decoding_result.loc[ind, 'r_null'] = [r_null]
    decoding_result.loc[ind, 'mse_null'] = [mse_null]
    decoding_result.loc[ind, 'r_train_null'] = [r_train_null]
    decoding_result.loc[ind, 'mse_train_null'] = [mse_train_null]
    decoding_result.loc[ind, 'p_value_r'] = np.sum(
        r_target > r_null) / len(r_null)
    decoding_result.loc[ind, 'p_value_r2'] = np.sum(
        r2_target > r2_null) / len(r2_null)
    decoding_result.loc[ind, 'p_value_mse'] = np.sum(
        mse_target > mse_null) / len(mse_null)

    decoding_result.to_pickle(join(SAVE_PATH, 'Decoding', DECODER,
                                   ('%s_%s_%s_%s_%s_cells_%s_%s_%s-%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                                            INCL_SESSIONS, INCL_NEURONS, ATLAS, TIMING,
                                                                            int(PRE_TIME*1000), int(POST_TIME*1000)))))
