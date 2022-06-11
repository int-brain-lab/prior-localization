#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 01 2022

@author: bensonb
"""
import os
import pickle
from plot_decoding_brain import brain_results, bar_results, brain_cortex_results, bar_results_basic, brain_SwansonFlat_results, aggregate_data, discretize_target
from bernoulli_confidenceinterval import Bernoulli_ci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.metrics import r2_score

#cmap='Purples'#,'Blues','Greens','Oranges','Reds'

RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/'
FIGURE_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'
FIGURE_SUFFIX = '.png'

#%%
SPECIFIC_DECODING = 'decode_pLeft_task_Logistic_accuracy_control_100_pseudos_align_goCue_times_timeWin_-0_6_-0_2_alleidProb_incTol'
VARIABLE_FOLDER = 'block/'
if not os.path.isdir(os.path.join(FIGURE_PATH,
                                  VARIABLE_FOLDER,
                                  SPECIFIC_DECODING)):
    os.mkdir(os.path.join(FIGURE_PATH,
                          VARIABLE_FOLDER,
                          SPECIFIC_DECODING))
RESULTS_DATE = '2022-03-17'
FILE_PATH = os.path.join(RESULTS_PATH,SPECIFIC_DECODING,('_'.join([RESULTS_DATE, 'results'])) + '.parquet')

results = pd.read_parquet(FILE_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]

data = aggregate_data(results, RESULTS_PATH, SPECIFIC_DECODING, 
                      get_probabilities=True)
    
all_pvalues = data['p-value']
all_scores = data['score']
all_null_scores = data['null_scores']
all_targets = data['target']
all_preds = data['prediction']
all_probs = data['probability']
all_block_pLeft = data['block_pLeft']
all_actn = data['active_neurons']
all_regions = data['region']
all_eids = data['eid']
all_probes = data['probe']
all_masks = data['mask']

save_data = pd.DataFrame({'probe': [int(p[6:]) for p in all_probes],
             'session_eid': all_eids,
             'acronym': all_regions,
             'score_accuracy': all_scores,
             'pvalue': all_pvalues})

save_dictionary = {'author': 'Brandon Benson',
                   'date': '2022-06-01',
                   'description': 'Logistic regression is used to decode the block from neural data.  accuracy values are reported after performing regression.  A 5-fold interleaved cross-validation is used to determine hyper-parameters on 80 percent of the data and predictions are made on the held-out 20 percent.  This is repeated 5 times with different held-out sets to cover all trials.  accuracy performance is then evaluated using the predictions across all trials.  The same process is used to decode 100 block pseudo-sessions which are used to evaluate the p-value of the reported accuracy value.',
                   'inclusion_crit': 'Sessions must have 400 or more trials.  Trials with reaction time (firstMovement_time-goCue_time) less than 80ms are excluded. Unbiased blocks are excluded.  Regions with less than 10 units are excluded.  QC criteria is 3/3',
                   'data': save_data}

with open('Block_BWM_'+save_dictionary['date']+'.p','wb') as f:
    pickle.dump(save_dictionary,f,pickle.HIGHEST_PROTOCOL)
    

#%%
SPECIFIC_DECODING = 'decode_signcont_task_Lasso_r2_control_100_pseudo-session_align_goCue_times_timeWin_0_0_1_alleid220414'
VARIABLE_FOLDER = 'stimulus/'
if not os.path.isdir(os.path.join(FIGURE_PATH,
                                  VARIABLE_FOLDER,
                                  SPECIFIC_DECODING)):
    os.mkdir(os.path.join(FIGURE_PATH,
                          VARIABLE_FOLDER,
                          SPECIFIC_DECODING))
RESULTS_DATE = '2022-04-14'
FILE_PATH = os.path.join(RESULTS_PATH,SPECIFIC_DECODING,('_'.join([RESULTS_DATE, 'results'])) + '.parquet')

results = pd.read_parquet(FILE_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]

data = aggregate_data(results, RESULTS_PATH, SPECIFIC_DECODING)
    
all_pvalues = data['p-value']
all_scores = data['score']
all_null_scores = data['null_scores']
all_targets = data['target']
all_preds = data['prediction']
all_block_pLeft = data['block_pLeft']
all_actn = data['active_neurons']
all_regions = data['region']
all_eids = data['eid']
all_probes = data['probe']
all_masks = data['mask']


save_data = pd.DataFrame({'probe': [int(p[6:]) for p in all_probes],
             'session_eid': all_eids,
             'acronym': all_regions,
             'score_r2': all_scores,
             'pvalue': all_pvalues})

save_dictionary = {'author': 'Brandon Benson',
                   'date': '2022-06-01',
                   'description': 'Linear regression is used to decode the stimulus contrast level from neural data where contrast levels are multipled by -1 for one side stimulus and 1 for the other side stimulus.  r-squared values are reported after performing regression.  A 5-fold interleaved cross-validation is used to determine hyper-parameters on 80 percent of the data and predictions are made on the held-out 20 percent.  This is repeated 5 times with different held-out sets to cover all trials.  r-squared performance is then evaluated using the predictions across all trials.  The same process is used to decode 100 stimulus pseudo-sessions which are used to evaluate the p-value of the reported r-squared value.',
                   'inclusion_crit': 'Sessions must have 400 or more trials.  Trials with reaction time (firstMovement_time-goCue_time) less than 80ms are excluded. Unbiased blocks are excluded.  Regions with less than 10 units are excluded.  QC criteria is 3/3',
                   'data': save_data}

with open('Stim_BWM_'+save_dictionary['date']+'.p','wb') as f:
    pickle.dump(save_dictionary,f,pickle.HIGHEST_PROTOCOL)
    

#%%
SPECIFIC_DECODING = 'decode_choice_task_Logistic_accuracy_control_100_impostor-session_align_firstMovement_times_timeWin_-0_1_0_0_testnulls1'
VARIABLE_FOLDER = 'choice/'
if not os.path.isdir(os.path.join(FIGURE_PATH,
                                  VARIABLE_FOLDER,
                                  SPECIFIC_DECODING)):
    os.mkdir(os.path.join(FIGURE_PATH,
                          VARIABLE_FOLDER,
                          SPECIFIC_DECODING))
RESULTS_DATE = '2022-03-29'
FILE_PATH = os.path.join(RESULTS_PATH,SPECIFIC_DECODING,('_'.join([RESULTS_DATE, 'results'])) + '.parquet')

results = pd.read_parquet(FILE_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]

data = aggregate_data(results, RESULTS_PATH, SPECIFIC_DECODING, 
                      get_probabilities=True)
    
all_pvalues = data['p-value']
all_scores = data['score']
all_null_scores = data['null_scores']
all_targets = data['target']
all_preds = data['prediction']
all_probs = data['probability']
all_block_pLeft = data['block_pLeft']
all_actn = data['active_neurons']
all_regions = data['region']
all_eids = data['eid']
all_probes = data['probe']
all_masks = data['mask']


save_data = pd.DataFrame({'probe': [int(p[6:]) for p in all_probes],
             'session_eid': all_eids,
             'acronym': all_regions,
             'score_accuracy': all_scores,
             'pvalue': all_pvalues})

save_dictionary = {'author': 'Brandon Benson',
                   'date': '2022-06-01',
                   'description': 'Logistic regression is used to decode the choice from neural data.  accuracy values are reported after performing regression.  A 5-fold interleaved cross-validation is used to determine hyper-parameters on 80 percent of the data and predictions are made on the held-out 20 percent.  This is repeated 5 times with different held-out sets to cover all trials.  accuracy performance is then evaluated using the predictions across all trials.  The same process is used to decode 100 choice impostor-sessions (see get_impostor_target here https://github.com/int-brain-lab/prior-localization/blob/bb/decoding/decoding_utils.py) which are used to evaluate the p-value of the reported accuracy value.',
                   'inclusion_crit': 'Sessions must have 400 or more trials.  Trials with reaction time (firstMovement_time-goCue_time) less than 80ms are excluded. Only choices of Left or Right are used.  Regions with less than 10 units are excluded.  QC criteria is 3/3',
                   'data': save_data}

with open('Choice_BWM_'+save_dictionary['date']+'.p','wb') as f:
    pickle.dump(save_dictionary,f,pickle.HIGHEST_PROTOCOL)
    
#%%
SPECIFIC_DECODING = 'decode_feedback_task_Logistic_accuracy_control_100_impostor-session_align_feedback_times_timeWin_0_0_2_testnulls2'
VARIABLE_FOLDER = 'reward/'
if not os.path.isdir(os.path.join(FIGURE_PATH,
                                  VARIABLE_FOLDER,
                                  SPECIFIC_DECODING)):
    os.mkdir(os.path.join(FIGURE_PATH,
                          VARIABLE_FOLDER,
                          SPECIFIC_DECODING))
RESULTS_DATE = '2022-03-31'
FILE_PATH = os.path.join(RESULTS_PATH,SPECIFIC_DECODING,('_'.join([RESULTS_DATE, 'results'])) + '.parquet')

results = pd.read_parquet(FILE_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]

data = aggregate_data(results, RESULTS_PATH, SPECIFIC_DECODING, 
                      get_probabilities=True)
    
all_pvalues = data['p-value']
all_scores = data['score']
all_null_scores = data['null_scores']
all_targets = data['target']
all_preds = data['prediction']
all_probs = data['probability']
all_block_pLeft = data['block_pLeft']
all_actn = data['active_neurons']
all_regions = data['region']
all_eids = data['eid']
all_probes = data['probe']
all_masks = data['mask']

save_data = pd.DataFrame({'probe': [int(p[6:]) for p in all_probes],
             'session_eid': all_eids,
             'acronym': all_regions,
             'score_accuracy': all_scores,
             'pvalue': all_pvalues})

save_dictionary = {'author': 'Brandon Benson',
                   'date': '2022-06-01',
                   'description': 'Logistic regression is used to decode the reward (aka feedback) from neural data.  accuracy values are reported after performing regression.  A 5-fold interleaved cross-validation is used to determine hyper-parameters on 80 percent of the data and predictions are made on the held-out 20 percent.  This is repeated 5 times with different held-out sets to cover all trials.  accuracy performance is then evaluated using the predictions across all trials.  The same process is used to decode 100 reward impostor-sessions (see get_impostor_target here https://github.com/int-brain-lab/prior-localization/blob/bb/decoding/decoding_utils.py) which are used to evaluate the p-value of the reported accuracy value.',
                   'inclusion_crit': 'Sessions must have 400 or more trials.  Trials with reaction time (firstMovement_time-goCue_time) less than 80ms are excluded. Only reward values of -1 or 1 are used.  Regions with less than 10 units are excluded.  QC criteria is 3/3',
                   'data': save_data}

with open('Reward_BWM_'+save_dictionary['date']+'.p','wb') as f:
    pickle.dump(save_dictionary,f,pickle.HIGHEST_PROTOCOL)
    
with open('Reward_BWM_'+save_dictionary['date']+'.p','rb') as f:
    out = pickle.load(f)
  