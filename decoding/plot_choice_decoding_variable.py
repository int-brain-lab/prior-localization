#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:54:53 2022

@author: bensonb
"""
import os
from plot_decoding_brain import brain_results, bar_results, bar_results_basic, aggregate_data
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

# plt.title('r2s')
plt.hist(results.loc[:,'Score_test'], bins=30, 
         histtype='step', density=True, lw=5)
plt.hist(results.loc[:,'Score_test_pseudo0'], bins=30, 
         histtype='step', density=True, lw=5)
plt.xlabel('score')
plt.ylabel('density')
plt.legend(['test','null'])
plt.show()
plt.hist(all_pvalues, bins=20, 
         histtype='step', density=True, lw=5)
plt.xlabel('p-value')
plt.ylabel('density')
plt.show()

# plt.title('prior: random example')
# plt.plot(all_targets[100:400])
# plt.plot(all_preds[100:400])
# plt.legend(['targets','predictions'])
# plt.savefig(os.path.join(FIGURE_PATH,
#                          VARIABLE_FOLDER,
#                          SPECIFIC_DECODING,
#             ('_'.join([RESULTS_DATE, 'predictionsTraceRandomExample'])) +
#             FIGURE_SUFFIX), 
#             dpi=600)
# plt.show()

# plt.title('prior: best example')
# plt.plot(best_targets[100:400])
# plt.plot(best_preds[100:400])
# plt.legend(['targets','predictions'])
# plt.savefig(os.path.join(FIGURE_PATH,
#                          VARIABLE_FOLDER,
#                          SPECIFIC_DECODING,
#             ('_'.join([RESULTS_DATE, 'predictionsTraceBestExample'])) +
#             FIGURE_SUFFIX), 
#             dpi=600)
# plt.show()

# plotting

# all_targets, best_targets = np.array(all_targets), np.array(best_targets)
# best_targets_continuous = np.copy(best_targets)
# edge = np.linspace(0,1,11)

# for i in range(len(edge)-1):
#     all_targets[(all_targets >= edge[i])&(all_targets < edge[i+1])] = .5*(edge[i]+edge[i+1])
#     best_targets[(best_targets >= edge[i])&(best_targets < edge[i+1])] = .5*(edge[i]+edge[i+1])
assert np.max(all_scores)-np.min(all_scores) < 999
index_max = np.argmax(all_scores - (999*(all_regions!='SSp-ul')))
best_targets = all_targets[index_max]
best_score = all_scores[index_max]
best_preds = all_preds[index_max]
best_probs = all_probs[index_max]
best_block_pLeft = all_block_pLeft[index_max]
best_actn = all_actn[index_max]
best_eid = all_eids[index_max]
best_probe = all_probes[index_max]
best_region = all_regions[index_max]
best_masks = all_masks[index_max]

all_probs_discrete = np.concatenate(all_probs)
all_probs_continuous = np.concatenate(all_probs)
best_probs_discrete = np.copy(best_probs)
best_probs_continuous = np.copy(best_probs)
edge = np.linspace(0,1,11)
for i in range(len(edge)-1):
    if i == len(edge)-2: # last edge includes boundary
        all_probs_discrete[(all_probs_continuous >= edge[i])&
                           (all_probs_continuous <= edge[i+1])] = .5*(edge[i]+edge[i+1])
        best_probs_discrete[(best_probs_continuous >= edge[i])&
                            (best_probs_continuous <= edge[i+1])] = .5*(edge[i]+edge[i+1])
    else:
        all_probs_discrete[(all_probs_continuous >= edge[i])&
                           (all_probs_continuous < edge[i+1])] = .5*(edge[i]+edge[i+1])
        best_probs_discrete[(best_probs_continuous >= edge[i])&
                            (best_probs_continuous < edge[i+1])] = .5*(edge[i]+edge[i+1])

plot_name = 'medianaccuracy'
reg_nulls = np.array([np.median(all_null_scores[all_regions==reg],axis=0) for reg in np.unique(all_regions)])
reg_values = np.array([all_scores[all_regions==reg] for reg in np.unique(all_regions)])
reg_pvalue = np.array([np.mean(np.median(reg_values[i])<=reg_nulls[i]) \
              for i in range(len(np.unique(all_regions)))])
acronyms = np.unique(all_regions)[reg_pvalue<0.05]
values = reg_values[reg_pvalue<0.05]
nulls_m = np.median(reg_nulls,axis=1)[reg_pvalue<0.05]
nulls_l = np.min(reg_nulls,axis=1)[reg_pvalue<0.05]
nulls_h = np.array([scipy.stats.scoreatpercentile(reg_nulls[i,:], 
                    95, interpolation_method='fraction') for i in range(reg_nulls.shape[0])])[reg_pvalue<0.05]
nulls = np.vstack((nulls_l,nulls_m,nulls_h))

brain_results(acronyms, 
                np.array([np.median(v) for v in values]), 
                os.path.join(VARIABLE_FOLDER,
                              SPECIFIC_DECODING,
                            ('_'.join([RESULTS_DATE, 'brains', plot_name])) +
                            FIGURE_SUFFIX), 
                FILE_PATH = FIGURE_PATH,
                cmap='Oranges',
                YMIN=0.5,
                value_title='       Accuracy\n               %d of %d sig.'%(np.sum(reg_pvalue<0.05),len(reg_pvalue)))#: %.3f'%(len(np.unique(np.random.choice(all_regions,size=int(len(all_regions)*0.05),replace=False)))/len(np.unique(all_regions))))
bar_results(acronyms,
            values,
            nulls,
            os.path.join(VARIABLE_FOLDER,
                          SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'bars', plot_name])) +
            FIGURE_SUFFIX),
            YMIN=0.5,
            ylab='Accuracy',
            TOP_N=15)

plot_name = 'maxsignificantaccuracy'
MIN_NUMBER_SESSIONS = 1
all_sigs = (all_pvalues<=0.05)
use_region = lambda reg: len(np.nonzero((all_regions==reg)&all_sigs)[0]) and len(np.nonzero(all_regions==reg)[0])>=MIN_NUMBER_SESSIONS
get_region_value = lambda reg: np.max(all_scores[(all_regions==reg)&all_sigs])
get_region_err = lambda reg: np.std(all_scores[(all_regions==reg)&all_sigs])

regions = np.array([reg for reg in np.unique(all_regions) if use_region(reg)])
reg_values = np.array([get_region_value(reg) for reg in regions])
reg_errs = np.array([get_region_err(reg) for reg in regions])
acronyms, values, errs = regions, reg_values, reg_errs

brain_results(acronyms, 
                values, 
                os.path.join(VARIABLE_FOLDER,
                              SPECIFIC_DECODING,
                            ('_'.join([RESULTS_DATE, 'brains', plot_name])) +
                            FIGURE_SUFFIX), 
                FILE_PATH = FIGURE_PATH,
                cmap='Oranges',
                value_title='       Accuracy')
bar_results_basic(acronyms,
            values,
            filename=os.path.join(VARIABLE_FOLDER,
                          SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'bars', plot_name])) +
            FIGURE_SUFFIX),
            YMIN=0.5)

plot_name = 'fsessionssignificant'
MIN_NUMBER_SESSIONS = 1
all_sigs = (all_pvalues<=0.05)
use_region = lambda reg: len(np.nonzero((all_regions==reg)&all_sigs)[0]) and len(np.nonzero(all_regions==reg)[0])>=MIN_NUMBER_SESSIONS
get_region_nsigsess = lambda reg: len(all_scores[(all_regions==reg)&all_sigs])
get_region_nsess = lambda reg: len(all_scores[all_regions==reg])
get_region_value = lambda reg: get_region_nsigsess(reg)/get_region_nsess(reg)
ci_lower, ci_upper = 0.05, 1.0
get_region_errm = lambda reg: get_region_value(reg) - Bernoulli_ci(get_region_nsigsess(reg),
                                             get_region_nsess(reg),
                                             ci_lower=ci_lower,
                                             ci_upper=ci_upper)[0]
get_region_errp = lambda reg: Bernoulli_ci(get_region_nsigsess(reg),
                                             get_region_nsess(reg),
                                             ci_lower=ci_lower,
                                             ci_upper=ci_upper)[1] - get_region_value(reg)

regions = np.array([reg for reg in np.unique(all_regions) if use_region(reg)])
reg_values = np.array([get_region_value(reg) for reg in regions])
reg_errsm = np.array([get_region_errm(reg) for reg in regions])
reg_errsp = np.array([get_region_errp(reg) for reg in regions])
acronyms, values, errs = regions, reg_values, np.vstack((reg_errsm, reg_errsp))

brain_results(acronyms, 
                values-errs[0,:], 
                os.path.join(VARIABLE_FOLDER,
                              SPECIFIC_DECODING,
                            ('_'.join([RESULTS_DATE, 'brains', plot_name])) +
                            FIGURE_SUFFIX), 
                FILE_PATH = FIGURE_PATH,
                cmap='Oranges',
                value_title='chance of \nsignificant session \n(95\% ci)')
bar_results_basic(acronyms,
            values,
            errs,
            os.path.join(VARIABLE_FOLDER,
                          SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'bars', plot_name])) +
            FIGURE_SUFFIX))

plot_name = 'nsessionssignificant'
MIN_NUMBER_SESSIONS = 1
all_sigs = (all_pvalues<=0.05)
use_region = lambda reg: len(np.nonzero((all_regions==reg)&all_sigs)[0]) and len(np.nonzero(all_regions==reg)[0])>=MIN_NUMBER_SESSIONS
get_region_value = lambda reg: len(all_scores[(all_regions==reg)&all_sigs])
get_region_err = lambda reg: 0

regions = np.array([reg for reg in np.unique(all_regions) if use_region(reg)])
reg_values = np.array([get_region_value(reg) for reg in regions])
reg_errs = np.array([get_region_err(reg) for reg in regions])
acronyms, values, errs = regions, reg_values, reg_errs

# brain_results(acronyms, 
#                 values, 
#                 os.path.join(VARIABLE_FOLDER,
#                               SPECIFIC_DECODING,
#                             ('_'.join([RESULTS_DATE, 'brains', plot_name])) +
#                             FIGURE_SUFFIX), 
#                 FILE_PATH = FIGURE_PATH,
#                 cmap='Purples')
bar_results_basic(acronyms,
            values,
            errs,
            os.path.join(VARIABLE_FOLDER,
                          SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'bars', plot_name])) +
            FIGURE_SUFFIX))


#%%
all_df = pd.DataFrame({'Probabilities':all_probs_discrete,
                       'Target':np.concatenate(all_targets),
                       'Predictions':np.concatenate(all_preds),
                       'pLeft':np.concatenate(all_block_pLeft)})

best_df = pd.DataFrame({'Probabilities':best_probs_discrete,
                        'Target':best_targets,
                        'Predictions':best_preds,
                        'pLeft':best_block_pLeft})

sns.set_theme(style="whitegrid")
ci = 95

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', 
                 data=all_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictions_xtarget'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Target', y='Predictions', 
                 data=best_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictionsBest_xtarget'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Probabilities', y='Target',
                 data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
xlabs = np.sort(np.unique(all_probs_discrete))
xlabs = [float('%.8f'%xlab) for xlab in xlabs]
ax.set_xticklabels(xlabs, rotation=45)
plt.ylim(0,1)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'calibration_probabilities'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Probabilities', y='Target',
                 data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
xlabs = np.sort(np.unique(best_probs_discrete))
xlabs = [float('%.8f'%xlab) for xlab in xlabs]
ax.set_xticklabels(xlabs, rotation=45)
plt.ylim(0,1)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'calibrationBest_probabilities'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()

# ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
#             data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
#             scale='count')
# #ax.set_ylim(0,1)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlock', SPECIFIC_DECODING])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

best_trials = np.arange(len(best_masks))[[m=='1' for m in best_masks]]
assert len(best_trials) == len(best_targets)
plt.figure(figsize=(10,3))
plt.title(best_eid+' ['+best_probe+'] ['+best_region+']'+'\n accuracy$=$%.4f, $n=$%d'%(best_score,best_actn))
plt.plot(best_trials[best_targets>0.5],best_probs[best_targets>0.5],'C0o',lw=2,ms=4)
plt.plot(best_trials[best_targets<0.5],best_probs[best_targets<0.5],'C1o',lw=2,ms=4)
plt.yticks([0,.5,1])
#plt.ylim(0,1)
#plt.xlim(0,len(best_masks))
plt.legend(['Probability given choice $=1$', 
            'Probability given choice $=0$'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Choice')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'probabilitiesTraceBest'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()


plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
vals, bins, _ = plt.hist(all_actn, density=True)
ymax, ymin = np.max(vals), np.min(vals)
plt.plot(best_actn*np.ones(51) ,np.linspace(0,ymax*1.1,51))
plt.ylim(0,ymax*1.1)
plt.legend(['Best','All'],frameon=True,loc=(-0.15,1.15))
plt.xlabel('Average Number of Neurons Active')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'averageNeuronsActiveDistribution'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()


