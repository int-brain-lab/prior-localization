#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:54:53 2022

@author: bensonb
"""
import os
from plot_decoding_brain import brain_results, bar_results, brain_cortex_results, bar_results_basic, aggregate_data
from bernoulli_confidenceinterval import Bernoulli_ci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

#cmap='Purples'#,'Blues','Greens','Oranges','Reds'

RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/'
FIGURE_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'
FIGURE_SUFFIX = '.png'

#%%
SPECIFIC_DECODING = 'decode_signcont_task_Lasso_r2_control_100_pseudos_align_goCue_times_timeWin_0_0_1_alleidProb_incTol'
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

# plt.title('r2s')
plt.hist(results.loc[:,'Score_test'], bins=30, 
         histtype='step', density=True, lw=5)

plt.hist(np.concatenate(all_null_scores), bins=30, 
         histtype='step', density=True, lw=5)#results.loc[:,'Score_test_pseudo0']
plt.xlabel('score')
plt.ylabel('density')
plt.legend(['test','null'])
plt.show()
plt.hist(all_pvalues, bins=20, 
         histtype='step', density=True, lw=5)
plt.xlabel('p-value')
plt.ylabel('density')
plt.show()

#%%

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
#             ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2Example'])) +
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
index_max = np.argmax(all_scores)
best_targets = all_targets[index_max]
best_score = all_scores[index_max]
best_preds = all_preds[index_max]
best_block_pLeft = all_block_pLeft[index_max]
best_actn = all_actn[index_max]
best_eid = all_eids[index_max]
best_probe = all_probes[index_max]
best_region = all_regions[index_max]
best_masks = all_masks[index_max]
index_max_orb = np.argmax(all_scores - (999*(all_regions!='PRNr')))
best_targets_orb = all_targets[index_max_orb]
best_score_orb = all_scores[index_max_orb]
best_preds_orb = all_preds[index_max_orb]
best_block_pLeft_orb = all_block_pLeft[index_max_orb]
best_actn_orb = all_actn[index_max_orb]
best_eid_orb = all_eids[index_max_orb]
best_probe_orb = all_probes[index_max_orb]
best_region_orb = all_regions[index_max_orb]
best_masks_orb = all_masks[index_max_orb]

all_preds_discrete = np.concatenate(all_preds)
all_preds_continuous = np.concatenate(all_preds)
best_preds_discrete = np.copy(best_preds)
best_preds_continuous = np.copy(best_preds)
best_preds_discrete_orb = np.copy(best_preds_orb)
best_preds_continuous_orb = np.copy(best_preds_orb)
edge = np.linspace(-1,1,11)
edge = np.insert(np.append(edge,np.infty),0,-np.infty)
for i in range(len(edge)-1):
    if i == len(edge)-2: # last edge includes boundary
        all_preds_discrete[(all_preds_continuous >= edge[i])&
                           (all_preds_continuous <= edge[i+1])] = .5*(edge[i]+edge[i+1])
        best_preds_discrete[(best_preds_continuous >= edge[i])&
                            (best_preds_continuous <= edge[i+1])] = .5*(edge[i]+edge[i+1])
        best_preds_discrete_orb[(best_preds_continuous_orb >= edge[i])&
                            (best_preds_continuous_orb <= edge[i+1])] = .5*(edge[i]+edge[i+1])
    else:
        all_preds_discrete[(all_preds_continuous >= edge[i])&
                           (all_preds_continuous < edge[i+1])] = .5*(edge[i]+edge[i+1])
        best_preds_discrete[(best_preds_continuous >= edge[i])&
                            (best_preds_continuous < edge[i+1])] = .5*(edge[i]+edge[i+1])
        best_preds_discrete_orb[(best_preds_continuous_orb >= edge[i])&
                            (best_preds_continuous_orb < edge[i+1])] = .5*(edge[i]+edge[i+1])
plot_name = 'medianr2'
reg_nulls = np.array([np.mean(all_null_scores[all_regions==reg],axis=0) for reg in np.unique(all_regions)])
reg_values = np.array([all_scores[all_regions==reg] for reg in np.unique(all_regions)])
reg_pvalue = np.array([np.mean(np.mean(reg_values[i])<=reg_nulls[i]) \
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
                cmap='Blues',
                YMIN=0,
                YMAX=0.08,
                value_title='$R^2$')# '\n             %d of %d sig.'%(np.sum(reg_pvalue<0.05),len(reg_pvalue))
brain_cortex_results(acronyms, 
                np.array([np.median(v) for v in values]), cmap='Blues')

bar_results(acronyms,
            values,
            nulls,
            os.path.join(VARIABLE_FOLDER,
                          SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'bars', plot_name])) +
            FIGURE_SUFFIX),
            YMIN=-0.1,
            ylab='$R^2$',
            POOL_PROTOCOL='mean',
            TOP_N=15)
# plot_name = 'mediansignificantr2'
# MIN_NUMBER_SESSIONS = 1
# all_sigs = (all_pvalues<=0.05)
# use_region = lambda reg: len(np.nonzero((all_regions==reg)&all_sigs)[0]) and len(np.nonzero(all_regions==reg)[0])>=MIN_NUMBER_SESSIONS
# get_region_value = lambda reg: np.median(all_scores[(all_regions==reg)&all_sigs])
# get_region_err = lambda reg: np.std(all_scores[(all_regions==reg)&all_sigs])

# regions = np.array([reg for reg in np.unique(all_regions) if use_region(reg)])
# reg_values = np.array([get_region_value(reg) for reg in regions])
# reg_errs = np.array([get_region_err(reg) for reg in regions])
# acronyms, values, errs = regions, reg_values, reg_errs

# brain_results(acronyms, 
#                 values, 
#                 os.path.join(VARIABLE_FOLDER,
#                               SPECIFIC_DECODING,
#                             ('_'.join([RESULTS_DATE, 'brains', plot_name])) +
#                             FIGURE_SUFFIX), 
#                 FILE_PATH = FIGURE_PATH,
#                 cmap='Blues',
#                 YMIN=0,
#                 value_title='$R^2$')
# bar_results(acronyms,
#             values,
#             errs,
#             os.path.join(VARIABLE_FOLDER,
#                           SPECIFIC_DECODING,
#             ('_'.join([RESULTS_DATE, 'bars', plot_name])) +
#             FIGURE_SUFFIX),
#             YMIN=0,
#             ylab='$R^2$')

plot_name = 'maxsignificantr2'
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
                cmap='Blues',
                YMIN=0,
                value_title='$R^2$')
bar_results_basic(acronyms,
            values,
            filename=os.path.join(VARIABLE_FOLDER,
                          SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'bars', plot_name])) +
            FIGURE_SUFFIX),
            YMIN=0,
            ylab='$R^2$')

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
                cmap='Blues',
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
sns.set_theme(style="whitegrid")

all_df = pd.DataFrame({'Target':np.concatenate(all_targets),
                       'Predictions':np.concatenate(all_preds),
                       'pLeft':np.concatenate(all_block_pLeft)})

best_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds,
                       'pLeft':best_block_pLeft})

best_df_orb = pd.DataFrame({'Target':best_targets_orb,
                       'Predictions':best_preds_orb,
                       'pLeft':best_block_pLeft_orb})

all_discrete_df = pd.DataFrame({'Target':np.concatenate(all_targets),
                       'Predictions':all_preds_discrete,
                       'pLeft':np.concatenate(all_block_pLeft)})

best_discrete_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds_discrete,
                       'pLeft':best_block_pLeft})

best_discrete_df_orb = pd.DataFrame({'Target':best_targets_orb,
                       'Predictions':best_preds_discrete_orb,
                       'pLeft':best_block_pLeft_orb})

ci = 95

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft',
                 data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set(xlabel='Stimulus')
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictions_xtarget'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()

plt.figure(figsize=(4.2,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft',
                 data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:], 
                 ci=ci, capsize=.2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set(xlabel='Stimulus')
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictionsBest_xtarget'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()

plt.figure(figsize=(4.2,5))
plt.title(best_eid_orb+' \n['+best_probe_orb+'] ['+best_region_orb+']')
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft',
                 data=best_df_orb.loc[(best_df_orb['pLeft']==0.8)|(best_df_orb['pLeft']==0.2),:], 
                 ci=ci, capsize=.2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set(xlabel='Stimulus')
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictionsBest_xtarget_orb'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Predictions', y='Target', hue='pLeft',
                 data=all_discrete_df.loc[(all_discrete_df['pLeft']==0.8)|(all_discrete_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
xlabs = np.sort(np.unique(all_preds_discrete))
xlabs = [float('%.8f'%xlab) for xlab in xlabs]
ax.set_xticklabels(xlabs, rotation=45)
#plt.ylim(0,1)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'calibration_probabilities'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Predictions', y='Target', hue='pLeft',
                 data=best_discrete_df.loc[(best_discrete_df['pLeft']==0.8)|(best_discrete_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
xlabs = np.sort(np.unique(best_preds_discrete))
xlabs = [float('%.8f'%xlab) for xlab in xlabs]
ax.set_xticklabels(xlabs, rotation=45)
#plt.ylim(0,1)

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
# ax.set_ylim(0,1)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlock', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

#plt.legend(['Prediction given stimulus $> 0$', 
#            'Prediction given stimulus $< 0$'],frameon=True,loc=(-0.15,1.1))
# ax = sns.violinplot(x='Target', y='Predictions',
#             data=all_df,
#             scale='count')
# ax.set_ylim(0,1)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictions', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
#             data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
#             scale='count')
# ax.set_ylim(0,1)
# plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions',
#             data=best_df,
#             scale='count')
# ax.set_ylim(0,1)
# plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsBestR2', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

best_trials = np.arange(len(best_masks))[[m=='1' for m in best_masks]]
assert len(best_trials) == len(best_targets)
plt.figure(figsize=(10,3))
plt.title(best_eid+' ['+best_probe+'] ['+best_region+']'+'\n $r^2=$%.4f'%best_score)
plt.plot(best_trials[best_targets>0],best_preds[best_targets>0],'C0o',lw=2,ms=4)
plt.plot(best_trials[best_targets<0],best_preds[best_targets<0],'C1o',lw=2,ms=4)
plt.yticks([-1,0,1])
plt.ylim(-1,1)
plt.legend(['Prediction given stimulus $> 0$', 
            'Prediction given stimulus $< 0$'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Stimulus')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()


best_trials_orb = np.arange(len(best_masks_orb))[[m=='1' for m in best_masks_orb]]
plt.figure(figsize=(10,3))
plt.title(best_eid_orb+' ['+best_probe_orb+'] ['+best_region_orb+']'+'\n $r^2=$%.4f'%best_score_orb)
plt.plot(best_trials_orb[best_targets_orb>0],best_preds_orb[best_targets_orb>0],'C0o',lw=2,ms=4)
plt.plot(best_trials_orb[best_targets_orb<0],best_preds_orb[best_targets_orb<0],'C1o',lw=2,ms=4)
plt.yticks([-1,0,1])
plt.ylim(-1,1)
plt.legend(['Prediction given stimulus $> 0$', 
            'Prediction given stimulus $< 0$'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Stimulus')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2_orb'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()

best_trials = np.arange(len(best_masks))[[m=='1' for m in best_masks]]
plt.figure(figsize=(10,3))
plt.title(best_eid+' ['+best_probe+'] ['+best_region+']'+'\n $r^2=$%.4f'%best_score)
plt.plot(best_trials,best_targets,'k',lw=2,ms=4)
plt.plot(best_trials,best_preds,'C0',lw=2,ms=4)
plt.yticks([-1,0,1])
plt.ylim(-1,1)
plt.legend(['True','Prediction'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Stimulus')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2_2'])) +
            FIGURE_SUFFIX), 
            dpi=600)
plt.show()

best_trials_orb = np.arange(len(best_masks_orb))[[m=='1' for m in best_masks_orb]]
plt.figure(figsize=(10,3))
plt.title(best_eid_orb+' ['+best_probe_orb+'] ['+best_region_orb+']'+'\n $r^2=$%.4f'%best_score_orb)
plt.plot(best_trials_orb,best_targets_orb,'k',lw=2,ms=4)
plt.plot(best_trials_orb,best_preds_orb,'C0',lw=2,ms=4)
plt.yticks([-1,0,1])
plt.ylim(-1,1)
plt.legend(['True','Prediction'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Stimulus')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2_orb2'])) +
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


