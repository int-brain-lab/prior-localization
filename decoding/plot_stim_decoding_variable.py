#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:54:53 2022

@author: bensonb
"""
import os
from plot_decoding_brain import brain_results, bar_results, aggregate_data
from bernoulli_confidenceinterval import Bernoulli_ci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#cmap='Purples'#,'Blues','Greens','Oranges','Reds'

RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/'
FIGURE_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'
FIGURE_SUFFIX = '.png'

#%%
SPECIFIC_DECODING = 'decode_signcont_task_Lasso_control_100_pseudosessions_align_goCue_times_timeWindow_0_0_1_eidall_AdjustFeatures'
VARIABLE_FOLDER = 'stimulus/'
if not os.path.isdir(os.path.join(FIGURE_PATH,
                                  VARIABLE_FOLDER,
                                  SPECIFIC_DECODING)):
    os.mkdir(os.path.join(FIGURE_PATH,
                          VARIABLE_FOLDER,
                          SPECIFIC_DECODING))
RESULTS_DATE = '2022-03-05'
FILE_PATH = os.path.join(RESULTS_PATH,SPECIFIC_DECODING,('_'.join([RESULTS_DATE, 'results'])) + '.parquet')

results = pd.read_parquet(FILE_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]

data = aggregate_data(results, RESULTS_PATH, SPECIFIC_DECODING)
    
all_pvalues = data['p-value']
all_r2s = data['r2']
all_accuracies = data['accuracy']
all_targets = data['target']
all_preds = data['prediction']
all_block_pLeft = data['block_pLeft']
all_actn = data['active_neurons']
all_regions = data['region']
all_eids = data['eid']
all_probes = data['probe']

#%%
# plt.title('r2s')
plt.hist(results.loc[:,'Rsquared_test'], bins=30, 
         histtype='step', density=True, lw=5)
plt.hist(results.loc[:,'Rsquared_test_pseudo0'], bins=30, 
         histtype='step', density=True, lw=5)
plt.xlabel('r2')
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



plot_name = 'mediansignificantr2'
MIN_NUMBER_SESSIONS = 1
all_sigs = (all_pvalues<=0.05)
use_region = lambda reg: len(np.nonzero((all_regions==reg)&all_sigs)[0]) and len(np.nonzero(all_regions==reg)[0])>=MIN_NUMBER_SESSIONS
get_region_value = lambda reg: np.max(all_r2s[(all_regions==reg)&all_sigs])
get_region_err = lambda reg: np.std(all_r2s[(all_regions==reg)&all_sigs])

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
                cmap='Blues')
bar_results(acronyms,
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
get_region_value = lambda reg: len(all_r2s[(all_regions==reg)&all_sigs])
get_region_err = lambda reg: 0

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
                cmap='Blues')
bar_results(acronyms,
            values,
            errs,
            os.path.join(VARIABLE_FOLDER,
                          SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'bars', plot_name])) +
            FIGURE_SUFFIX))

plot_name = 'fsessionssignificant'
MIN_NUMBER_SESSIONS = 1
all_sigs = (all_pvalues<=0.05)
use_region = lambda reg: len(np.nonzero((all_regions==reg)&all_sigs)[0]) and len(np.nonzero(all_regions==reg)[0])>=MIN_NUMBER_SESSIONS
get_region_nsigsess = lambda reg: len(all_r2s[(all_regions==reg)&all_sigs])
get_region_nsess = lambda reg: len(all_r2s[all_regions==reg])
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
                value_title='min fraction of \nsignificant sessions \n(95\% confidence)')
bar_results(acronyms,
            values,
            errs,
            os.path.join(VARIABLE_FOLDER,
                          SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'bars', plot_name])) +
            FIGURE_SUFFIX))

sns.set_theme(style="whitegrid")

all_df = pd.DataFrame({'Target':np.concatenate(all_targets),
                       'Predictions':np.concatenate(all_preds),
                       'pLeft':np.concatenate(all_block_pLeft)})
index_max = np.argmax(all_r2s)
best_targets = all_targets[index_max]
best_preds = all_preds[index_max]
best_block_pLeft = all_block_pLeft[index_max]
best_actn = all_actn[index_max]
best_eid = all_eids[index_max]
best_probe = all_probes[index_max]
best_region = all_regions[index_max]

best_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds,
                       'pLeft':best_block_pLeft})

ci = 95

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', 
                 data=all_df, 
                 ci=ci, capsize=.2)
plt.ylim(-1,1)
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
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictionsBestR2_xtarget'])) +
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

plt.figure(figsize=(8,3.5))
plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
plt.plot(best_targets,'k')
plt.plot(best_preds,'blue')
plt.legend(['targets','predictions'])
plt.yticks([-1,0,1])
plt.ylim(-1,1)
plt.legend(['True','Predicted'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Prior')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH,
                         VARIABLE_FOLDER,
                         SPECIFIC_DECODING,
            ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2'])) +
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


