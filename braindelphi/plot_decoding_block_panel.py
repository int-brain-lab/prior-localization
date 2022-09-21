#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:29:32 2022

@author: bensonb
"""
import numpy as np
import pandas as pd
from plot_utils import brain_SwansonFlat_results, bar_results, discretize_target
import os
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.metrics import r2_score



#%% Block
res_table = pd.read_csv('decoding_processing/10-09-2022_block.csv')

frac_sig_region = lambda reg: np.mean(np.array(res_table.loc[res_table['region']==reg,'p-value']<=0.05))
uni_regs = np.unique(res_table['region'])
fs_regs = [frac_sig_region(reg) for reg in uni_regs]


brain_SwansonFlat_results(uni_regs, fs_regs, 
                  filename='block_swanson_fs', 
                  cmap='Purples',
                  clevels=[None, None],
                  ticks=None,
                  extend=None,
                  value_title='Frac. Sig.')

def get_ms_reg(reg):
    c1 = (res_table['region']==reg)
    c2 = (res_table['p-value']<0.05)
    return np.median(res_table.loc[c1 & c2, 'score'])
# frac_sig_region = lambda reg: np.median(np.array(res_table.loc[res_table['region']==reg,'balanced_acc_test']))
ms_regs = [get_ms_reg(reg) for reg in uni_regs]

brain_SwansonFlat_results(uni_regs, ms_regs, 
                  filename='block_swanson_ms', 
                  cmap='Purples',
                  clevels=[None, None],
                  ticks=None,
                  extend=None,
                  value_title='Median Sig. \nScore')

n_reg = lambda reg: len(np.array(res_table.loc[res_table['region']==reg,'p-value']))
n_regs = [n_reg(reg) for reg in uni_regs]

brain_SwansonFlat_results(uni_regs, n_regs, 
                  filename='block_swanson_n', 
                  cmap='Purples',
                  clevels=[None, None],
                  ticks=None,
                  extend=None,
                  value_title='N Sessions')

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
get_nulls = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])

regions = np.unique(res_table['region'])
values = np.array([get_vals(reg) for reg in regions])
nulls = np.array([np.median(get_nulls(reg)) for reg in regions])
bar_results(regions, 
            values,
            nulls,
            'block_bars',
            YMIN=0.5,
            ylab='Bal. Acc.',
            TOP_N=15)

file = 'decoding_results/10-09-2022_decode_pLeft_oracle_Logistic_align_stimOn_times_100_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0.parquet'
res = pd.read_parquet(file)

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
plt.figure(figsize=(10,2.5))
plt.title(best_eid+' ['+best_probe+'] ['+best_region+']'+'\n accuracy$=$%.4f, $n=$%d'%(best_score,best_actn))
plt.plot(best_trials, best_targets, '-', c='k')
plt.plot(best_trials, best_probs, '-', c='indigo')
plt.yticks([0,.5,1])
#plt.ylim(0,1)
#plt.xlim(0,len(best_masks))
plt.legend(['True','Predicted Probability'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Block')
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
