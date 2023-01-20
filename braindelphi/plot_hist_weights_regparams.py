#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:03:18 2023

@author: bensonb
"""
import os
import numpy as np
import pandas as pd
from plot_utils import acronym2name, get_xy_vals, get_res_vals, brain_SwansonFlat_results, bar_results
from plot_utils import heatmap, activity_and_decoding_weights
from plot_utils import comb_regs_df, get_within_region_mean_var
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style('whitegrid')


DATE = '18-01-2023'
VARI = 'block'
file_all_results = 'decoding_results/summary/01-18-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = 'decoding_results/summary/01-18-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

cs = [x[:,:,0,1].reshape((50,)) for x in list(xy_table['params'])]
all_cs = np.concatenate(cs)
all_cs = [np.log10(float(c)) for c in all_cs]
ws = [x[:,:,0,:].reshape((50,-1)) for x in list(xy_table['weights'])]
all_ws = []
for w in ws:
    for i in range(50):
        all_ws.append(w[i,:])

plt.figure(figsize=(7,5))
ucs = np.unique(all_cs)
ws_fzeros = []
ws_cs = []
bins = np.linspace(0,
                   1.2*int(np.max(np.abs(np.concatenate(all_ws)))), 
                   10001)
for ui in range(len(ucs)):
    c = ucs[ui]
    inds = np.nonzero(np.array(all_cs) == c)[0]
    ws_c = np.concatenate([np.abs(all_ws[ind]) for ind in inds])
    ws_cs.append(ws_c)
    ws_fzeros.append(np.mean(ws_c==0))
    plt.hist(ws_c[np.nonzero(ws_c)[0]], 
             color=f'C{ui}', lw=2,
             bins=bins, histtype='step', density=False)
plt.legend([f'C={c:.0e}, frac. zeros = {ws_fzeros[i]:.3f}' for i, c in enumerate(np.power(10,ucs))], 
           fontsize=10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Weights')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_weights_separatereg.png',dpi=200)
plt.show()

all_ws = np.concatenate([np.ndarray.flatten(x) for x in list(xy_table['weights'])])
plt.figure(figsize=(7,5))
sns.histplot(np.abs(all_ws), stat="density")
plt.text(.11, 3, f'Fraction of weights \nequal to zero: \n{np.mean(np.abs(all_ws)==0.0):.3f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Weights')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_weights.png',dpi=200)
plt.show()

prms = np.ndarray.flatten(np.array([x[:,:,0,1] for x in list(xy_table['params'])]))
plt.figure(figsize=(8,5))
sns.histplot([np.log10(float(p)) for p in prms], 
             discrete=True, 
             stat="density")
xts = np.arange(-5,2, dtype=float)
plt.xticks(xts, labels=[f'{x:.1e}' for x in np.power(10,xts)], rotation=-20)
plt.xlabel('Regularization Coefficient')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_regprms.png',dpi=200)
plt.show()

#%%

DATE = '30-11-2022'
VARI = 'stimside'
file_all_results = 'decoding_results/summary/30-11-2022_decode_signcont_task_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = 'decoding_results/summary/30-11-2022_decode_signcont_task_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

all_ws = np.concatenate([np.ndarray.flatten(x) for x in list(xy_table['weights'])])
plt.figure(figsize=(7,5))
sns.histplot(np.abs(all_ws), stat="density")
plt.text(.11, 3, f'Fraction of weights \nequal to zero: \n{np.mean(np.abs(all_ws)==0.0):.3f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Weights')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_weights.png',dpi=200)
plt.show()

prms = np.ndarray.flatten(np.array([x[:,:,0,1] for x in list(xy_table['params'])]))
plt.figure(figsize=(8,5))
sns.histplot([np.log10(float(p)) for p in prms], 
             discrete=True, 
             stat="density")
xts = np.arange(-5,2, dtype=float)
plt.xticks(xts, labels=[f'{x:.1e}' for x in np.power(10,xts)], rotation=-20)
plt.xlabel('Regularization Coefficient')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_regprms.png',dpi=200)
plt.show()

#%%

DATE = '29-11-2022'
VARI = 'stim'
file_all_results = 'decoding_results/summary/29-11-2022_decode_signcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = 'decoding_results/summary/29-11-2022_decode_signcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

all_ws = np.concatenate([np.ndarray.flatten(x) for x in list(xy_table['weights'])])
plt.figure(figsize=(7,5))
sns.histplot(np.abs(all_ws), stat="density")
plt.text(.11, 3, f'Fraction of weights \nequal to zero: \n{np.mean(np.abs(all_ws)==0.0):.3f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Weights')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_weights.png',dpi=200)
plt.show()

prms = np.ndarray.flatten(np.array([x[:,:,0,1] for x in list(xy_table['params'])]))
plt.figure(figsize=(8,5))
sns.histplot([np.log10(float(p)) for p in prms], 
             discrete=True, 
             stat="density")
xts = np.arange(-5,2, dtype=float)
plt.xticks(xts, labels=[f'{x:.1e}' for x in np.power(10,xts)], rotation=-20)
plt.xlabel('Regularization Coefficient')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_regprms.png',dpi=200)
plt.show()

#%%

VARI = 'choice'
# DATE = '28-11-2022'
# file_all_results = 'decoding_results/summary/28-11-2022_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
# file_xy_results = 'decoding_results/summary/28-11-2022_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'
DATE = '18-01-2023'
file_all_results = 'decoding_results/summary/18-01-2023_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = 'decoding_results/summary/18-01-2023_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

cs = [x[:,:,0,1].reshape((50,)) for x in list(xy_table['params'])]
all_cs = np.concatenate(cs)
all_cs = [np.log10(float(c)) for c in all_cs]
ws = [x[:,:,0,:].reshape((50,-1)) for x in list(xy_table['weights'])]
all_ws = []
for w in ws:
    for i in range(50):
        all_ws.append(w[i,:])

plt.figure(figsize=(7,5))
ucs = np.unique(all_cs)
ws_fzeros = []
ws_cs = []
bins = np.linspace(0,
                   1.2*int(np.max(np.abs(np.concatenate(all_ws)))), 
                   10001)
for ui in range(len(ucs)):
    c = ucs[ui]
    inds = np.nonzero(np.array(all_cs) == c)[0]
    ws_c = np.concatenate([np.abs(all_ws[ind]) for ind in inds])
    ws_cs.append(ws_c)
    ws_fzeros.append(np.mean(ws_c==0))
    plt.hist(ws_c[np.nonzero(ws_c)[0]], 
             color=f'C{ui}', lw=2,
             bins=bins, histtype='step', density=False)
plt.legend([f'C={c:.0e}, frac. zeros = {ws_fzeros[i]:.3f}' for i, c in enumerate(np.power(10,ucs))], 
           fontsize=10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Weights')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_weights_separatereg.png',dpi=200)
plt.show()

all_ws = np.concatenate([np.ndarray.flatten(x) for x in list(xy_table['weights'])])
plt.figure(figsize=(7,5))
sns.histplot(np.abs(all_ws), stat="density")
plt.text(.11, 3, f'Fraction of weights \nequal to zero: \n{np.mean(np.abs(all_ws)==0.0):.3f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Weights')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_weights.png',dpi=200)
plt.show()

prms = np.ndarray.flatten(np.array([x[:,:,0,1] for x in list(xy_table['params'])]))
plt.figure(figsize=(8,5))
sns.histplot([np.log10(float(p)) for p in prms], 
             discrete=True, 
             stat="density")
xts = np.arange(-5,2, dtype=float)
plt.xticks(xts, labels=[f'{x:.1e}' for x in np.power(10,xts)], rotation=-20)
plt.xlabel('Regularization Coefficient')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_regprms.png',dpi=200)
plt.show()

#%%

DATE = '28-11-2022'
VARI = 'feedback'
file_all_results = 'decoding_results/summary/28-11-2022_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = 'decoding_results/summary/28-11-2022_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

all_ws = np.concatenate([np.ndarray.flatten(x) for x in list(xy_table['weights'])])
plt.figure(figsize=(7,5))
sns.histplot(np.abs(all_ws), stat="density")
plt.text(.11, 3, f'Fraction of weights \nequal to zero: \n{np.mean(np.abs(all_ws)==0.0):.3f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Weights')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_weights.png',dpi=200)
plt.show()

prms = np.ndarray.flatten(np.array([x[:,:,0,1] for x in list(xy_table['params'])]))
plt.figure(figsize=(8,5))
sns.histplot([np.log10(float(p)) for p in prms], 
             discrete=True, 
             stat="density")
xts = np.arange(-5,2, dtype=float)
plt.xticks(xts, labels=[f'{x:.1e}' for x in np.power(10,xts)], rotation=-20)
plt.xlabel('Regularization Coefficient')
plt.tight_layout()
plt.savefig(f'random_plots/{VARI}_hist_regprms.png',dpi=200)
plt.show()
