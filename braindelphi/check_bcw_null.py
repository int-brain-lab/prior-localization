#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 06:14:04 2022

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
from sklearn.metrics import confusion_matrix
from scipy.stats import fisher_exact

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

file_all_results = 'decoding_results/summary/18-01-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
ers = {f'{e}_{r}': p for e, r, p in zip(res_table['eid'],res_table['region'],res_table['p-value'])}
erscores = {f'{e}_{r}': p for e, r, p in zip(res_table['eid'],res_table['region'],res_table['score'])}


file_all_results = 'decoding_results/summary/21-01-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table_ephys = pd.read_csv(file_all_results)
ers_ephys = {f'{e}_{r}': p for e, r, p in zip(res_table_ephys['eid'],res_table_ephys['region'],res_table_ephys['p-value'])}
erscore_ephys = {f'{e}_{r}': p for e, r, p in zip(res_table_ephys['eid'],res_table_ephys['region'],res_table_ephys['score'])}

file_all_results = 'decoding_results/summary/20-01-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table_bcw = pd.read_csv(file_all_results)
ers_bcw = {f'{e}_{r}': p for e, r, p in zip(res_table_bcw['eid'],res_table_bcw['region'],res_table_bcw['p-value'])}
erscore_bcw = {f'{e}_{r}': p for e, r, p in zip(res_table_bcw['eid'],res_table_bcw['region'],res_table_bcw['score'])}

er_set = set(ers.keys())
ere_set = set(ers_ephys.keys())
erb_set = set(ers_bcw.keys())

ks = er_set.intersection(ere_set)
ks = ks.intersection(erb_set)
ks = list(ks)

er_scores = np.array([ers[k] for k in ks])
ere_scores = np.array([ers_ephys[k] for k in ks])
erb_scores = np.array([ers_bcw[k] for k in ks])

M_self = confusion_matrix(er_scores<0.05, ere_scores<0.05) # run 1 value by run 2 value
M_btw = confusion_matrix(er_scores<0.05, erb_scores<0.05)

p_self = fisher_exact(M_self)[1]
p_btw = fisher_exact(M_btw)[1]

plt.title('Wheel-speed decoding \ncontingency between \ntwo ephys nulls')
sns.heatmap(M_self, vmin=0, vmax=np.sum(M_self), 
            cmap='viridis',
            annot=True)
plt.ylabel('Ephys 1')
plt.xlabel('Ephys 2')
plt.tight_layout()
plt.savefig('random_plots/checknull_ContingencyEphys.png')
plt.show()

plt.title('Wheel-speed decoding \ncontingency between \nephys and bias-choice-world nulls')
sns.heatmap(M_btw, vmin=0, vmax=np.sum(M_btw), 
            cmap='viridis',
            annot=True)
plt.ylabel('Ephys 1')
plt.xlabel('BCW')
plt.tight_layout()
plt.savefig('random_plots/checknull_ContingencyBCW.png')
plt.show()

plt.title('P-values')
plt.plot(er_scores, ere_scores,'o')
plt.plot(er_scores, erb_scores, 'o')
plt.xlabel('Ephys 1')
plt.ylabel('Other')
plt.legend(['Ephys 2', 'BCW'])
plt.show()

er_scores = np.array([erscores[k] for k in ks])
ere_scores = np.array([erscore_ephys[k] for k in ks])
erb_scores = np.array([erscore_bcw[k] for k in ks])

plt.title('R2 scores')
plt.plot(er_scores, ere_scores,'o')
plt.plot(er_scores, erb_scores, 'o')
plt.xlabel('Ephys 1')
plt.ylabel('Other')
plt.legend(['Ephys 2', 'BCW'])
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()