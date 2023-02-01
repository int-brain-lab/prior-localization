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

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

file_all_results = 'decoding_results/summary/18-01-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
ers = {f'{e}_{r}': p for e, r, p in zip(res_table['eid'],res_table['region'],res_table['p-value'])}

file_all_results = 'decoding_results/summary/19-01-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table_ephys = pd.read_csv(file_all_results)
ers_ephys = {f'{e}_{r}': p for e, r, p in zip(res_table_ephys['eid'],res_table_ephys['region'],res_table_ephys['p-value'])}

file_all_results = 'decoding_results/summary/20-01-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table_bcw = pd.read_csv(file_all_results)
ers_bcw = {f'{e}_{r}': p for e, r, p in zip(res_table_bcw['eid'],res_table_bcw['region'],res_table_bcw['p-value'])}

er_set = set(ers.keys())
ere_set = set(ers_ephys.keys())
erb_set = set(ers_bcw.keys())

ks = er_set.intersection(ere_set)
ks = ks.intersection(erb_set)
ks = list(ks)

er_scores = [ers[k] for k in ks]
ere_scores = [ers_ephys[k] for k in ks]
erb_scores = [ers_bcw[k] for k in ks]

plt.plot(er_scores)
plt.plot(ere_scores)
plt.plot(erb_scores)
plt.show()