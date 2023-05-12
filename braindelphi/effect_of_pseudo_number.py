#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 23:40:28 2023

@author: bensonb
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

preamb_comb = 'decoding_processing/'
new_comb = '04-05-2023_block_regs_nsig19_fsig0.154_wi2ovar2.248.csv'
old_comb = '01-04-2023_block_regs_nsig19_fsig0.154_wi2ovar2.248.csv'

preamb_all = 'decoding_results/summary/'
new_all = '04-05-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_1000_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
old_all = '01-04-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

new_res = pd.read_csv(preamb_comb + new_comb)
old_res = pd.read_csv(preamb_comb + old_comb)

new_res_all = pd.read_csv(preamb_all + new_all)
old_res_all = pd.read_csv(preamb_all + old_all)

edge_regs = np.unique(new_res_all['region'][(new_res_all['p-value']<0.001) & (old_res_all['p-value']<0.005)])
is_edge_reg = np.isin(new_res['region'], edge_regs)

plt.figure(figsize=(5,5))
xs = np.linspace(0,1)
plt.plot(old_res['combined_p-value_corr'][~is_edge_reg], 
         new_res['combined_p-value_corr'][~is_edge_reg], 'C0o') 
plt.plot(old_res['combined_p-value_corr'][is_edge_reg], 
         new_res['combined_p-value_corr'][is_edge_reg], 'C1o')
plt.plot(xs, xs, 'k--', lw=1)
plt.xlabel('200 pseudo p-value')
plt.ylabel('1000 pseudo p-value')
plt.show()

plt.figure(figsize=(5,5))
xs = np.linspace(0,1)
plt.plot(old_res['combined_p-value'], 
         new_res['combined_p-value'], 'C0o')
plt.plot(old_res['combined_p-value'][is_edge_reg], 
         new_res['combined_p-value'][is_edge_reg], 'C1o')
plt.plot(xs, xs, 'k--', lw=1)
plt.xlabel('200 pseudo p-value')
plt.ylabel('1000 pseudo p-value')
plt.show()

plt.figure(figsize=(5,5))
xs = np.linspace(0,1)
plt.plot(old_res_all['p-value'], 
         new_res_all['p-value'], 'o')
plt.plot(xs, xs, 'k--', lw=1)
plt.xlabel('200 pseudo')
plt.ylabel('1000 pseudo')
plt.show()

