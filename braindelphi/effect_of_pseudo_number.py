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
plt.title('Effect of near-zero p-value sessions (orange)')
xs = np.linspace(0,1)
plt.plot(old_res['combined_p-value_corr'][~is_edge_reg], 
         new_res['combined_p-value_corr'][~is_edge_reg], 'C0o') 
plt.plot(old_res['combined_p-value_corr'][is_edge_reg], 
         new_res['combined_p-value_corr'][is_edge_reg], 'C1o')
for x, y, s in zip(old_res['combined_p-value_corr'][is_edge_reg], 
                new_res['combined_p-value_corr'][is_edge_reg],
                old_res['region'][is_edge_reg]):
    plt.text(x, y, s, fontsize=12, color='k')
plt.plot(xs, xs, 'k--', lw=1)
plt.xlabel('200 pseudo p-value (fisher combined and FDR corrected')
plt.ylabel('1000 pseudo p-value (fisher combined and FDR corrected)')
plt.savefig('decoding_figures/SI/effect_of_npseudo_with_near-zero_pvals.png', dpi=200)
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

plt.figure(figsize=(5,4))
plt.title('p-values across all sessions')
plt.hist(old_res_all['p-value'],
         bins=100,
         density=True,
         histtype='step')

plt.hist(new_res_all['p-value'],
         bins=100,
         density=True,
         histtype='step')
plt.legend(['npseudo=200', 'nspeudo=1000'])
plt.xlabel('p-value')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('decoding_figures/SI/pvalues_block_allsessions.png', 
            dpi=200)
plt.show()

plt.figure(figsize=(5,4))
plt.title('combined p-values across all regions')
plt.hist(old_res['combined_p-value'],
         bins=30,
         density=True,
         histtype='step')

plt.hist(new_res['combined_p-value'],
         bins=30,
         density=True,
         histtype='step')
plt.hist(old_res['combined_p-value_corr'],
         bins=30,
         density=True,
         histtype='step')

plt.hist(new_res['combined_p-value_corr'],
         bins=30,
         density=True,
         histtype='step')
plt.legend(['npseudo=200', 'nspeudo=1000',
            'npseudo=200 with FDR', 'nspeudo=1000 with FDR'])
plt.xlabel('p-value')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('decoding_figures/SI/combcorrpvalues_block_allregions.png', 
            dpi=200)
plt.show()

