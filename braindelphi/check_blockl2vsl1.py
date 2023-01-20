#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:26:04 2023

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

VARI = 'block'

DATE = '28-11-2022'
file_all_results = 'decoding_results/summary/28-11-2022_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
regs_data_old = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=True)

DATE = '18-01-2023'
file_all_results = 'decoding_results/summary/01-18-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
regs_data = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=True)

plt.figure(figsize=(7,6))
plt.plot(list(regs_data_old['values_median']),
         list(regs_data['values_median']), '.')
plt.xlabel('Median decoding score, \nL2 regularization')
plt.ylabel('Median decoding score, \nL1 regularization')
plt.tight_layout()
plt.savefig(f'random_plots/check_l1vsl2_{VARI}_medvals.png',dpi=200)
plt.show()

plt.figure(figsize=(7,6))
plt.plot(list(regs_data_old['values_median_sig']),
         list(regs_data['values_median_sig']), '.')
plt.xlabel('Median decoding score of Sig Sess, \nL2 regularization')
plt.ylabel('Median decoding score of Sig Sess, \nL1 regularization')
plt.tight_layout()
plt.savefig(f'random_plots/check_l1vsl2_{VARI}_medsigvals.png',dpi=200)
plt.show()

plt.figure(figsize=(7,6))
plt.plot(list(regs_data_old['n_units_mean']),
         list(regs_data['n_units_mean']), '.')
plt.xlabel('Average N units, \nL2 regularization')
plt.ylabel('Average N units, \nL1 regularization')
plt.tight_layout()
plt.savefig(f'random_plots/check_l1vsl2_{VARI}_nunits.png',dpi=200)
plt.show()

plt.figure(figsize=(7,6))
plt.plot(list(regs_data_old['frac_sig']),
         list(regs_data['frac_sig']), '.')
plt.xlabel('Frac Sig, \nL2 regularization')
plt.ylabel('Frac Sig, \nL1 regularization')
plt.tight_layout()
plt.savefig(f'random_plots/check_l1vsl2_{VARI}_fracsig.png',dpi=200)
plt.show()

plt.figure(figsize=(15,14))
plt.plot(list(regs_data_old['combined_p-value']),
         list(regs_data['combined_p-value']), 'o', ms=5)
plt.xlabel('Combined p-value, \nL2 regularization')
plt.ylabel('Combned p-value, \nL1 regularization')
xs = np.linspace(0,1)
plt.plot(xs, 0.05*np.ones(len(xs)), 'r--')
plt.plot(0.05*np.ones(len(xs)), xs, 'r--')
plt.tight_layout()
plt.savefig(f'random_plots/check_l1vsl2_{VARI}_combpval.png',dpi=200)
plt.show()

#%%
VARI = 'choice'

DATE = '28-11-2022'
file_all_results = 'decoding_results/summary/28-11-2022_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
regs_data_old = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=True)

DATE = '18-01-2023'
file_all_results = 'decoding_results/summary/18-01-2023_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
res_table = pd.read_csv(file_all_results)
regs_data = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=True)

plt.figure(figsize=(7,6))
plt.plot(list(regs_data_old['values_median']),
         list(regs_data['values_median']), '.')
plt.xlabel('Median decoding score, \nL2 regularization')
plt.ylabel('Median decoding score, \nL1 regularization')
plt.tight_layout()
plt.savefig(f'random_plots/check_l1vsl2_{VARI}_medvals.png',dpi=200)
plt.show()

plt.figure(figsize=(7,6))
plt.plot(list(regs_data_old['values_median_sig']),
         list(regs_data['values_median_sig']), '.')
plt.xlabel('Median decoding score of Sig Sess, \nL2 regularization')
plt.ylabel('Median decoding score of Sig Sess, \nL1 regularization')
plt.tight_layout()
plt.savefig(f'random_plots/check_l1vsl2_{VARI}_medsigvals.png',dpi=200)
plt.show()

plt.figure(figsize=(7,6))
plt.plot(list(regs_data_old['n_units_mean']),
         list(regs_data['n_units_mean']), '.')
plt.xlabel('Average N units, \nL2 regularization')
plt.ylabel('Average N units, \nL1 regularization')
plt.tight_layout()
plt.savefig(f'random_plots/check_l1vsl2_{VARI}_nunits.png',dpi=200)
plt.show()

plt.figure(figsize=(7,6))
plt.plot(list(regs_data_old['frac_sig']),
         list(regs_data['frac_sig']), '.')
plt.xlabel('Frac Sig, \nL2 regularization')
plt.ylabel('Frac Sig, \nL1 regularization')
plt.tight_layout()
plt.savefig(f'random_plots/check_l1vsl2_{VARI}_fracsig.png',dpi=200)
plt.show()

plt.figure(figsize=(15,14))
plt.plot(list(regs_data_old['combined_p-value']),
         list(regs_data['combined_p-value']), 'o', ms=5)
plt.xlabel('Combined p-value, \nL2 regularization')
plt.ylabel('Combned p-value, \nL1 regularization')
xs = np.linspace(0,1)
plt.plot(xs, 0.05*np.ones(len(xs)), 'r--')
plt.plot(0.05*np.ones(len(xs)), xs, 'r--')
plt.tight_layout()
plt.savefig(f'random_plots/check_l1vsl2_{VARI}_combpval.png',dpi=200)
plt.show()