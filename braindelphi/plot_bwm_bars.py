#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:35:55 2023

@author: bensonb
"""
import numpy as np
import pandas as pd
from plot_utils import comb_regs_df, bar_results
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.5)
sns.set_style('ticks')

preamb = 'decoding_results/summary/'

#%% stimside

file_all_results = preamb + '02-04-2023_decode_signcont_task_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.loc[regs_table['combined_sig_corr'],'null_median_of_medians'])
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='stimside_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8], [0.5,0.6,0.7,0.8]),
                            #TOP_N=15,
                            sort_args=None, 
                            bolded_regions=[])
# check criteria.
for reg in acr_plotted:
    print(reg)
    assert np.median(get_vals(reg)) > np.median(res_table.loc[res_table['region']==reg, 'median-null'])

#%% choice

file_all_results = preamb + '01-04-2023_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.loc[regs_table['combined_sig_corr'],'null_median_of_medians'])
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='choice_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8,0.9,1.0], [0.5,0.6,0.7,0.8,0.9,1.0]),
                            #TOP_N=15,
                            sort_args=None, 
                            bolded_regions=[])
# check criteria.
for reg in acr_plotted:
    print(reg)
    assert np.median(get_vals(reg)) > np.median(res_table.loc[res_table['region']==reg, 'median-null'])

#%% feedback

file_all_results = preamb + '01-04-2023_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.loc[regs_table['combined_sig_corr'],'null_median_of_medians'])
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='feedback_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8,0.9,1.0], [0.5,0.6,0.7,0.8,0.9,1.0]),
                            #TOP_N=15,
                            sort_args=None, 
                            bolded_regions=[])
# check criteria.
for reg in acr_plotted:
    print(reg)
    assert np.median(get_vals(reg)) > np.median(res_table.loc[res_table['region']==reg, 'median-null'])

#%% block

file_all_results = preamb + '04-05-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_1000_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.loc[regs_table['combined_sig_corr'],'null_median_of_medians'])
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='block_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8], [0.5,0.6,0.7,0.8]),
                            #TOP_N=15,
                            sort_args=None, 
                            bolded_regions=[])
# check criteria.
for reg in acr_plotted:
    print(reg)
    assert np.median(get_vals(reg)) > np.median(res_table.loc[res_table['region']==reg, 'median-null'])

#%% wheel-speed

file_all_results = preamb + '01-04-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.loc[regs_table['combined_sig_corr'],'null_median_of_medians'])
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='wheelspeed_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='$R^2$',
                            # ticks=([0.5,0.6,0.7,0.8], [0.5,0.6,0.7,0.8]),
                            #TOP_N=15,
                            sort_args=None, 
                            bolded_regions=[])

#%% wheel-velocity

file_all_results = preamb + '01-04-2023_decode_wheel-vel_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

res_table = pd.read_csv(file_all_results)
regs_table = comb_regs_df(res_table, q_level=0.01,
                          USE_ALL_BERYL_REGIONS=False)
regions = np.array(regs_table.loc[regs_table['combined_sig_corr'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.loc[regs_table['combined_sig_corr'],'null_median_of_medians'])
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='wheelvel_bars.svg', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='$R^2$',
                            # ticks=([0.5,0.6,0.7,0.8], [0.5,0.6,0.7,0.8]),
                            #TOP_N=15,
                            sort_args=None, 
                            bolded_regions=[])