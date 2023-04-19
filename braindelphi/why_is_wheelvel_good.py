#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:28:46 2023

@author: bensonb
"""
import os
import numpy as np
import pandas as pd
from plot_utils import acronym2name, get_xy_vals, get_res_vals, brain_SwansonFlat_results, bar_results
from plot_utils import heatmap, activity_and_decoding_weights
from plot_utils import comb_regs_df, get_within_region_mean_var
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import seaborn as sns
from one.api import ONE
from brainwidemap.bwm_loading import bwm_units
sns.set(font_scale=1.5)

# load wv, ws, c region sets

# get reference cluster dataframe
julias_clusters = bwm_units(ONE(base_url='https://openalyx.internationalbrainlab.org',
                                password='international'))
julias_clusters['sessreg'] = julias_clusters.apply(lambda x: f"{x['eid']}_{x['Beryl']}", axis=1)
ref_clusters = julias_clusters[['uuids','sessreg']]

DATE = '01-04-2023'
VARI = 'wheel-vel'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '01-04-2023_decode_wheel-vel_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1]/11 for i in range(xy_table.shape[0])])

#filter according to reference  session_regions
res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]
xy_table = xy_table.loc[xy_table['eid_region'].isin(ref_clusters['sessreg'])]

# check clusters
cuuids = np.concatenate(list(xy_table['cluster_uuids']))
assert set(cuuids) == set(ref_clusters['uuids'])

regs_table = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=False)
regs_table_wv = regs_table
regs_wv = set(regs_table_wv[regs_table_wv['combined_sig']]['region'])

DATE = '01-04-2023'
VARI = 'wheel-speed'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '01-04-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1]/11 for i in range(xy_table.shape[0])])

#filter according to reference  session_regions
res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]
xy_table = xy_table.loc[xy_table['eid_region'].isin(ref_clusters['sessreg'])]

# check clusters
cuuids = np.concatenate(list(xy_table['cluster_uuids']))
assert set(cuuids) == set(ref_clusters['uuids'])

regs_table = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=False)
regs_table_ws = regs_table
regs_ws = set(regs_table_ws[regs_table_ws['combined_sig']]['region'])

DATE = '01-04-2023'
VARI = 'choice'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '01-04-2023_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1] for i in range(xy_table.shape[0])])

#filter according to reference  session_regions
res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]
xy_table = xy_table.loc[xy_table['eid_region'].isin(ref_clusters['sessreg'])]

# check clusters
cuuids = np.concatenate(list(xy_table['cluster_uuids']))
assert set(cuuids) == set(ref_clusters['uuids'])

# combine regions and save
regs_table = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=False)
regs_table_c = regs_table
regs_c = set(regs_table_c[regs_table_c['combined_sig']]['region'])

#%%
# Plot the Venn diagram
venn3([regs_ws, regs_wv, regs_c], ('Wheel-speed', 'Wheel-velocity', 'Choice'))
plt.savefig('decoding_figures/wheelspeed_vel_choice_venndiag.png')
plt.show()

#%%
assert np.all(regs_table_wv['region'] == regs_table_ws['region'])

sns.set_style('ticks')

plt.figure(figsize=(8,8))
# plt.plot(regs_table_wv['valuesminusnull_median'], 
#          regs_table_ws['valuesminusnull_median'],
#          'o')
for x, y, s in zip(regs_table_wv['valuesminusnull_median'],
                   regs_table_ws['valuesminusnull_median'],
                   regs_table_wv['region']):
    plt.text(x, y, s, fontsize=7)
xs = np.linspace(0,0.4)
plt.plot(xs,xs,'k--')
plt.xlabel('Wheel-velocity')
plt.ylabel('Wheel-speed')
plt.tight_layout()
plt.savefig('decoding_figures/wheelspeedvsvel_effectsize_scatter.png', dpi=200)
plt.show()

plt.figure(figsize=(8,8))
# plt.plot(regs_table_wv['valuesminusnull_median'], 
#          regs_table_ws['valuesminusnull_median'],
#          'o')
for x, y, s in zip(regs_table_wv['values_median'],
                   regs_table_ws['values_median'],
                   regs_table_wv['region']):
    plt.text(x, y, s, fontsize=7)
xs = np.linspace(0,0.4)
plt.plot(xs,xs,'k--')
plt.xlabel('Wheel-velocity')
plt.ylabel('Wheel-speed')
plt.tight_layout()
plt.savefig('decoding_figures/wheelspeedvsvel_r2_scatter.png', dpi=200)
plt.show()

plt.figure(figsize=(10,8))
# plt.plot(regs_table_wv['valuesminusnull_median'], 
#          regs_table_ws['valuesminusnull_median'],
#          'o')
for x, y, s in zip(regs_table_wv['null_median_of_medians'],
                   regs_table_ws['null_median_of_medians'],
                   regs_table_wv['region']):
    plt.text(x, y, s, fontsize=7)
xs = np.linspace(0,0.25)
plt.plot(xs,xs,'k--')
plt.xlabel('Wheel-velocity')
plt.ylabel('Wheel-speed')
plt.xlim(0,0.021)
plt.ylim(0,0.23)
plt.tight_layout()
plt.savefig('decoding_figures/wheelspeedvsvel_r2null_scatter.png', dpi=200)
plt.show()

# plt.figure(figsize=(10,8))
# # plt.plot(regs_table_wv['valuesminusnull_median'], 
# #          regs_table_ws['valuesminusnull_median'],
# #          'o')
# for x, y, s in zip(regs_table_wv['null_median_of_medians'],
#                    regs_table_ws['n_units_mean'],
#                    regs_table_wv['region']):
#     plt.text(x, y, s, fontsize=7)
# xs = np.linspace(0,0.25)
# plt.plot(xs,xs,'k--')
# plt.xlabel('Wheel-velocity')
# plt.ylabel('Wheel-speed')
# plt.xlim(0,0.021)
# plt.ylim(0,0.23)
# plt.tight_layout()
# plt.savefig('decoding_figures/wheelspeedvsnunits_scatter.png', dpi=200)
# plt.show()