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
from sklearn.metrics import r2_score
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

# get targets
xy_eids = xy_table.apply(lambda x: x['eid_region'].split('_')[0], axis=1)
all_targs_wv = np.vstack([xy_table.loc[xy_eids[xy_eids==eid].index[0], 
                          'targets'] for eid in xy_eids.unique()])


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

# get targets
all_targs_ws = np.vstack([xy_table.loc[xy_eids[xy_eids==eid].index[0], 
                          'targets'] for eid in xy_eids.unique()])

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
plt.plot(regs_table_wv['valuesminusnull_median'], 
          regs_table_ws['valuesminusnull_median'],
          'o', ms=1)
for x, y, s in zip(regs_table_wv['valuesminusnull_median'],
                   regs_table_ws['valuesminusnull_median'],
                   regs_table_wv['region']):
    plt.text(x, y, s, fontsize=9)
xs = np.linspace(0,0.55)
plt.plot(xs,xs,'k--')
plt.xlabel('Wheel-velocity')
plt.ylabel('Wheel-speed')
plt.tight_layout()
plt.savefig('decoding_figures/SI/wheelspeedvsvel_effectsize_scatter.svg')
plt.show()

plt.figure(figsize=(8,8))
plt.plot(regs_table_wv['values_median'],
                   regs_table_ws['values_median'],
          'o', ms=1)
for x, y, s in zip(regs_table_wv['values_median'],
                   regs_table_ws['values_median'],
                   regs_table_wv['region']):
    plt.text(x, y, s, fontsize=9)
xs = np.linspace(0,0.55)
plt.plot(xs,xs,'k--')
plt.xlabel('Wheel-velocity')
plt.ylabel('Wheel-speed')
plt.tight_layout()
plt.savefig('decoding_figures/SI/wheelspeedvsvel_r2_scatter.svg')
plt.show()

plt.figure(figsize=(8,8))
plt.plot(regs_table_wv['null_median_of_medians'],
                   regs_table_ws['null_median_of_medians'],
          'o', ms=1)
for x, y, s in zip(regs_table_wv['null_median_of_medians'],
                   regs_table_ws['null_median_of_medians'],
                   regs_table_wv['region']):
    plt.text(x, y, s, fontsize=9)
# xs = np.linspace(0,0.25)
# plt.plot(xs,xs,'k--')
plt.xlabel('Wheel-velocity')
plt.ylabel('Wheel-speed')
# plt.xlim(0,0.021)
# plt.ylim(0,0.23)
plt.tight_layout()
plt.savefig('decoding_figures/SI/wheelspeedvsvel_r2null_scatter.svg')
plt.show()

plt.figure(figsize=(8,8))
plt.plot(regs_table_ws['null_median_of_medians'], 
          regs_table_ws['n_units_mean'],
          'o', ms=1)
for x, y, s in zip(regs_table_wv['null_median_of_medians'],
                    regs_table_wv['n_units_mean'],
                    regs_table_wv['region']):
    plt.text(x, y, s, fontsize=7)
# xs = np.linspace(0,0.25)
# plt.plot(xs,xs,'k--')
plt.xlabel('Wheel-velocity')
plt.ylabel('Wheel-speed')
# plt.xlim(0,0.021)
# plt.ylim(0,0.23)
plt.tight_layout()
# plt.savefig('decoding_figures/wheelspeedvsnunits_scatter.png', dpi=200)
plt.show()

plt.figure(figsize = (6,6))
ts = np.arange(60)*0.020 - 0.2
plt.plot(ts, np.median(all_targs_ws, axis=0), lw=3)
plt.plot(ts, np.median(all_targs_wv, axis=0), lw=3)
plt.fill_between(ts, 
                 np.percentile(all_targs_ws, 5, axis=0),
                 np.percentile(all_targs_ws, 95, axis=0), 
                 alpha=0.2, color='C0')
plt.fill_between(ts, 
                 np.percentile(all_targs_wv, 5, axis=0),
                 np.percentile(all_targs_wv, 95, axis=0), 
                 alpha=0.2, color='C1')
plt.legend(['wheel-speed', 'wheel-velocity'], fontsize=14)
plt.xlabel('Time (s)')
plt.ylabel('Speed/velocity')
plt.tight_layout()
plt.savefig('decoding_figures/SI/wheelspeedvsvel_stereoshape.svg')
plt.show()

#%%

n_boot = 50
n_iter = 1000
n_trials = 400

stereo_ws = np.median(all_targs_ws, axis=0)
stereo_ws = np.concatenate([stereo_ws] * n_trials)
stereo_wv = np.median(all_targs_wv, axis=0)
stereo_wv = np.concatenate([stereo_wv] * n_trials)

n_ws = all_targs_ws.shape[0]
sampletrials_ws = lambda : np.concatenate(all_targs_ws[np.random.randint(0,
                                                    n_ws,
                                                    size=n_trials),:])
n_wv = all_targs_wv.shape[0]
sampletrials_wv = lambda : np.concatenate(all_targs_wv[np.random.randint(0,
                                                    n_wv,
                                                    size=n_trials),:])

r2_wss = []
r2_wvs = []
for _ in range(n_boot):
    r2s = [r2_score(sampletrials_ws(), stereo_ws) for _ in range(n_iter)]
    r2_wss.append(np.mean(r2s))
    
    r2s = [r2_score(sampletrials_wv(), stereo_wv) for _ in range(n_iter)]
    r2_wvs.append(np.mean(r2s))

r2mu_wv = np.mean(r2_wvs)
r2err_wv = np.std(r2_wvs)/np.sqrt(n_boot)
r2mu_ws = np.mean(r2_wss)
r2err_ws = np.std(r2_wss)/np.sqrt(n_boot)