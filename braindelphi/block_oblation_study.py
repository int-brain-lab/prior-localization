#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 22:19:24 2023

@author: bensonb
"""
import numpy as np
import pandas as pd
from plot_utils import acronym2name, get_xy_vals, get_res_vals, brain_SwansonFlat_results, bar_results
from plot_utils import heatmap
from plot_utils import comb_regs_df, get_within_region_mean_var, get_predprob_vals
from yanliang_brain_slice_plot import get_cmap
import matplotlib.pyplot as plt
import seaborn as sns
from one.api import ONE
from brainwidemap.bwm_loading import bwm_units
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

# get reference cluster dataframe
julias_clusters = bwm_units(ONE(base_url='https://openalyx.internationalbrainlab.org',
                                password='international'))
julias_clusters['sessreg'] = julias_clusters.apply(lambda x: f"{x['eid']}_{x['Beryl']}", axis=1)
ref_clusters = julias_clusters[['uuids','sessreg']]

CUSTOM_SESSREG_FILTER = (10,1) # can use something other than ref_clusters
                             # if this is a tuple (min_units, min_reg)

'''
01-04-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv
'''

DATE = '01-05-2023'
VARI = 'block'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '01-05-2023_decode_pLeft_oracle_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_6_-0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'
FIG_SUF = '.svg'

FOCUS_REGIONS = ['ORBvl']

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1] for i in range(xy_table.shape[0])])

# filter results
if not (CUSTOM_SESSREG_FILTER is None):
    min_units, min_reg = CUSTOM_SESSREG_FILTER
    res_table = pd.read_csv(file_all_results)
    res_table = res_table.loc[res_table['n_units']>=min_units]
    res_table = res_table.loc[res_table['region']!='void']
    res_table = res_table.loc[res_table['region']!='root']
    reg_counts = res_table['region'].value_counts()
    res_table = res_table.loc[res_table['region'].isin(reg_counts[reg_counts>=min_reg].index)]
    
    xy_table = pd.read_pickle(file_xy_results)
    eid_regs_filtered = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
    xy_table = xy_table.loc[xy_table['eid_region'].isin(eid_regs_filtered)]
    
    # check clusters
    cuuids = np.concatenate(list(xy_table['cluster_uuids']))
    assert set(ref_clusters['uuids']).issubset(set(cuuids))
else:
    #filter according to reference  session_regions
    res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
    res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]
    xy_table = xy_table.loc[xy_table['eid_region'].isin(ref_clusters['sessreg'])]

    # check clusters
    cuuids = np.concatenate(list(xy_table['cluster_uuids']))
    assert set(cuuids) == set(ref_clusters['uuids'])

# combine regions and save
save_comb_regs_data = comb_regs_df(res_table, 
                                   USE_ALL_BERYL_REGIONS=True)
regs_table = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=False)
n_sig = regs_table['combined_sig'].sum()
f_sig = regs_table['combined_sig'].mean()
wi_means, wi_vars = get_within_region_mean_var(res_table)
save_comb_regs_data.to_csv(
    f'decoding_processing/{DATE}_{VARI}_regs_nsig{n_sig}_fsig{f_sig:.3f}_wi2ovar{np.mean(wi_vars)/np.var(wi_means):.3f}.csv')

