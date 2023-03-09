#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:42:36 2023

@author: bensonb
"""
import numpy as np
import pandas as pd

DATE = '18-01-2023'
VARI = 'block'
out = pd.read_csv(
    f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
block_cuuids = list(out['cluster_uuids'])

DATE = '18-01-2023'
VARI = 'stim'
out = pd.read_csv(
    f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
stim_cuuids = list(out['cluster_uuids'])

DATE = '18-01-2023'
VARI = 'choice'
out = pd.read_csv(
    f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
choice_cuuids = list(out['cluster_uuids'])

DATE = '18-01-2023'
VARI = 'feedback'
out = pd.read_csv(
    f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
feedback_cuuids = list(out['cluster_uuids'])

# DATE = '18-01-2023'
# VARI = 'wheel-speed'
# out = pd.read_csv(
#     f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
# wheel_cuuids = list(out['cluster_uuids'])

intersect_cuuids = list(set(block_cuuids).intersection(set(stim_cuuids)).intersection(set(choice_cuuids)).intersection(set(feedback_cuuids)))

pd.DataFrame(intersect_cuuids, 
             columns=['cluster_uuid']).to_csv('BWM_cluster_uuids.csv', 
                                              index=False)
#%%                                           
# check that every eid_region has trial inclustion criteria that are the same
# check that mask_diagnostics are the same

file_all_results = 'decoding_results/summary/01-18-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = 'decoding_results/summary/01-18-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'
res_table0 = pd.read_csv(file_all_results)
xy_table0 = pd.read_pickle(file_xy_results)

file_all_results = 'decoding_results/summary/18-01-2023_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = 'decoding_results/summary/18-01-2023_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

er = xy_table0.iloc[0]['eid_region']
md0 = xy_table0.iloc[0]['mask_diagnostics']
assert np.all(np.std(np.stack(md0[0]), axis=0) == 0) # make sure all repeats are the same
assert np.all(np.std(np.stack(md0[1]), axis=0) == 0)

md = xy_table.query('`eid_region` == @er').iloc[0]['mask_diagnostics']

md0_mask = md0[0][0]
md0_diag = md0[1][0]
md_mask = md[0][0]
md_diag = md[1][0]

# proof that trial inclusion criteria vary across variables.
assert np.all(md0_mask == md_mask)
