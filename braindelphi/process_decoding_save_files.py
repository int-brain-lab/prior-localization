#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 06:14:04 2022

@author: bensonb
"""
import numpy as np
import pandas as pd
from process_outputs import fix_pd_regions, create_pdtable_from_raw


DATE = '28-11-2022'
print('Working on Block')
file_pre = 'decoding_results/28-11-2022_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
#file_pre = 'decoding_results/27-10-2022_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
res = pd.DataFrame()
for i in range(50):
    res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
    res = pd.concat([res, res_new], axis=0)
#res = res.loc[res['eid']=='02fbb6da-3034-47d6-a61b-7d06c796a830']
res = fix_pd_regions(res)
print('working on table')
res_table, xy_table = create_pdtable_from_raw(res, 
                                    score_name='balanced_acc_test',
                                    N_PSEUDO=200,
                                    N_RUN=10,
                                    RETURN_X_Y=True)
valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
res_table = res_table.loc[valid_reg]
xy_table = xy_table.loc[valid_reg]
res_table.to_csv(f'decoding_processing/{DATE}_block.csv')
xy_table.to_pickle(f'decoding_processing/{DATE}_block_xy.pkl')

# DATE = '07-11-2022'
# print('Working on Stim')
# file_pre = 'decoding_results/28-10-2022_decode_signcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
#     res = pd.concat([res, res_new], axis=0)
# res = fix_pd_regions(res)
# res_table, xy_table = create_pdtable_from_raw(res, 
#                                     score_name='R2_test',
#                                     N_PSEUDO=200,
#                                     RETURN_X_Y=True)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# xy_table = xy_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_stim.csv')
# xy_table.to_pickle(f'decoding_processing/{DATE}_stim_xy.pkl')

# DATE = '07-11-2022'
# print('Working on Choice')
# file_pre = 'decoding_results/28-10-2022_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
#     res = pd.concat([res, res_new], axis=0)
# res = fix_pd_regions(res)
# res_table, xy_table = create_pdtable_from_raw(res, 
#                                     score_name='balanced_acc_test',
#                                     N_PSEUDO=200,
#                                     RETURN_X_Y=True)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# xy_table = xy_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_choice.csv')
# xy_table.to_pickle(f'decoding_processing/{DATE}_choice_xy.pkl')

# DATE = '07-11-2022'
# print('Working on Feedback')
# file_pre = 'decoding_results/27-10-2022_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
#     res = pd.concat([res, res_new], axis=0)
# res = fix_pd_regions(res)
# res = fix_pd_regions(res)
# res_table, xy_table = create_pdtable_from_raw(res, 
#                                     score_name='balanced_acc_test',
#                                     N_PSEUDO=200,
#                                     RETURN_X_Y=True)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# xy_table = xy_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_reward.csv')
# xy_table.to_pickle(f'decoding_processing/{DATE}_reward_xy.pkl')

# DATE = '07-11-2022'
# print('Working on Wheel-Speed')
# file_pre = 'decoding_results/27-10-2022_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
#     #res_new['eid'] = res_new['eid']+'_'+res_new['probe']
#     res = pd.concat([res, res_new], axis=0)
# res = fix_pd_regions(res)
# res_table = create_pdtable_from_raw(res, 
#                                     score_name='R2_test',
#                                     N_PSEUDO=100, N_RUN=2,
#                                     RETURN_X_Y=False,
#                                     SCALAR_PER_TRIAL=False)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# xy_table = xy_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_wheel-speed.csv')
# xy_table.to_pickle(f'decoding_processing/{DATE}_wheel-speed_xy.pkl')

# DATE = '02-11-2022'
# print('Working on Stimside')
# file_pre = 'decoding_results/02-11-2022_decode_signcont_task_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
#     res = pd.concat([res, res_new], axis=0)
# res = fix_pd_regions(res)
# res_table, xy_table = create_pdtable_from_raw(res, 
#                                     score_name='balanced_acc_test',
#                                     N_PSEUDO=200,
#                                     N_RUN=10,
#                                     RETURN_X_Y=True)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# xy_table = xy_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_stimside.csv')
# xy_table.to_pickle(f'decoding_processing/{DATE}_stimside_xy.pkl')

