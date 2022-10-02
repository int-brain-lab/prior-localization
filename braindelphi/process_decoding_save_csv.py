#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 06:14:04 2022

@author: bensonb
"""
import numpy as np
import pandas as pd

def create_pdtable_from_raw(res, 
                            score_name='balanced_acc_test',
                            N_PSEUDO=200):
    
    res_table = []
    
    for eid in np.unique(res['eid']):
        reseid = res.loc[res['eid']==eid]
        subject = np.unique(reseid['subject'])
        assert len(subject) == 1
        subject = subject[0]
        for reg in np.unique(reseid['region']):
            assert len(reg) == 1
            reg = reg[0]
            
            reseidreg = reseid.loc[reseid['region']==reg]
            
            pids = np.sort(np.unique(reseidreg['pseudo_id']))
            if len(pids) == N_PSEUDO+1:
                assert pids[0] == -1
                assert np.all(pids[1:] == np.arange(1,N_PSEUDO+1))
                real_scores = reseidreg.loc[reseidreg['pseudo_id']==-1,score_name]
                assert len(real_scores) == 10 # 10 repeats of decoding to reduce variance
                score = np.mean(real_scores)
            
                p_scores = [np.mean(reseidreg.loc[reseidreg['pseudo_id']==pid,score_name]) for pid in pids[1:]]
                median_null = np.median(p_scores)
                pval = np.mean(np.array(p_scores)>score)
                
                
                res_table.append([subject,eid,reg,score,pval,median_null])
                
    res_table = pd.DataFrame(res_table, columns=['subject',
                                                 'eid',
                                                 'region',
                                                 'score',
                                                 'p-value',
                                                 'median-null'])
    return res_table

DATE = '20-09-2022'
file = 'decoding_results/20-09-2022_decode_pLeft_oracle_Logistic_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0.parquet'
res = pd.read_parquet(file)
res_table = create_pdtable_from_raw(res, 
                                    score_name='balanced_acc_test',
                                    N_PSEUDO=200)
valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
res_table = res_table.loc[valid_reg]
res_table.to_csv(f'decoding_processing/{DATE}_block.csv')

file_pre = 'decoding_results/20-09-2022_decode_strengthcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0_paraindex'
res = pd.DataFrame()
for i in range(50):
    res_new = pd.read_parquet(file_pre+str(i)+'.parquet')
    res = pd.concat([res, res_new], axis=0)
res_table = create_pdtable_from_raw(res, 
                                    score_name='R2_test',
                                    N_PSEUDO=200)
valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
res_table = res_table.loc[valid_reg]
res_table.to_csv(f'decoding_processing/{DATE}_stim.csv')

file_pre = 'decoding_results/20-09-2022_decode_strengthcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0_paraindex'
res = pd.DataFrame()
for i in range(50):
    res_new = pd.read_parquet(file_pre+str(i)+'.parquet')
    res = pd.concat([res, res_new], axis=0)
res_table = create_pdtable_from_raw(res, 
                                    score_name='balanced_acc_test',
                                    N_PSEUDO=200)
valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
res_table = res_table.loc[valid_reg]
res_table.to_csv(f'decoding_processing/{DATE}_choice.csv')

# file_pre = 'decoding_results/20-09-2022_decode_strengthcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_parquet(file_pre+str(i)+'.parquet')
#     res = pd.concat([res, res_new], axis=0)
# res_table = create_pdtable_from_raw(res, 
#                                     score_name='balanced_acc_test',
#                                     N_PSEUDO=200)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_reward.csv')
