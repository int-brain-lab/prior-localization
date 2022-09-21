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
                            N_PSEUDO=90):
    
    res_table = []
    
    for eid in np.unique(res['eid']):
        reseid = res.loc[res['eid']==eid]
        for reg in np.unique(reseid['region']):
            assert len(reg) == 1
            reg = reg[0]
            
            reseidreg = reseid.loc[reseid['region']==reg]
            
            pids = np.sort(np.unique(reseidreg['pseudo_id']))
            if len(pids) == N_PSEUDO+1:
                assert pids[0] == -1
                assert np.all(pids[1:] == np.arange(1,N_PSEUDO+1))
                score = np.mean(reseidreg.loc[reseidreg['pseudo_id']==-1,score_name])
            
                p_scores = [np.mean(reseidreg.loc[reseidreg['pseudo_id']==pid,score_name]) for pid in pids[1:]]
                median_null = np.median(p_scores)
                pval = np.mean(np.array(p_scores)>score)
                
                res_table.append([eid,reg,score,pval,median_null])
                
    res_table = pd.DataFrame(res_table, columns=['eid',
                                                 'region',
                                                 'score',
                                                 'p-value',
                                                 'median-null'])
    return res_table

DATE = '10-09-2022'
file = 'decoding_results/10-09-2022_decode_pLeft_oracle_Logistic_align_stimOn_times_100_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0.parquet'
res = pd.read_parquet(file)
res_table = create_pdtable_from_raw(res)
res_table.to_csv(f'decoding_processing/{DATE}_block.csv')
