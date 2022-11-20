#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 06:14:04 2022

@author: bensonb
"""
import numpy as np
import pandas as pd

def gini(x, weights=None):
    '''
    Implementation copied from 
    https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    weights : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if weights is None:
        weights = np.ones_like(x)
    # Calculate mean absolute deviation in two steps, for weights.
    count = np.multiply.outer(weights, weights)
    mad = np.abs(np.subtract.outer(x, x) * count).sum() / count.sum()
    rmad = mad / np.average(x, weights=weights)
    # Gini equals half the relative mean absolute deviation.
    return 0.5 * rmad

def create_pdtable_from_raw(res, 
                            score_name='balanced_acc_test',
                            N_PSEUDO=200, N_RUN=10, 
                            N_PSEUDO_LOWER_THRESH = np.infty):
    
    res = res.reset_index()
    for i, r in enumerate(res.loc[:,'region']):
        if type(r) is list:
            assert len(r) == 1
            res.loc[i,'region'] = r[0]
    
    
    res_table = []
    
    for eid in np.unique(res['eid']):
        reseid = res.loc[res['eid']==eid]
        subject = np.unique(reseid['subject'])
        assert len(subject) == 1
        subject = subject[0]
        
        #print(reseid['region'])
        for reg in np.unique(reseid['region']):
            # assert len(reg) == 1
            # reg = reg[0]
            
            reseidreg = reseid.loc[reseid['region']==reg]
            eidreg_probes = np.unique(reseidreg['probe'])
            if not (len(eidreg_probes)==1):
                print(eidreg_probes)
                assert (eidreg_probes[0]=='probe00') and (eidreg_probes[1]=='probe01')
                assert len(eidreg_probes)==2
                cur_p = np.random.choice(['probe00','probe01'])
                reseidreg = reseidreg.loc[reseidreg['probe']==cur_p]
            
            pids = np.sort(np.unique(reseidreg['pseudo_id']))
            #print(reseidreg.head())
            if len(pids) == N_PSEUDO+1:
                assert pids[0] == -1
                assert np.all(pids[1:] == np.arange(1,N_PSEUDO+1))
                real_scores = reseidreg.loc[reseidreg['pseudo_id']==-1,score_name]
                ws = list(reseidreg.loc[reseidreg['pseudo_id']==-1, 'weights'])
                assert len(ws)==N_RUN
                ws = np.abs(np.ndarray.flatten(np.array(ws)))
                #print(ws)
                frac_lg_w = np.mean(ws > 0.1)#1.0/len(ws))
                gini_w = gini(ws)
                assert len(real_scores) == N_RUN # 10 repeats of decoding to reduce variance
                score = np.mean(real_scores)
                
                # include real score in null scores
                p_scores = [np.mean(reseidreg.loc[reseidreg['pseudo_id']==pid,score_name]) for pid in pids]#[1:]
                try:
                    assert not np.any(np.isnan(p_scores))
                except:
                    print(f'{eid}, region {reg} has {np.sum(np.isnan(p_scores))} scores which are nan')
                    continue
                median_null = np.median(p_scores)
                pval = np.mean(np.array(p_scores)>=score)
                n_units = np.array(reseidreg.loc[reseidreg['pseudo_id']==-1,'N_units'])
                assert np.all(n_units == n_units[0])
                n_units = n_units[0]
                
                res_table.append([subject,eid,reg,score,pval,median_null,n_units,frac_lg_w,gini_w])
            elif len(pids) >= N_PSEUDO_LOWER_THRESH+1 and pids[0] == -1:
                real_scores = reseidreg.loc[reseidreg['pseudo_id']==-1,score_name]
                
                ws = list(reseidreg.loc[reseidreg['pseudo_id']==-1, 'weights'])
                assert len(ws)==N_RUN
                ws = np.abs(np.ndarray.flatten(np.array(ws)))
                frac_lg_w = np.mean(ws > 0.1)#1.0/len(ws))
                gini_w = gini(ws)
                
                assert len(real_scores) >= N_RUN-1 # 10 repeats of decoding to reduce variance
                score = np.mean(real_scores)
                
                # include real score in null scores
                p_scores = [np.mean(reseidreg.loc[reseidreg['pseudo_id']==pid,score_name]) for pid in pids]#[1:]
                try:
                    assert not np.any(np.isnan(p_scores))
                except:
                    print(f'{eid}, region {reg} has {np.sum(np.isnan(p_scores))} scores which are nan')
                    continue
                median_null = np.median(p_scores)
                pval = np.mean(np.array(p_scores)>=score)
                n_units = np.array(reseidreg.loc[reseidreg['pseudo_id']==-1,'N_units'])
                assert np.all(n_units == n_units[0])
                n_units = n_units[0]
                
                res_table.append([subject,eid,reg,score,pval,median_null,n_units,frac_lg_w,gini_w])
                print('not full pseudo_ids', len(pids))
            else:
                print('not enough pseudo_ids', len(pids))
                
    res_table = pd.DataFrame(res_table, columns=['subject',
                                                 'eid',
                                                 'region',
                                                 'score',
                                                 'p-value',
                                                 'median-null',
                                                 'n_units',
                                                 'frac_large_w',
                                                 'gini_w'])
    return res_table

# DATE = '07-11-2022'
# print('Working on Block')
# file_pre = 'decoding_results/27-10-2022_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
#     res = pd.concat([res, res_new], axis=0)
# res_table = create_pdtable_from_raw(res, 
#                                     score_name='balanced_acc_test',
#                                     N_PSEUDO=200)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_block.csv')

# DATE = '07-11-2022'
# print('Working on Stim')
# file_pre = 'decoding_results/28-10-2022_decode_signcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
#     res = pd.concat([res, res_new], axis=0)
# res_table = create_pdtable_from_raw(res, 
#                                     score_name='R2_test',
#                                     N_PSEUDO=200)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_stim.csv')

# DATE = '07-11-2022'
# print('Working on Choice')
# file_pre = 'decoding_results/28-10-2022_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
#     res = pd.concat([res, res_new], axis=0)
# res_table = create_pdtable_from_raw(res, 
#                                     score_name='balanced_acc_test',
#                                     N_PSEUDO=200)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_choice.csv')

DATE = '07-11-2022'
print('Working on Feedback')
file_pre = 'decoding_results/27-10-2022_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
res = pd.DataFrame()
for i in range(50):
    res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
    res = pd.concat([res, res_new], axis=0)
res_table = create_pdtable_from_raw(res, 
                                    score_name='balanced_acc_test',
                                    N_PSEUDO=200)
valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
res_table = res_table.loc[valid_reg]
res_table.to_csv(f'decoding_processing/{DATE}_reward.csv')

# DATE = '07-11-2022'
# print('Working on Wheel-Speed')
# file_pre = 'decoding_results/27-10-2022_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
#     #res_new['eid'] = res_new['eid']+'_'+res_new['probe']
#     res = pd.concat([res, res_new], axis=0)
# res_table = create_pdtable_from_raw(res, 
#                                     score_name='R2_test',
#                                     N_PSEUDO=100, N_RUN=2)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_wheel-speed.csv')

# DATE = '02-11-2022'
# print('Working on Stim Metaanalysis')
# file_pre = 'decoding_results/02-11-2022_decode_signcont_task_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
# res = pd.DataFrame()
# for i in range(50):
#     res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
#     res = pd.concat([res, res_new], axis=0)
# res_table = create_pdtable_from_raw(res, 
#                                     score_name='balanced_acc_test',
#                                     N_PSEUDO=200)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_stim_metaanalysis_logistic.csv')