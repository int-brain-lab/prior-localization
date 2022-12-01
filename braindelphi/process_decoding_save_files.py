#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 06:14:04 2022

@author: bensonb
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, balanced_accuracy_score

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

def fix_pd_regions(res):
    print('fixing pd regions')
    res = res.reset_index()
    for i, r in enumerate(res.loc[:,'region']):
        if type(r) is list:
            assert len(r) == 1
            res.loc[i,'region'] = r[0]
    return res

def get_val_from_realsession(reseidreg, value_name, RUN_ID=1):
    my_vals = list(reseidreg.loc[(reseidreg['pseudo_id']==-1)&(reseidreg['run_id']==RUN_ID), value_name])
    if (len(my_vals) != 1) or (my_vals[0] is None):
        return None
    return np.array(my_vals[0])

def check_scores(my_preds, my_targets, score_name, real_scores):
    '''
    returns boolean, True if the recored scores match the scores calculated
    using predictions, targets, and score_name

    '''
    my_targets_flat = my_targets[:,0]
    my_preds_flat = [my_preds[pi][:,0] for pi in range(my_preds.shape[0])]
    assert len(my_targets_flat.shape)==1
    assert np.all(np.array([len(p.shape) for p in my_preds_flat])==1)
    
    if score_name == 'balanced_acc_test':
        calc_score = lambda x: balanced_accuracy_score(my_targets_flat, x)
    elif score_name == 'R2_test':
        calc_score = lambda x: r2_score(my_targets_flat, x)
    my_calc_real_scores = [calc_score(p) for p in my_preds_flat]
    isequal_scores = [my_calc_real_scores[i]==real_scores[i] for i in range(len(real_scores))]
    return np.all(np.array(isequal_scores))
        

def create_pdtable_from_raw(res, 
                            score_name='balanced_acc_test',
                            N_PSEUDO=200, N_RUN=10, 
                            N_PSEUDO_LOWER_THRESH = np.infty,
                            RETURN_X_Y=False,
                            SCALAR_PER_TRIAL=True):
    '''
    Takes formatted outputs of decoders and aggregates important values for post-processing
    including subject, eid, region, test statistic, p-value, median of null distribution,
    number of units used for decoding, fraction of weights which exceed a threshold of 0.1, 
    and the gini index of the weights to capture sparsity.
    Optionally returns a secondary table which also includes all the regressors, targets,
    predictions, and weights.

    Parameters
    ----------
    res : pandas DataFrame 
        formatted as done in the output of 04_format_slurm.py.
    score_name : str, optional
        'balanced_acc_test' or 'R2_test'. The default is 'balanced_acc_test'.
    N_PSEUDO : int, optional
        from settings.py. The default is 200.
    N_RUN : int, optional
        from settings.py. The default is 10.
    N_PSEUDO_LOWER_THRESH : scalar, optional
        if finite, allows processing of outputs when not all N_PSEUDO nulls are present. 
        The default is np.infty.
    RETURN_X_Y : bool
        Only works for scalar values per trial, e.g. not wheel-speed. 
        returns an additional table with regressors, targets, and predictions
        The default is False.
    SCALAR_PER_TRIAL : bool
        set to False if decoding multiple values per trial e.g. wheel decoding
        The default is True.

    Returns
    -------
    pandas DataFrame 
        summary of all decoding test statistics, p-values, weight analysis, and units.
    pandas DataFrame
        Optional, only returned if RETURN_X_Y.
        summary of add decoder regressors, targets, and weights

    '''
    if not score_name in ['balanced_acc_test', 'R2_test']:
        raise NotImplementedError('this score is not implemented')
    
    res_table = []
    xy_table = []
    
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
                real_scores = [get_val_from_realsession(reseidreg, 
                                                        score_name, 
                                                        RUN_ID=runid) for runid in range(1,N_RUN+1)]
                #real_scores = reseidreg.loc[reseidreg['pseudo_id']==-1,score_name]
                #assert len(real_scores) == N_RUN
            # elif len(pids) >= N_PSEUDO_LOWER_THRESH+1 and pids[0] == -1:
            #     print('not full pseudo_ids', len(pids))
            #     real_scores = reseidreg.loc[reseidreg['pseudo_id']==-1,score_name]
            #     assert len(real_scores) >= N_RUN - 1
                
            else:
                print('not enough pseudo_ids', len(pids))
                continue
            
            ws = list(reseidreg.loc[reseidreg['pseudo_id']==-1, 'weights'])
            my_weights = np.stack(ws)
            assert len(ws)==N_RUN
            ws = np.abs(np.ndarray.flatten(my_weights))
            #print(ws)
            frac_lg_w = np.mean(ws > 0.1)#1.0/len(ws))
            gini_w = gini(ws)
            # 10 repeats of decoding to reduce variance
            score = np.mean(real_scores)
            
            # include real score in null scores
            n_runs_per_p = [len(np.array(reseidreg.loc[reseidreg['pseudo_id']==pid,score_name])) for pid in pids]#[1:]
            assert np.all(np.array(n_runs_per_p)==N_RUN)
            p_scores = [np.mean(reseidreg.loc[reseidreg['pseudo_id']==pid,score_name]) for pid in pids]#[1:]
            if np.any(np.isnan(p_scores)):
                print(f'{eid}, region {reg} has {np.sum(np.isnan(p_scores))} scores which are nan')
                continue
            
            median_null = np.median(p_scores)
            pval = np.mean(np.array(p_scores)>=score)
            n_units = np.array(reseidreg.loc[reseidreg['pseudo_id']==-1,'N_units'])
            assert np.all(n_units == n_units[0])
            n_units = n_units[0]
            
            if RETURN_X_Y:
                
                my_regressors = get_val_from_realsession(reseidreg, 'regressors')
                if my_regressors is None:
                    print(f'did not find regressors for {eid} {reg}')
                    continue
                
                my_targets = get_val_from_realsession(reseidreg, 'target')
                if my_targets is None:
                    print(f'did not find targets for {eid} {reg}')
                    continue
                
                my_preds = [get_val_from_realsession(reseidreg, 
                                                     'prediction', 
                                                     RUN_ID=runid) for runid in range(1,N_RUN+1)] 
                my_preds = np.stack(my_preds)
                assert my_preds.shape[0] == N_RUN
                
                if np.any(np.array([mps is None for mps in my_preds])):
                    print(f'did not find predictions for {eid} {reg}')
                    continue
                
                # check and reshape arrays
                assert my_targets.shape == my_preds[0].shape
                #assert (my_regressors.shape[1] == 1) and (my_targets.shape[1] == 1) and (my_preds[0].shape[1] == 1)
                assert my_regressors.shape[0] == my_targets.shape[0]
                # my_regressors = my_regressors[:,0,:]
                # my_targets = my_targets[:,0]
                # my_preds = [mps[:,0] for mps in my_preds]
                if np.any(np.array([len(np.unique(my_preds[pi]))==1 for pi in range(my_preds.shape[0])])):
                    print(f'at least one pred is constant {eid} {reg}')
                    continue
                
                if SCALAR_PER_TRIAL:
                    if not check_scores(my_preds, 
                                        my_targets, 
                                        score_name, 
                                        real_scores):
                        raise ValueError('recorded scores do not match calculated scores')
                
                #my_preds = np.mean(my_preds,axis=0)
                #assert np.all(np.unique(my_targets) == np.array([0,1]))
            
            res_table.append([subject,
                              eid,
                              reg,
                              score,
                              pval,
                              median_null,
                              n_units,
                              frac_lg_w,
                              gini_w])
            if RETURN_X_Y:
                xy_table.append([f'{eid}_{reg}', 
                                 my_regressors, 
                                 my_targets, 
                                 my_preds,
                                 my_weights])
                
    res_table = pd.DataFrame(res_table, columns=['subject',
                                                 'eid',
                                                 'region',
                                                 'score',
                                                 'p-value',
                                                 'median-null',
                                                 'n_units',
                                                 'frac_large_w',
                                                 'gini_w'])
    
    if RETURN_X_Y:
        xy_table = pd.DataFrame(xy_table, columns=['eid_region',
                                                   'regressors',
                                                   'targets',
                                                   'predictions',
                                                   'weights'])
        return res_table, xy_table
    
    return res_table


# DATE = '07-11-2022'
# print('Working on Block')
# file_pre = 'decoding_results/27-10-2022_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
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
# res_table.to_csv(f'decoding_processing/{DATE}_block.csv')
# xy_table.to_pickle(f'decoding_processing/{DATE}_block_xy.pkl')

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

DATE = '07-11-2022'
print('Working on Wheel-Speed')
file_pre = 'decoding_results/27-10-2022_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_paraindex'
res = pd.DataFrame()
for i in range(50):
    res_new = pd.read_pickle(file_pre+str(i)+'.pkl')
    #res_new['eid'] = res_new['eid']+'_'+res_new['probe']
    res = pd.concat([res, res_new], axis=0)
res = fix_pd_regions(res)
res_table = create_pdtable_from_raw(res, 
                                    score_name='R2_test',
                                    N_PSEUDO=100, N_RUN=2,
                                    RETURN_X_Y=False,
                                    SCALAR_PER_TRIAL=False)
# valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
# res_table = res_table.loc[valid_reg]
# res_table.to_csv(f'decoding_processing/{DATE}_wheel-speed.csv')
valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=2 for reg in res_table['region']])
res_table = res_table.loc[valid_reg]
#xy_table = xy_table.loc[valid_reg]
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

