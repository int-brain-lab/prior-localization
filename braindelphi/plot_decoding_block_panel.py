#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:29:32 2022

@author: bensonb
"""
import numpy as np
import pandas as pd
from plot_utils import brain_SwansonFlat_results, bar_results, discretize_target
import matplotlib.pyplot as plt
import seaborn as sns

#%% Block
res_table = pd.read_csv('decoding_processing/20-09-2022_block.csv')

frac_sig_region = lambda reg: np.mean(np.array(res_table.loc[res_table['region']==reg,'p-value']<=0.05))
uni_regs = np.unique(res_table['region'])
uni_regs = uni_regs[(uni_regs!='root')&(uni_regs!='void')]
fs_regs = np.array([frac_sig_region(reg) for reg in uni_regs])
assert not np.any(np.isnan(fs_regs))

brain_SwansonFlat_results(uni_regs, 
                          fs_regs, 
                  filename='block_swanson_fs', 
                  cmap='Purples',
                  clevels=[None, None],
                  ticks=None,
                  extend=None,
                  value_title='Frac. Sig.')

def get_ms_reg(reg):
    c1 = (res_table['region']==reg)
    c2 = (res_table['p-value']<=0.05)
    return np.median(res_table.loc[c1 & c2, 'score'])
# frac_sig_region = lambda reg: np.median(np.array(res_table.loc[res_table['region']==reg,'balanced_acc_test']))
ms_regs = np.array([get_ms_reg(reg) for reg in uni_regs])
r2olivier, v2olivier = uni_regs, ms_regs
brain_SwansonFlat_results(uni_regs[~np.isnan(ms_regs)], 
                          ms_regs[~np.isnan(ms_regs)], 
                  filename='block_swanson_ms', 
                  cmap='Purples',
                  clevels=[None, None],
                  ticks=None,
                  extend=None,
                  value_title='Median Sig. \nScore')

n_reg = lambda reg: len(np.array(res_table.loc[res_table['region']==reg,'p-value']))
n_regs = np.array([n_reg(reg) for reg in uni_regs])
assert not np.any(n_regs==0)
n_regs = np.log(n_regs)/np.log(2)

brain_SwansonFlat_results(uni_regs, 
                          n_regs, 
                  filename='block_swanson_n', 
                  cmap='Purples',
                  clevels=[None, None],
                  ticks=([0,1,2,3,4,5],[1,2,4,8,16,32]),
                  extend=None,
                  value_title='N Sessions')

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
get_nulls = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])

# assert regions have at least 1 sig session TODO bon. corr, 
#        sorted by best median performance (TOPN values plotted), 
#        and greater median performance than the median of the null

regions = np.unique(res_table['region'])
reg_1sigsession = lambda reg: np.any(res_table.loc[res_table['region']==reg,
                                                   'p-value']<=(0.05/len(res_table.loc[res_table['region']==reg,
                                                                                                      'p-value'])))
regions = np.array([reg for reg in regions if reg_1sigsession(reg)])
values = np.array([get_vals(reg) for reg in regions])
nulls = np.array([np.median(get_nulls(reg)) for reg in regions])
acr_plotted = bar_results(regions, 
                            values,
                            nulls,
                            'block_bars', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8],[0.5,0.6,0.7,0.8]),
                            TOP_N=15)
# check criteria.
for reg in acr_plotted:
    print(reg)
    assert np.any(res_table.loc[res_table['region']==reg,
                                'p-value']<=(0.05/len(res_table.loc[res_table['region']==reg,
                                                       'p-value'])))
    assert np.median(get_vals(reg)) > np.median(get_nulls(reg))


#%% plot single session traces

folder = 'decoding_results/20-09-2022_singlesessions/CSHL060_1191f865-b10a-45c8-9c48-24a980fd9402/'
file = '20-09-2022_ORBvl_target_pLeft_timeWindow_-0_4_-0_1_pseudo_id_-1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0.pkl'
ss_res = pd.read_pickle(folder+file)
all_test_predictions = []
all_targets = []
all_masks = []
all_scores = []
for i_run in range(len(ss_res['fit'])):
    # side, stim, act, _ = format_data_mut(result["fit"][i_run]["df"])
    mask = ss_res["fit"][i_run]["mask"]  # np.all(result["fit"][i_run]["target"] == stim[mask])
    # full_test_prediction = np.zeros(np.array(ss_res["fit"][i_run]["target"]).size)
    # for k in range(len(ss_res["fit"][i_run]["idxes_test"])):
    #     full_test_prediction[ss_res["fit"][i_run]['idxes_test'][k]] = ss_res["fit"][i_run]['predictions_test'][k]
    pt = ss_res["fit"][i_run]["predictions_test"]
    full_test_prediction = np.array([p[0] for p in pt])
    all_test_predictions.append(full_test_prediction)
    all_targets.append(ss_res["fit"][i_run]["target"])
    all_masks.append(ss_res["fit"][i_run]["mask"])
    all_scores.append(ss_res["fit"][i_run]["scores_test_full"])
    
probs = np.mean(np.vstack(all_test_predictions),axis=0)
fltg = lambda j: [all_targets[j][i][0] for i in range(len(all_targets[j]))]
targs = np.array([np.array(fltg(j)) for j in range(len(all_targets))])
assert np.all(np.std(np.vstack(targs),axis=0)==0)
assert np.all(np.std(np.vstack(all_masks),axis=0)==0)
targs = targs[0]
mask = np.array(all_masks[0])+0


trials = np.arange(len(mask))[[m==1 for m in mask]]
plt.figure(figsize=(10,2.5))
sessreg_score = np.array(res_table.loc[(res_table['eid']==ss_res['eid'])&
                                       (res_table['region']=='ORBvl'),'score'])
assert len(sessreg_score) == 1
sessreg_score = sessreg_score[0]
plt.title(f'session: {ss_res["eid"]} \n region: ORBvl \n balanced accuracy = {sessreg_score:.3f} (average across 10 models)')
plt.plot(trials, targs, '-', c='k',lw=4)
plt.plot(trials, probs, '-', c='mediumpurple')
cs = (np.array(ss_res["fit"][0]["df"]["choice"])+1)*.5
# plt.plot(np.arange(len(cs)), cs,alpha=.3)
plt.yticks([0,.5,1])
plt.ylim(-0.1,1.1)
plt.xlim(0,len(mask))
plt.legend(['Left Biased Block','Probability of left prediction \n(across 10 models)'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Block')
plt.tight_layout()
plt.savefig('decoding_figures/block_trace', dpi=600)
plt.show()
