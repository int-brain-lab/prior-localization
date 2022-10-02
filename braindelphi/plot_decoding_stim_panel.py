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
res_table = pd.read_csv('decoding_processing/20-09-2022_stim.csv')

frac_sig_region = lambda reg: np.mean(np.array(res_table.loc[res_table['region']==reg,'p-value']<=0.05))
uni_regs = np.unique(res_table['region'])
uni_regs = uni_regs[(uni_regs!='root')&(uni_regs!='void')]
fs_regs = np.array([frac_sig_region(reg) for reg in uni_regs])
assert not np.any(np.isnan(fs_regs))

brain_SwansonFlat_results(uni_regs, 
                          fs_regs, 
                  filename='stim_swanson_fs', 
                  cmap='Blues',
                  clevels=[0, 0.55],
                  ticks=None,
                  extend='max',
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
                  filename='stim_swanson_ms', 
                  cmap='Blues',
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
                  filename='stim_swanson_n', 
                  cmap='Blues',
                  clevels=[None, None],
                  ticks=([0,1,2,3,4,5],[1,2,4,8,16,32]),
                  extend=None,
                  value_title='N Sessions')

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
get_nulls = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])

# assert regions have at least 1 sig session, 
#        sorted by best median performance (TOPN values plotted), 
#        and greater median performance than the median of the null

regions = np.unique(res_table['region'])
reg_1sigsession = lambda reg: np.any(res_table.loc[res_table['region']==reg,
                                                   'p-value']<=0.05)
regions = np.array([reg for reg in regions if reg_1sigsession(reg)])
values = np.array([get_vals(reg) for reg in regions])
nulls = np.array([np.median(get_nulls(reg)) for reg in regions])
acr_plotted = bar_results(regions, 
                            values,
                            nulls,
                            'stim_bars', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='Bal. Acc.', #ticks=([0.5,0.6,0.7,0.8],[0.5,0.6,0.7,0.8]),
                            TOP_N=15)
for reg in acr_plotted:
    print(reg)
    assert np.any(res_table.loc[res_table['region']==reg,'p-value']<=0.05)


#%% plot single session traces

def sess2preds(ss_res, inverse_transf = None):
    if inverse_transf is None:
        inverse_transf = lambda x: x
        
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
        all_test_predictions.append(inverse_transf(full_test_prediction))
        all_targets.append(inverse_transf(ss_res["fit"][i_run]["target"]))
        all_masks.append(ss_res["fit"][i_run]["mask"])
        all_scores.append(ss_res["fit"][i_run]["scores_test_full"])
    
    preds = all_test_predictions[0]
    preds = np.mean(np.vstack(all_test_predictions),axis=0)
    fltg = lambda j: [all_targets[j][i][0] for i in range(len(all_targets[j]))]
    targs = np.array([np.array(fltg(j)) for j in range(len(all_targets))])
    assert np.all(np.abs(np.std(np.vstack(targs),axis=0))<1e-12)
    assert np.all(np.std(np.vstack(all_masks),axis=0)==0)
    targs = targs[0]
    mask = np.array(all_masks[0])+0
    return preds, targs, mask

clp = lambda x: np.minimum(np.maximum(x,-1),1)
inverse_stim_transf = lambda x : np.round(np.arctanh(clp(x)*np.tanh(5))/5,
                                          decimals=8)

folder = 'decoding_results/20-09-2022_singlesessions/CSHL059_dda5fc59-f09a-4256-9fb5-66c67667a466/'
cur_plot_region = 'VISpm'
file = f'20-09-2022_{cur_plot_region}_target_strengthcont_timeWindow_0_0_1_pseudo_id_-1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0.pkl'
ss_res = pd.read_pickle(folder+file)
preds, targs, mask = sess2preds(ss_res, 
                                inverse_transf=inverse_stim_transf)

trials = np.arange(len(mask))[[m==1 for m in mask]]
plt.figure(figsize=(10,2.5))#2.5
sessreg_score = np.array(res_table.loc[(res_table['eid']==ss_res['eid'])&
                                       (res_table['region']==cur_plot_region),
                                       'score'])
assert len(sessreg_score) == 1
sessreg_score = sessreg_score[0]
plt.title(f'session: {ss_res["eid"]} \n region: {cur_plot_region} \n $R^2$ = {sessreg_score:.3f} (average across 10 models)')

plt.plot(trials[targs>0], preds[targs>0],'C0o',lw=2,ms=4)
plt.plot(trials[targs<0],preds[targs<0],'C1o',lw=2,ms=4)
# plt.yticks([-1,0,1])
# plt.ylim(-1,1)
plt.legend(['Prediction given stimulus $> 0$', 
            'Prediction given stimulus $< 0$'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Stimulus')
plt.tight_layout()
plt.savefig(f'decoding_figures/stim_trace_{cur_plot_region}', dpi=600)
plt.show()

best_df = pd.DataFrame({'Target': targs,
                       'Predictions': preds})

plt.figure(figsize=(4.2,5))
plt.title(f'session: {ss_res["eid"]} \n region: {cur_plot_region} \n $R^2$ = {sessreg_score:.3f} (average across 10 models)')
ax = sns.barplot(x='Target', y='Predictions',
                 data=best_df, 
                 ci=95, capsize=.2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set(xlabel='Stimulus')
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(f'decoding_figures/stim_calibrate_{cur_plot_region}', dpi=600)
plt.show()


# folder = 'decoding_results/20-09-2022_singlesessions/KS014_b9c205c3-feac-485b-a89d-afc96d9cb280/'
# cur_plot_region = 'MRN'
folder = 'decoding_results/20-09-2022_singlesessions/KS016_16c3667b-e0ea-43fb-9ad4-8dcd1e6c40e1/'
cur_plot_region = 'PRNr'
file = f'20-09-2022_{cur_plot_region}_target_strengthcont_timeWindow_0_0_1_pseudo_id_-1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0.pkl'
ss_res = pd.read_pickle(folder+file)
preds, targs, mask = sess2preds(ss_res, 
                                inverse_transf=inverse_stim_transf)

trials = np.arange(len(mask))[[m==1 for m in mask]]
plt.figure(figsize=(10,2.5))
sessreg_score = np.array(res_table.loc[(res_table['eid']==ss_res['eid'])&
                                       (res_table['region']==cur_plot_region),
                                       'score'])
assert len(sessreg_score) == 1
sessreg_score = sessreg_score[0]
plt.title(f'session: {ss_res["eid"]} \n region: {cur_plot_region} \n $R^2$ = {sessreg_score:.3f} (average across 10 models)')

plt.plot(trials[targs>0], preds[targs>0],'C0o',lw=2,ms=4)
plt.plot(trials[targs<0],preds[targs<0],'C1o',lw=2,ms=4)
# plt.yticks([-1,0,1])
# plt.ylim(-1,1)
plt.legend(['Prediction given stimulus $> 0$', 
            'Prediction given stimulus $< 0$'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Stimulus')
plt.tight_layout()
plt.savefig(f'decoding_figures/stim_trace_{cur_plot_region}', dpi=600)
plt.show()

best_df = pd.DataFrame({'Target': targs,
                       'Predictions': preds})

plt.figure(figsize=(4.2,5))
plt.title(f'session: {ss_res["eid"]} \n region: {cur_plot_region} \n $R^2$ = {sessreg_score:.3f} (average across 10 models)')
ax = sns.barplot(x='Target', y='Predictions',
                 data=best_df, 
                 ci=95, capsize=.2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set(xlabel='Stimulus')
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(f'decoding_figures/stim_calibrate_{cur_plot_region}', dpi=600)
plt.show()