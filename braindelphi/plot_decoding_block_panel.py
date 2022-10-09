#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:29:32 2022

@author: bensonb
"""
import numpy as np
import pandas as pd
import scipy.stats
from plot_utils import brain_SwansonFlat_results, bar_results, sess2preds
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#%% Block
file_all_results = 'decoding_processing/20-09-2022_block.csv'
res_table = pd.read_csv(file_all_results)

frac_sig_region = lambda reg: np.mean(np.array(res_table.loc[res_table['region']==reg,'p-value']<0.05))
uni_regs = np.unique(res_table['region'])
uni_regs = uni_regs[(uni_regs!='root')&(uni_regs!='void')]
fs_regs = np.array([frac_sig_region(reg) for reg in uni_regs])
assert not np.any(np.isnan(fs_regs))

brain_SwansonFlat_results(uni_regs, 
                          fs_regs, 
                  filename='block_swanson_fs', 
                  cmap='Purples',
                  clevels=[0, 0.55],
                  ticks=None,
                  extend=None,
                  value_title='Frac. Sig.')

def get_ms_reg(reg):
    c1 = (res_table['region']==reg)
    c2 = (res_table['p-value']<0.05)
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
get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
get_nulls = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])

# assert regions have at least 1 sig session TODO bon. corr, 
#        sorted by best median performance (TOPN values plotted), 
#        and greater median performance than the median of the null

regions = np.unique(res_table['region'])
regions = np.array([reg for reg in regions if not ((reg=='root') or (reg=='void'))])
reg_comb_pval = lambda reg: scipy.stats.combine_pvalues(get_pvals(reg)
                                                        , method='fisher')[1]

save_comb_regs_data = pd.DataFrame({'regions': regions, 
              'combined_p-values': [reg_comb_pval(reg) for reg in regions],
              'combined_sig': [reg_comb_pval(reg)<0.05 for reg in regions],
              'n_sessions': [len(get_vals(reg)) for reg in regions],
              'frac_sig': [frac_sig_region(reg) for reg in regions],
              'median_sig': [get_ms_reg(reg) for reg in regions]})
n_sig = np.sum([reg_comb_pval(reg)<0.05 for reg in regions])
f_sig = np.mean([reg_comb_pval(reg)<0.05 for reg in regions])
save_comb_regs_data.to_csv(file_all_results.split('.')[0]+'_regs_nsig%s_fsig%.3f.csv'%(n_sig,f_sig))
# reg_1sigsession = lambda reg: np.any(res_table.loc[res_table['region']==reg,
#                                                    'p-value']<=(0.05/len(res_table.loc[res_table['region']==reg,
#                                                                                                       'p-value'])))
# regions = np.array([reg for reg in regions if reg_1sigsession(reg)])
regions = np.array([reg for reg in regions if reg_comb_pval(reg)<0.05])
print('regions 1sig', regions, np.unique(res_table['region']))
print('frac regions', (len(regions)-1)/(len(np.unique(res_table['region']))-2))
values = np.array([get_vals(reg) for reg in regions])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])
comb_pvalues = np.array([reg_comb_pval(reg) for reg in regions])
comb_nulls = np.array([np.median(get_nulls(reg)) for reg in regions])
acr_plotted = bar_results(regions, 
                            values,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename='block_bars', 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='Bal. Acc.',
                            ticks=([0.5,0.6,0.7,0.8],[0.5,0.6,0.7,0.8]),
                            TOP_N=15,
                            sort_args=None)
# check criteria.
for reg in acr_plotted:
    print(reg)
    # assert np.any(res_table.loc[res_table['region']==reg,
    #                             'p-value']<=(0.05/len(res_table.loc[res_table['region']==reg,
    #                                                    'p-value'])))
    assert np.median(get_vals(reg)) > np.median(get_nulls(reg))


#%% plot single session traces


folder = 'decoding_results/20-09-2022_singlesessions/CSHL060_1191f865-b10a-45c8-9c48-24a980fd9402/'
cur_plot_region = 'ORBvl'
file = f'20-09-2022_{cur_plot_region}_target_pLeft_timeWindow_-0_4_-0_1_pseudo_id_-1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_simulated_0_constrainNullSess_0.pkl'
ss_res = pd.read_pickle(folder+file)

preds, targs, mask = sess2preds(ss_res, 
                                inverse_transf=None)

trials = np.arange(len(mask))[[m==1 for m in mask]]
plt.figure(figsize=(10,2.5))
sessreg_score = np.array(res_table.loc[(res_table['eid']==ss_res['eid'])&
                                       (res_table['region']==cur_plot_region),
                                       'score'])
assert len(sessreg_score) == 1
sessreg_score = sessreg_score[0]
plt.title(f'session: {ss_res["eid"]} \n region: {cur_plot_region} \n balanced accuracy = {sessreg_score:.3f} (average across 10 models)')
plt.plot(trials, targs, '-', c='k',lw=4)
plt.plot(trials, preds, '-', c='mediumpurple')
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
