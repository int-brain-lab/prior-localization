#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:29:32 2022

@author: bensonb
"""
import numpy as np
import pandas as pd
import scipy.stats
from plot_utils import get_xy_vals, get_res_vals, brain_SwansonFlat_results, bar_results, sess2preds
from plot_utils import heatmap, annotate_heatmap
import matplotlib.pyplot as plt
import seaborn as sns
from ibllib.atlas import AllenAtlas
sns.set_style('whitegrid')

br = AllenAtlas()
all_regs = br.regions.id2acronym(np.load('../../beryl.npy'))

file_all_results = 'decoding_processing/28-11-2022_block.csv'
file_xy_results = 'decoding_processing/28-11-2022_block_xy.pkl'
FIG_SUF = ''

#%% Block

res_table = pd.read_csv(file_all_results)

frac_sig_region = lambda reg: np.mean(np.array(res_table.loc[res_table['region']==reg,'p-value']<0.05))
uni_regs = np.unique(res_table['region'])
uni_regs = uni_regs[(uni_regs!='root')&(uni_regs!='void')]
fs_regs = np.array([frac_sig_region(reg) for reg in uni_regs])
assert not np.any(np.isnan(fs_regs))

brain_SwansonFlat_results(uni_regs, 
                          fs_regs, 
                  filename='block_swanson_fs'+FIG_SUF, 
                  cmap='Purples',
                  clevels=[0, 0.55],
                  ticks=None,
                  extend='max',
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
                  filename='block_swanson_ms'+FIG_SUF, 
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
                  filename='block_swanson_n'+FIG_SUF, 
                  cmap='Purples',
                  clevels=[None, None],
                  ticks=([1,2,3,4,5],[2,4,8,16,32]),
                  extend=None,
                  value_title='N Sessions')

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
get_nulls = lambda reg: np.array(res_table.loc[res_table['region']==reg,'median-null'])
get_nunits = lambda reg: np.array(res_table.loc[res_table['region']==reg,'n_units'])

# assert regions have at least 1 sig session TODO bon. corr, 
#        sorted by best median performance (TOPN values plotted), 
#        and greater median performance than the median of the null

regions = np.unique(res_table['region'])
regions = np.array([reg for reg in regions if not ((reg=='root') or (reg=='void'))])
reg_comb_pval = lambda reg: scipy.stats.combine_pvalues(get_pvals(reg)
                                                        , method='fisher')[1]

save_comb_regs_data = pd.DataFrame({'regions': all_regs, 
              'combined_p-values': [reg_comb_pval(r) if r in regions else np.nan for r in all_regs],
              'combined_sig': [reg_comb_pval(r)<0.05 if r in regions else np.nan for r in all_regs],
              'n_sessions': [len(get_vals(r)) if r in regions else np.nan for r in all_regs],
              'n_units_average': [np.mean(get_nunits(r)) if r in regions else np.nan for r in all_regs],
              'std_vals': [np.std(get_vals(r)) if r in regions else np.nan for r in all_regs],
              'median_vals': [np.median(get_vals(r)) if r in regions else np.nan for r in all_regs],
              'frac_sig': [frac_sig_region(r) if r in regions else np.nan for r in all_regs],
              'median_sig': [get_ms_reg(r) if r in regions else np.nan for r in all_regs]})
n_sig = np.sum([reg_comb_pval(reg)<0.05 for reg in regions])
f_sig = np.mean([reg_comb_pval(reg)<0.05 for reg in regions])
wi_var = np.mean([np.var(get_vals(reg)) for reg in regions])
wo_var = np.var([np.mean(get_vals(reg)) for reg in regions])
wi2wo_var = wi_var/wo_var
save_comb_regs_data.to_csv(file_all_results.split('.')[0]+'_regs_nsig%s_fsig%.3f_wi2ovar%.3f.csv'%(n_sig,f_sig,wi2wo_var))
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
                            filename='block_bars'+FIG_SUF, 
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

#%% plot p-value histogram of individual regions

ir = 'CA1'
pvs_ir = np.array(res_table.loc[res_table['region']==ir,'p-value'])

plt.hist(pvs_ir, bins=20, histtype='step', lw=3)
plt.xlabel(f'p-value ({ir})')
plt.ylabel('Density')
plt.show()

ir = 'SNr'
pvs_ir = np.array(res_table.loc[res_table['region']==ir,'p-value'])

plt.hist(pvs_ir, bins=20, histtype='step', lw=3)
plt.xlabel(f'p-value ({ir})')
plt.ylabel('Density')
plt.show()

#%% plot single session traces

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

# load single trial data
eid = '1191f865-b10a-45c8-9c48-24a980fd9402'
region = 'ORBvl'
xy_vals = get_xy_vals(xy_table, eid, region)
er_vals = get_res_vals(res_table, eid, region)

l = xy_vals['regressors'].shape[0]
X = np.squeeze(xy_vals['regressors']).T
ws = np.squeeze(xy_vals['weights'])
assert len(ws.shape) == 3
W = np.stack([np.ndarray.flatten(ws[:,:,i]) for i in range(ws.shape[2])]).T
assert W.shape[0] == 50
mask = xy_vals['mask']
preds = np.mean(np.squeeze(xy_vals['predictions']), axis=0)
targs = np.squeeze(xy_vals['targets'])
trials = np.arange(len(mask))[[m==1 for m in mask]]

# Wmean = np.abs(np.mean(W,axis=0))
# Walpha = Wmean/np.max(Wmean)
# for i in range(X.shape[0]):
#     plt.plot(X[i,:],alpha=Walpha[i]**2)
# plt.show()

plt.figure(figsize=(10,2.5))

plt.title(f"session: {eid} \n region: {region} \n balanced accuracy = {er_vals['score']:.3f} (average across 10 models)")
plt.plot(trials, targs, '-', c='k',lw=4)
plt.plot(trials, preds, '-', c='mediumpurple')
plt.yticks([0,.5,1])
plt.ylim(-0.1,1.1)
plt.xlim(0,len(mask))
plt.legend(['Left Biased Block',
            'Probability of left prediction \n(across 10 models)'],
           frameon=True,
           loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Block')
plt.tight_layout()
plt.savefig('decoding_figures/block_trace', dpi=600)
plt.show()

#%%

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

regions = np.unique(res_table['region'])
regions = np.array([reg for reg in regions if not ((reg=='root') or (reg=='void'))])
regions = np.array(['ORBvl'])
for my_reg in regions:
    xy_eids = [er.split('_')[0] for er in xy_table['eid_region'] if er.split('_')[1] == my_reg]


    xy_bool = np.array([er.split('_')[1] == my_reg for er in xy_table['eid_region']])
    xy_rgrs = list(xy_table.loc[xy_bool,'regressors'])
    xy_trgs = list(xy_table.loc[xy_bool,'targets'])
    xy_prds = list(xy_table.loc[xy_bool,'predictions'])
    # assert (len(xy_rgrs) == len(xy_trgs)) and (len(xy_rgrs)==len(xy_prds))
    # assert len(xy_rgrs) == len(xy_eids)
    N = np.max([xyr.shape[-1] for xyr in xy_rgrs])
    
    fig, axs = plt.subplots(2*len(xy_rgrs), 
                            figsize=(int(N)+1, int(3*len(xy_eids))))
    # plt.figure(figsize=(int(N/3)+1,32))
    
    for ei in range(len(xy_eids)):
        xyi = ei*2
        my_eid = xy_eids[ei]
        xy_rgr_old, xy_trg_old, xy_prd_old = xy_rgrs[ei], xy_trgs[ei], xy_prds[ei] 
        
        xy_vals = get_xy_vals(xy_table, my_eid, my_reg)
        
        xy_rgr = np.squeeze(xy_vals['regressors']).T
        ws = np.squeeze(xy_vals['weights'])
        assert len(ws.shape) == 3
        xy_w = np.stack([np.ndarray.flatten(ws[:,:,i]) for i in range(ws.shape[2])]).T
        assert xy_w.shape[0] == 50
        xy_prd = np.mean(np.squeeze(xy_vals['predictions']), axis=0)
        xy_trg = np.squeeze(xy_vals['targets'])
        
        '''
        --------------------------------------------------
        first plot: violin plots of activity and predictions
        --------------------------------------------------
        '''
        MAX_SPIKES = np.max(xy_rgr)
        
        x = []
        for i in range(xy_rgr.shape[-1]):
            for t in range(xy_rgr.shape[0]):
                x.append([i, t, xy_rgr[t,0,i], xy_trg[t,0]])
        
        for t in range(xy_rgr.shape[0]):
            x.append([-1, t, MAX_SPIKES*np.mean(xy_prd[:,t]), xy_trg[t,0]])
        
        df = pd.DataFrame(x, columns=['neuron',
                                      'trial',
                                      'spikes',
                                      'target'])
        mr = res_table.loc[(res_table['eid']==my_eid)&(res_table['region']==my_reg)]
        assert len(mr)==1
        mr = mr.iloc[0]
        nt = len(xy_trg)
        nt0 = len(xy_trg[xy_trg==0])
        nt1 = len(xy_trg[xy_trg==1])
        assert nt == (nt0 + nt1)
        axs[xyi].set_title(f'eid:{my_eid}, score:{mr["score"]:.3f}, p:{mr["p-value"]:.3f}, frac_w:{mr["frac_large_w"]:.3f}, gini_w:{mr["gini_w"]:.3f}, n_trials:{nt}, n_trials0:{nt0}, n_trials1:{nt1}',
                           fontsize=10)
        sns.violinplot(ax=axs[xyi], data=df, 
                       x="neuron", y="spikes", 
                       hue="target", split=True,
                       cut=0, linewidth=0)
        lxs = np.linspace(-0.4,0.4)
        lys = MAX_SPIKES*np.ones_like(lxs)
        axs[xyi].plot(lxs,lys,'r',lw=4)
        axs[xyi].plot(lxs,np.zeros_like(lxs),'r',lw=4)
        axs[xyi].text(-0.5,MAX_SPIKES*(1+0.02),'Y*=1')
        axs[xyi].text(-0.5,MAX_SPIKES*(-0.08),'Y*=0')
        tlabels = np.arange(xy_rgr.shape[-1]+1)-1
        newlabels = ['Y*' if l==-1 else str(l) for l in tlabels]
        # ax.set_xticks(tlabels)
        axs[xyi].set_xticklabels(newlabels)
        
        '''
        --------------------------------------------
        second plot: distribution of decoder weights
        --------------------------------------------
        '''
        im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                   cmap="YlGn", cbarlabel="harvest [t/year]")
        texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    
    plt.tight_layout()
    plt.savefig(f'decoding_figures/block_bin_dist/{my_reg}.png',dpi=100)
    print('hi')
    

