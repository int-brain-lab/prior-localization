#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:29:32 2022

@author: bensonb
"""
import os
import numpy as np
import pandas as pd
from plot_utils import acronym2name, get_xy_vals, get_res_vals, brain_SwansonFlat_results, bar_results
from plot_utils import heatmap, activity_and_decoding_weights
from plot_utils import comb_regs_df, get_within_region_mean_var
from yanliang_brain_slice_plot import get_cmap
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

VARI = 'stim'

# DATE = '29-11-2022'
# file_all_results = 'decoding_results/summary/29-11-2022_decode_signcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
# file_xy_results = 'decoding_results/summary/29-11-2022_decode_signcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'
DATE = '18-01-2023'
file_all_results = 'decoding_results/summary/18-01-2023_decode_signcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = 'decoding_results/summary/18-01-2023_decode_signcont_task_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'

FIG_SUF = '.svg'

FOCUS_REGIONS = ['VISpm', 'PRNr']

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
save_comb_regs_data = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=True)
regs_table = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=False)

n_sig = regs_table['combined_sig'].sum()
f_sig = regs_table['combined_sig'].mean()
wi_means, wi_vars = get_within_region_mean_var(res_table)
wi_var = np.mean(wi_vars)
wo_var = np.var(wi_means)
wi2wo_var = wi_var/wo_var
save_comb_regs_data.to_csv(f'decoding_processing/{DATE}_{VARI}_regs_nsig{n_sig}_fsig{f_sig:.3f}_wi2ovar{wi2wo_var:.3f}.csv')

assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]['weights'].shape[-1] for i in range(xy_table.shape[0])])
cuuids = np.concatenate(list(xy_table['cluster_uuids']))
ws = np.concatenate(list(xy_table['weights']),axis=-1)
ws = ws.reshape((50, -1))
ws_dict = {f'ws_fold{i%5}_runid{i//5}' : ws[i,:] for i in range(50)}
save_cluster_weights = pd.DataFrame({'cluster_uuids': cuuids, 
                                     **ws_dict})
save_cluster_weights.to_csv(f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
#%% Stim

regs = np.array(regs_table['region'])
fs_regs = np.array(regs_table['frac_sig'])
assert not np.any(np.isnan(fs_regs))

brain_SwansonFlat_results(regs, 
                          fs_regs, 
                  filename=f'{VARI}_swanson_fs'+FIG_SUF, 
                  cmap=get_cmap(VARI),
                  clevels=[0, 0.55],
                  ticks=None,
                  extend='max',
                  cbar_orientation='horizontal',
                  value_title='Fraction of significant sessions')

ms_regs = np.array(regs_table['values_median_sig'])

brain_SwansonFlat_results(regs[~np.isnan(ms_regs)], 
                          ms_regs[~np.isnan(ms_regs)], 
                  filename=f'{VARI}_swanson_ms'+FIG_SUF, 
                  cmap=get_cmap(VARI),
                  clevels=[None, None],
                  ticks=None,
                  extend=None,
                  cbar_orientation='horizontal',
                  value_title='Median significant $R^2$')

n_regs = np.array(regs_table['n_sessions'])
assert not np.any(n_regs==0)
n_regs = np.log(n_regs)/np.log(2)

brain_SwansonFlat_results(regs, 
                          n_regs, 
                  filename=f'{VARI}_swanson_n'+FIG_SUF, 
                  cmap=get_cmap(VARI),
                  clevels=[None, None],
                  ticks=([1,2,3,4,5],[2,4,8,16,32]),
                  extend=None,
                  cbar_orientation='vertical',
                  value_title='N Sessions')

# assert regions have a fisher combined p-value<0.05,
#        sorted by best median performance (TOPN values plotted), 
#        and greater median performance than the median of the null

regions = np.array(regs_table.loc[regs_table['combined_sig'],'region'])

get_vals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'score'])
values = np.array([get_vals(reg) for reg in regions])

get_pvals = lambda reg: np.array(res_table.loc[res_table['region']==reg,'p-value'])
values_sig = np.array([(get_pvals(reg)<0.05)+0 for reg in regions])

comb_vals = np.array([np.median(v) for v in values])
comb_nulls = np.array(regs_table.loc[regs_table['combined_sig'],'null_median_of_medians'])
acr_plotted = bar_results(regions, 
                            values,
                            comb_vals,
                            comb_nulls,
                            fillcircle_eids_unordered=values_sig,
                            filename=f'{VARI}_bars'+FIG_SUF, 
                            YMIN=np.min([np.min(v) for v in values]),
                            ylab='$R^2$',
                            TOP_N=15,
                            sort_args=None,
                            bolded_regions=FOCUS_REGIONS)
# check criteria.
for reg in acr_plotted:
    print(reg)
    assert np.median(get_vals(reg)) > np.median(res_table.loc[res_table['region']==reg, 'median-null'])

#%% plot single session traces

clp = lambda x: np.minimum(np.maximum(x,-1),1)
inverse_stim_transf = lambda x : np.round(np.arctanh(clp(x)*np.tanh(5))/5,
                                          decimals=8)

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

# load single trial data
eid = 'dda5fc59-f09a-4256-9fb5-66c67667a466'
region = 'VISpm'
xy_vals = get_xy_vals(xy_table, eid, region)
er_vals = get_res_vals(res_table, eid, region)

l = xy_vals['regressors'].shape[0]
X = np.squeeze(xy_vals['regressors']).T
ws = np.squeeze(xy_vals['weights'])
assert len(ws.shape) == 3
W = np.stack([np.ndarray.flatten(ws[:,:,i]) for i in range(ws.shape[2])]).T
assert W.shape[0] == 50
mask = xy_vals['mask']
preds_multirun = inverse_stim_transf(np.squeeze(xy_vals['predictions']))
preds = np.mean(preds_multirun, axis=0)
targs = inverse_stim_transf(np.squeeze(xy_vals['targets']))
targs_multirun = np.stack((targs for _ in range(10)))
trials = np.arange(len(mask))[[m==1 for m in mask]]

plt.figure(figsize=(14,3.3))
plt.title(f"session: {eid} \n region: {acronym2name(region)} ({region}) \n $R^2$ = {er_vals['score']:.3f} (average across 10 models)")

plt.plot(trials[targs>0], preds[targs>0],'C0o',lw=2,ms=4)
plt.plot(trials[targs<0],preds[targs<0],'C1o',lw=2,ms=4)
# plt.yticks([-1,0,1])
# plt.ylim(-1,1)
plt.xlim(0,len(mask))
plt.legend(['Prediction given stimulus$> 0$', 
            'Prediction given stimulus$< 0$'],
           frameon=True,
           loc=(0.9,1.1))
plt.xlabel('Trials')
plt.ylabel('Average predicted \nsigned stimulus \ncontrast')
plt.tight_layout()
plt.savefig(f'decoding_figures/stim_trace_{region}.svg', dpi=600)
plt.show()

plt.figure(figsize=(8,5))
plt.title(f"session: {eid} \n region: {acronym2name(region)} {(region)} \n $R^2$ = {er_vals['score']:.3f} (average across 10 models)")

ts = targs_multirun.flatten()
ps = preds_multirun.flatten()
best_df = pd.DataFrame({'Target': [str(t) for t in ts[np.argsort(ts)]],
                       'Predicted stimulus contrast': ps[np.argsort(ts)]})
ax = sns.histplot(best_df, 
             x='Target', 
             y='Predicted stimulus contrast',
             bins=[c for c in np.linspace(-1,1,21)],
             cbar=True,
             cbar_kws={'label':'Frequency'},
             cmap = get_cmap(VARI),
             stat='probability')
ax.tick_params(axis='x', rotation=45)

ax.set(xlabel='Signed stimulus contrast')
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(f'decoding_figures/stim_calibrate_{region}.svg', dpi=600)
plt.show()

eid = '16c3667b-e0ea-43fb-9ad4-8dcd1e6c40e1'
region = 'PRNr'
xy_vals = get_xy_vals(xy_table, eid, region)
er_vals = get_res_vals(res_table, eid, region)

l = xy_vals['regressors'].shape[0]
X = np.squeeze(xy_vals['regressors']).T
ws = np.squeeze(xy_vals['weights'])
assert len(ws.shape) == 3
W = np.stack([np.ndarray.flatten(ws[:,:,i]) for i in range(ws.shape[2])]).T
assert W.shape[0] == 50
mask = xy_vals['mask']
preds_multirun = inverse_stim_transf(np.squeeze(xy_vals['predictions']))
preds = np.mean(preds_multirun, axis=0)
targs = inverse_stim_transf(np.squeeze(xy_vals['targets']))
targs_multirun = np.stack((targs for _ in range(10)))
trials = np.arange(len(mask))[[m==1 for m in mask]]

plt.figure(figsize=(14,3.3))
plt.title(f"session: {eid} \n region: {acronym2name(region)} {(region)} \n $R^2$ = {er_vals['score']:.3f} (average across 10 models)")

plt.plot(trials[targs>0], preds[targs>0],'C0o',lw=2,ms=4)
plt.plot(trials[targs<0],preds[targs<0],'C1o',lw=2,ms=4)
# plt.yticks([-1,0,1])
# plt.ylim(-1,1)
plt.xlim(0,len(mask))
plt.legend(['Prediction given stimulus$> 0$', 
            'Prediction given stimulus$< 0$'],
           frameon=True,
           loc=(0.9,1.1))
plt.xlabel('Trials')
plt.ylabel('Average predicted \nsigned stimulus \ncontrast')
plt.tight_layout()
plt.savefig(f'decoding_figures/stim_trace_{region}.svg', dpi=600)
plt.show()

plt.figure(figsize=(8,5))
plt.title(f"session: {eid} \n region: {acronym2name(region)} {(region)} \n $R^2$ = {er_vals['score']:.3f} (average across 10 models)")

ts = targs_multirun.flatten()
ps = preds_multirun.flatten()
best_df = pd.DataFrame({'Target': [str(t) for t in ts[np.argsort(ts)]],
                       'Predicted stimulus contrast': ps[np.argsort(ts)]})
ax = sns.histplot(best_df, 
             x='Target', 
             y='Predicted stimulus contrast',
             bins=[c for c in np.linspace(-1,1,21)],
             cbar=True,
             cbar_kws={'label':'Frequency'},
             cmap = get_cmap(VARI),
             stat='probability')
ax.tick_params(axis='x', rotation=45)

ax.set(xlabel='Signed stimulus contrast')
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(f'decoding_figures/stim_calibrate_{region}.svg', dpi=600)
plt.show()
