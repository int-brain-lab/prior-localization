import os
import numpy as np
import pandas as pd
from plot_utils import acronym2name, get_xy_vals, get_res_vals, brain_SwansonFlat_results, bar_results
from plot_utils import heatmap, activity_and_decoding_weights
from plot_utils import comb_regs_df, get_within_region_mean_var
import matplotlib.pyplot as plt
import seaborn as sns
from one.api import ONE
one = ONE()

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

DATE = '28-11-2022'
VARI = 'wheel-speed'
file_all_results = 'decoding_results/summary/28-11-2022_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = 'decoding_results/summary/28-11-2022_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0_xy.pkl'
FIG_SUF = ''

FOCUS_REGIONS = ['ORBvl']

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

#%%

regs = np.array(regs_table['region'])
fs_regs = np.array(regs_table['frac_sig'])
assert not np.any(np.isnan(fs_regs))

brain_SwansonFlat_results(regs, 
                          fs_regs, 
                  filename=f'{VARI}_swanson_fs'+FIG_SUF, 
                  cmap='Reds',
                  clevels=[0, 0.55],
                  ticks=None,
                  extend='max',
                  cbar_orientation='horizontal',
                  value_title='Fraction of significant sessions')

ms_regs = np.array(regs_table['values_median_sig'])

brain_SwansonFlat_results(regs[~np.isnan(ms_regs)], 
                          ms_regs[~np.isnan(ms_regs)], 
                  filename=f'{VARI}_swanson_ms'+FIG_SUF, 
                  cmap='Reds',
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
                  cmap='Reds',
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

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)

eid = '671c7ea7-6726-4fbe-adeb-f89c2c8e489b'
region = 'GRN'

# load single trial data
xy_vals = get_xy_vals(xy_table, eid, region)
er_vals = get_res_vals(res_table, eid, region)

fmts = one.load_object(eid, 'trials', collection='alf')['firstMovement_times']
mask = xy_vals['mask']
assert len(mask) == len(fmts)
movetimes = fmts[np.array(mask,dtype=bool)]
preds_multirun = np.squeeze(xy_vals['predictions'])
preds_alltrials = np.mean(preds_multirun, axis=0)
targs_alltrials = np.squeeze(xy_vals['targets'])
assert targs_alltrials.shape[0] == len(movetimes)
trials = np.arange(len(mask))[[m==1 for m in mask]]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14,6))
fig.suptitle(f"session: {eid} \n region: {acronym2name(region)} ({region}) \n $R^2$ = {er_vals['score']:.3f} (average across 2 models)")
LW = 3
YMIN, YMAX = -0.9, 8.75

t = 0
targs, preds, movetime, trial = targs_alltrials[t,:], preds_alltrials[t,:], movetimes[t], trials[t]
ax1.set_title(f'Trial {trial}')
ax1.plot((np.arange(len(targs))-10)*0.02 + movetime, targs, 'k', lw=LW)
ax1.plot((np.arange(len(targs))-10)*0.02 + movetime, preds, 'r', lw=LW)
ax1.plot(np.zeros(50)+movetime, np.linspace(YMIN, YMAX), 'k--', lw=LW*0.5)
# ymin, ymax = ax1.get_ylim()
ax1.set_ylim(YMIN,YMAX)
ax1.set_ylabel('Wheel speed (rad./s)')
ax1.set_xticks([movetime, movetime+0.7],
               [f'{movetime:.2f}', f'{movetime+0.7:.2f}'])
ax1.set_xlabel(' ')

t = 173
targs, preds, movetime, trial = targs_alltrials[t,:], preds_alltrials[t,:], movetimes[t], trials[t]
ax2.set_title(f'Trial {trial}')
ax2.plot((np.arange(len(targs))-10)*0.02 + movetime, targs, 'k', lw=LW)
ax2.plot((np.arange(len(targs))-10)*0.02 + movetime, preds, 'r', lw=LW)
ax2.plot(np.zeros(50)+movetime, np.linspace(YMIN, YMAX), 'k--',lw=LW*0.5)
ax2.set_ylim(YMIN,YMAX)
ax2.set_yticklabels([])
ax2.set_xticks([movetime, movetime+0.7],
               [f'{movetime:.2f}', f'{movetime+0.7:.2f}'])

t = 342
targs, preds, movetime, trial = targs_alltrials[t,:], preds_alltrials[t,:], movetimes[t], trials[t]
ax3.set_title(f'Trial {trial}')
ax3.plot((np.arange(len(targs))-10)*0.02 + movetime, targs, 'k', lw=LW)
ax3.plot((np.arange(len(targs))-10)*0.02 + movetime, preds, 'r', lw=LW)
ax3.plot(np.zeros(50)+movetime, np.linspace(YMIN, YMAX), 'k--', lw=LW*0.5)
ax3.set_ylim(YMIN,YMAX)
ax3.set_yticklabels([])
ax3.set_xticks([movetime, movetime+0.7],
               [f'{movetime:.2f}', f'{movetime+0.7:.2f}'])

t = 506
targs, preds, movetime, trial = targs_alltrials[t,:], preds_alltrials[t,:], movetimes[t], trials[t]
ax4.set_title(f'Trial {trial}')
ax4.plot((np.arange(len(targs))-10)*0.02 + movetime, targs, 'k', lw=LW)
ax4.plot((np.arange(len(targs))-10)*0.02 + movetime, preds, 'r', lw=LW)
ax4.plot(np.zeros(50)+movetime, np.linspace(YMIN, YMAX), 'k--', lw=LW*0.5)
ax4.set_ylim(YMIN,YMAX)
ax4.set_yticklabels([])
ax4.set_xticks([movetime, movetime+0.7],
               [f'{movetime:.2f}', f'{movetime+0.7:.2f}'])

fig.text(.5, 0.04, 'Time (s)', ha='center')
fig.legend(['Wheel-speed', 'Prediction average \n (across 2 models)', 'Movement Onset'], loc='upper left')

plt.tight_layout()
plt.savefig('decoding_figures/wheel-speed_trace', dpi=600)
plt.show()
