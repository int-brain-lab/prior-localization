#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 08:55:00 2023

@author: bensonb
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from one.api import ONE
from ibllib.atlas import BrainRegions

sns.set(font_scale=1.5)
sns.set_style('ticks')
br = BrainRegions()

def acronym2name(acronym):
    return br.name[np.argwhere(br.acronym==acronym)[0]][0]

def get_xy_vals(xy_table, eid, region):
    xy_vals = xy_table.loc[xy_table['eid_region']==f'{eid}_{region}']
    assert xy_vals.shape[0] == 1
    return xy_vals.iloc[0]

def get_res_vals(res_table, eid, region):
    er_vals = res_table[(res_table['eid']==eid) & (res_table['region']==region)]
    assert len(er_vals)==1
    return er_vals.iloc[0]


preamb = 'decoding_results/summary/'
save_dir = 'decoding_figures/'

#%% stimulus

eid = '5d01d14e-aced-4465-8f8e-9a1c674f62ec'
region = 'VISp'

file_all_results = preamb + '02-04-2023_decode_signcont_task_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
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

plt.figure(figsize=(7,4.5))
plt.title(f"session: {eid} \n region: {acronym2name(region)} ({region}) \n balanced accuracy = {er_vals['score']:.3f} (average across 10 models)")
targ_conts, trials_in = np.load(f'stimextrapanel_targsandtrials_eid_{eid}_reg_{region}.npy')
assert np.all(trials == trials_in)
u_conts = np.unique(targ_conts)
neurometric_curve = 1-np.array([np.mean(preds[targ_conts==c]) for c in u_conts])
neurometric_curve_err = np.array([2*np.std(preds[targ_conts==c])/np.sqrt(np.sum(targ_conts==c)) for c in u_conts])
plt.plot(-u_conts, neurometric_curve, lw = 3, c='k')
plt.plot(-u_conts, neurometric_curve, 'ko', ms=8)
plt.errorbar(-u_conts, neurometric_curve, neurometric_curve_err, color='k')
plt.ylim(0,1)
plt.yticks([0, 0.5, 1.0])
plt.xlim(-1.03,1.03)
plt.xticks([-1.    , -0.25  , -0.125 , -0.0625,  0, 0.0625,  0.125 ,  0.25  ,
        1.    ])
plt.xlabel('Contrast (right is >0, left is <0)')
plt.ylabel('Probability of Right Stim side')
# plt.tick_params(axis='both', length=10)
plt.gca().spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(save_dir + 'stimside_neurocurve.svg')
plt.show()

#%% choice

eid = '671c7ea7-6726-4fbe-adeb-f89c2c8e489b'
region = 'GRN'

file_all_results = preamb + '01-04-2023_decode_choice_task_LogisticsRegression_align_firstMovement_times_200_pseudosessions_regionWise_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
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

plt.figure(figsize=(5,4))

plt.title(f"session: {eid} \n region: {acronym2name(region)} ({region}) \n balanced accuracy = {er_vals['score']:.3f} (average across 10 models)")
plt.plot(trials[targs==0],
         1-preds[targs==0],
         'o', c = (255/255, 48/255, 23/255),
         lw=2,ms=4)
plt.plot(trials[targs==1], 
         1-preds[targs==1],
         'o', c = (34/255,77/255,169/255),
         lw=2,ms=4)
plt.yticks([0, 0.5, 1])
# plt.ylim(-1,1)
plt.xlim(100,400)
plt.legend(['Prediction given choice$=$R',
            'Prediction given choice$=$L'],
           frameon=True,
           loc=(0.9,1.1))
plt.xlabel('Trials')
plt.ylabel('Average predicted \nright choice')
plt.gca().spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(save_dir + 'choice_trace.svg')
plt.show()

#%% feedback

eid = 'e012d3e3-fdbc-4661-9ffa-5fa284e4e706'
region = 'IRN'

file_all_results = preamb + '01-04-2023_decode_feedback_task_LogisticsRegression_align_feedback_times_200_pseudosessions_regionWise_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
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

plt.figure(figsize=(5,4))

plt.title(f"session: {eid} \n region: {acronym2name(region)} ({region}) \n balanced accuracy = {er_vals['score']:.3f} (average across 10 models)")
plt.plot(trials[targs==1], preds[targs==1],
         'o', c = (34/255,77/255,169/255), lw=2,ms=4)
plt.plot(trials[targs==0],preds[targs==0],
         'o', c = (255/255, 48/255, 23/255), lw=2,ms=4)
plt.legend(['Prediction given reward$= 1$', 
            'Prediction given reward$= 0$'],
           frameon=True,
           loc=(0.9,1.1))#,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Average predicted \nreward')
plt.xlim(100,400)
plt.yticks([0, 0.5, 1.0])
plt.gca().spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(save_dir + 'feedback_trace.svg')
plt.show()

#%% block

eid = '9e9c6fc0-4769-4d83-9ea4-b59a1230510e'
# eid = 'bd456d8f-d36e-434a-8051-ff3997253802'
# eid = '81a78eac-9d36-4f90-a73a-7eb3ad7f770b'
region = 'MOp'

file_all_results = preamb + '04-05-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_1000_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
xy_vals = get_xy_vals(xy_table, eid, region)
er_vals = get_res_vals(res_table, eid, region)

l = xy_vals['regressors'].shape[0]
X = np.squeeze(xy_vals['regressors']).T
ws = np.squeeze(xy_vals['weights'])
assert len(ws.shape) == 3
W = np.stack([np.ndarray.flatten(ws[:, :, i]) for i in range(ws.shape[2])]).T
assert W.shape[0] == 50
mask = xy_vals['mask']
preds = np.mean(np.squeeze(xy_vals['predictions']), axis=0)
targs = np.squeeze(xy_vals['targets'])
trials = np.arange(len(mask))[[m == 1 for m in mask]]

plt.figure(figsize=(5, 4))
plt.title(
    f"session: {eid} \n region: {acronym2name(region)} ({region}) \n balanced accuracy = {er_vals['score']:.3f} (average across 10 models)")
plt.plot(trials[targs==0],
         1-preds[targs==0],
         'o', c = (255/255, 48/255, 23/255),
         lw=2,ms=4)
plt.plot(trials[targs==1], 
         1-preds[targs==1],
         'o', c = (34/255,77/255,169/255),
         lw=2,ms=4)
plt.legend(['Prediction given choice$=$R',
            'Prediction given choice$=$L'],
           frameon=True,
           loc=(0.9,1.1))
plt.yticks([0, .5, 1])
plt.ylim(-0.1, 1.1)
plt.xlim(500, 800)
plt.xlabel('Trials')
plt.ylabel('Average predicted \nright block')
plt.gca().spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(save_dir + 'block_trace.svg')
plt.show()

#%% wheel
sns.set_style('ticks')


one = ONE()

eid = '671c7ea7-6726-4fbe-adeb-f89c2c8e489b'
region = 'GRN'

# speed

file_all_results = preamb + '01-04-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
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

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,6))
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
ax1.set_ylabel('Actual/predicted wheel speed \n(rad./s)')
ax1.set_xticks([movetime, movetime+0.7],
               [f'{movetime:.2f}', f'{movetime+0.7:.2f}'])
ax1.set_xlabel(' ')
ax1.spines[['top','right']].set_visible(False)

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
ax2.spines[['top','right']].set_visible(False)

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
ax3.spines[['top','right']].set_visible(False)

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
ax4.spines[['top','right']].set_visible(False)

fig.text(.5, 0.04, 'Time (s)', ha='center')
fig.legend(['Actual wheel-speed', 'Predicted wheel-speed \n(average across 2 models)', 'Movement onset'],
           frameon=True,
           loc=(0.75,0.765))

plt.tight_layout()
plt.savefig(save_dir + 'wheel-speed_trace.svg')
plt.show()


# velocity

file_all_results = preamb + '01-04-2023_decode_wheel-vel_task_Lasso_align_firstMovement_times_100_pseudosessions_regionWise_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
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

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,6))
fig.suptitle(f"session: {eid} \n region: {acronym2name(region)} ({region}) \n $R^2$ = {er_vals['score']:.3f} (average across 2 models)")
LW = 3
YMIN, YMAX = -4.1, 4.1

t = 0
targs, preds, movetime, trial = targs_alltrials[t,:], preds_alltrials[t,:], movetimes[t], trials[t]
ax1.set_title(f'Trial {trial}')
ax1.plot((np.arange(len(targs))-10)*0.02 + movetime, targs, 'k', lw=LW)
ax1.plot((np.arange(len(targs))-10)*0.02 + movetime, preds, 'r', lw=LW)
ax1.plot(np.zeros(50)+movetime, np.linspace(YMIN, YMAX), 'k--', lw=LW*0.5)
# ymin, ymax = ax1.get_ylim()
ax1.set_ylim(YMIN,YMAX)
ax1.set_ylabel('Actual/predicted wheel velocity \n(rad./s)')
ax1.set_xticks([movetime, movetime+0.7],
               [f'{movetime:.2f}', f'{movetime+0.7:.2f}'])
ax1.set_xlabel(' ')
ax1.spines[['top','right']].set_visible(False)

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
ax2.spines[['top','right']].set_visible(False)

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
ax3.spines[['top','right']].set_visible(False)

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
ax4.spines[['top','right']].set_visible(False)

fig.text(.5, 0.04, 'Time (s)', ha='center')
fig.legend(['Actual wheel velocity', 'Predicted wheel velocity \n(average across 2 models)', 'Movement onset'],
           frameon=True,
           loc=(0.75,0.765))

plt.tight_layout()
plt.savefig(save_dir + 'wheel-vel_trace.svg')
plt.show()
