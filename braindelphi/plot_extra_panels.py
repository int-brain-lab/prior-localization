#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 08:55:00 2023

@author: bensonb
"""
import numpy as np
import pandas as pd
from plot_utils import acronym2name, get_xy_vals, get_res_vals
import matplotlib.pyplot as plt
import seaborn as sns



preamb = 'decoding_results/summary/'
file_all_results = preamb + '02-04-2023_decode_signcont_task_LogisticsRegression_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'
FIG_SUF = '.svg'

res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)


eid = '5d01d14e-aced-4465-8f8e-9a1c674f62ec'
region = 'VISp'
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

sns.set_style('ticks')
plt.figure(figsize=(7,4.5))
plt.title(f"session: {eid} \n region: {acronym2name(region)} ({region}) \n balanced accuracy = {er_vals['score']:.3f} (average across 10 models)")
u_conts = np.unique(targ_conts)
neurometric_curve = 1-np.array([np.mean(preds[targ_conts==c]) for c in u_conts])
neurometric_curve_err = np.array([2*np.std(preds[targ_conts==c]) for c in u_conts])
plt.plot(-u_conts, neurometric_curve, lw = 4, c='k')
plt.ylim(0,1)
plt.xlim(-1,1)
plt.xticks([-1.    , -0.25  , -0.125 , -0.0625,  0, 0.0625,  0.125 ,  0.25  ,
        1.    ])
plt.xlabel('Contrast (right is >0, left is <0)')
plt.ylabel('Probability of Right Stim side')
# plt.tick_params(axis='both', length=10)
plt.tight_layout()
plt.savefig(f'decoding_figures/stimside_neurocurve.svg', dpi=200)
plt.show()