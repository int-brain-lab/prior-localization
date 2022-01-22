#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:54:53 2022

@author: bensonb
"""
from plot_decoding_brain import plot_decoding_results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#cmap='Purples'#,'Blues','Greens','Oranges','Reds'

RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/2022-01-21_decode_signcont_task_Lasso_align_goCue_times_0_pseudosessions_timeWindow_0_0_1_v0.parquet'

results = pd.read_parquet(RESULTS_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]
acronyms = np.array(results['region'])
values = np.array(results['Rsquared_test'])

regions = np.unique(acronyms)
reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

plot_decoding_results(regions, reg_values, 'stim_20_nopseudo.png',cmap='Blues')

#%%
RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/2022-01-20_decode_choice_task_Lasso_align_goCue_times_0_pseudosessions_timeWindow_0_0_1_v0.parquet'

results = pd.read_parquet(RESULTS_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]
acronyms = np.array(results['region'])
values = np.array(results['Rsquared_test'])

regions = np.unique(acronyms)
reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

plot_decoding_results(regions, reg_values, 'choice_20_nopseudo.png',cmap='Oranges')

#%%
RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/2022-01-21_decode_feedback_task_Lasso_align_feedback_times_0_pseudosessions_timeWindow_0_0_1_v0.parquet'

results = pd.read_parquet(RESULTS_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]
acronyms = np.array(results['region'])
values = np.array(results['Rsquared_test'])

regions = np.unique(acronyms)
reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

plot_decoding_results(regions, reg_values, 'feedback_20_nopseudo.png',cmap='Greens')

#%%
RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/2022-01-21_decode_prior_expSmoothingPrevActions_Lasso_align_goCue_times_0_pseudosessions_timeWindow_-0_6_-0_2_v0.parquet'

results = pd.read_parquet(RESULTS_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]
acronyms = np.array(results['region'])
values = np.array(results['Rsquared_test'])

regions = np.unique(acronyms)
reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

plot_decoding_results(regions, reg_values, 'prior_10_nopseudo.png',cmap='Purples')

# assuming prior was run last, the data will still be stored there.
mdf = pd.read_pickle('results/decoding/CSHL052/3663d82b-f197-4e8b-b299-7b803a155b84/probe01/2022-01-21_ZI_timeWindow_-0_6_-0_2.pkl')['fit']
preds = np.concatenate(mdf['predictions'])
inds = np.concatenate(mdf['idxes_test'])
preds = preds[np.argsort(inds)]
inds = inds[np.argsort(inds)]
target = mdf['target']

plt.figure(figsize=(6,2.5))
plt.title('3663d82b-f197-4e8b-b299-7b803a155b84 [ZI] \n$r^2=0.19$')
plt.plot(inds,target,'k')
plt.plot(inds,preds,'.',color='indigo')
# plt.ylim(0,1)
plt.yticks([0,.5,1])
plt.legend(['True','Predicted'],frameon=False,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Prior')
plt.tight_layout()
plt.savefig('/home/bensonb/IntBrainLab/prior-localization/decoding_figures/prior_example_ZI_3663d82b-f197-4e8b-b299-7b803a155b84_probe01.png',dpi=600)
plt.show()