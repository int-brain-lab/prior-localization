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
import seaborn as sns
#cmap='Purples'#,'Blues','Greens','Oranges','Reds'

def get_saved_data(results,result_index,
                              RESULTS_PATH,RESULTS_DATE,SAVE_DETAILS):
    subject = results.loc[result_index,'subject']
    eid = results.loc[result_index,'eid']
    probe = results.loc[result_index,'probe']
    region = results.loc[result_index,'region']
    data_df = pd.read_pickle(RESULTS_PATH + 
                             subject + '/' + 
                             eid + '/' + 
                             probe + '/' + 
                             ('_'.join([RESULTS_DATE, region, SAVE_DETAILS])) +
                             '.pkl')
    datafit_df = data_df['fit']
    preds = np.concatenate(datafit_df['predictions_test'])
    inds = np.concatenate(datafit_df['idxes_test'])
    preds = preds[np.argsort(inds)]
    inds = inds[np.argsort(inds)]
    assert len(np.unique(inds))==len(inds)
    assert np.max(inds)==len(inds)-1
    target = datafit_df['target']
    block_pLeft = datafit_df['pLeft_vec']
    
    return target, preds, block_pLeft[datafit_df['mask']]

RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/'
FIGURE_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'

#%%

RESULTS_DATE = '2022-01-25'
SAVE_DETAILS = 'decode_signcont_task_Lasso_align_goCue_times_0_pseudosessions_timeWindow_0_0_1_20eidV0'

FILE_PATH = RESULTS_PATH + ('_'.join([RESULTS_DATE, SAVE_DETAILS])) + '.parquet'
results = pd.read_parquet(FILE_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]
acronyms = np.array(results['region'])
values = np.array(results['Rsquared_test'])
regions = np.unique(acronyms)
reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

all_targets = []
all_preds = []
all_block_pLeft = []
for result_index in results.index:
    target, preds, block_pLeft = get_saved_data(results,
                                        result_index,
                                        RESULTS_PATH,RESULTS_DATE,SAVE_DETAILS)
    all_targets.extend(target)
    all_preds.extend(preds)
    all_block_pLeft.extend(block_pLeft)

best_targets, best_preds, best_block_pLeft = get_saved_data(results,
                                        results['Rsquared_test'].idxmax(),
                                        RESULTS_PATH,RESULTS_DATE,SAVE_DETAILS)
    
# plotting

plot_decoding_results(regions, reg_values, 
                      ('_'.join([RESULTS_DATE, 'brains', SAVE_DETAILS]))+'.png',
                      FILE_PATH = FIGURE_PATH,
                      cmap='Blues')

sns.set_theme(style="whitegrid")

all_df = pd.DataFrame({'Target':all_targets,
                       'Predictions':all_preds,
                       'pLeft':all_block_pLeft})
best_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds,
                       'pLeft':best_block_pLeft})

ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
            data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
            scale='count')
ax.set_ylim(-.5,.5)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsByBlock', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions',
            data=all_df,
            scale='count')
ax.set_ylim(-.5,.5)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictions', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
            data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
            scale='count')
ax.set_ylim(-1,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions',
            data=best_df,
            scale='count')
ax.set_ylim(-1,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsBestR2', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()
#%%
# RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/2022-01-20_decode_choice_task_Lasso_align_goCue_times_0_pseudosessions_timeWindow_0_0_1_v0.parquet'

# results = pd.read_parquet(RESULTS_PATH).reset_index()
# results = results.loc[results.loc[:,'fold']==-1]
# acronyms = np.array(results['region'])
# values = np.array(results['Rsquared_test'])

# regions = np.unique(acronyms)
# reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

# plot_decoding_results(regions, reg_values, 'choice_20_nopseudo.png',cmap='Oranges')

RESULTS_DATE = '2022-01-25'
SAVE_DETAILS = 'decode_choice_task_Lasso_align_goCue_times_0_pseudosessions_timeWindow_0_0_1_20eidV0'

FILE_PATH = RESULTS_PATH + ('_'.join([RESULTS_DATE, SAVE_DETAILS])) + '.parquet'
results = pd.read_parquet(FILE_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]
acronyms = np.array(results['region'])
values = np.array(results['Rsquared_test'])
regions = np.unique(acronyms)
reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

all_targets = []
all_preds = []
all_block_pLeft = []
for result_index in results.index:
    target, preds, block_pLeft = get_saved_data(results,
                                        result_index,
                                        RESULTS_PATH,RESULTS_DATE,SAVE_DETAILS)
    if not np.any(target==0):
        all_targets.extend(target)
        all_preds.extend(preds)
        all_block_pLeft.extend(block_pLeft)

best_targets, best_preds, best_block_pLeft = get_saved_data(results,
                                        results['Rsquared_test'].idxmax(),
                                        RESULTS_PATH,RESULTS_DATE,SAVE_DETAILS)
    
# plotting

plot_decoding_results(regions, reg_values, 
                      ('_'.join([RESULTS_DATE, 'brains', SAVE_DETAILS]))+'.png',
                      FILE_PATH = FIGURE_PATH,
                      cmap='Oranges')

sns.set_theme(style="whitegrid")

all_df = pd.DataFrame({'Target':all_targets,
                       'Predictions':all_preds,
                       'pLeft':all_block_pLeft})
best_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds,
                       'pLeft':best_block_pLeft})

ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
            data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
            scale='count')
ax.set_ylim(-.5,.5)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsByBlock', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions',
            data=all_df,
            scale='count')
ax.set_ylim(-.5,.5)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictions', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
            data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
            scale='count')
ax.set_ylim(-1,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions',
            data=best_df,
            scale='count')
ax.set_ylim(-1,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsBestR2', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

#%%
# RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/2022-01-21_decode_feedback_task_Lasso_align_feedback_times_0_pseudosessions_timeWindow_0_0_1_v0.parquet'

# results = pd.read_parquet(RESULTS_PATH).reset_index()
# results = results.loc[results.loc[:,'fold']==-1]
# acronyms = np.array(results['region'])
# values = np.array(results['Rsquared_test'])

# regions = np.unique(acronyms)
# reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

# plot_decoding_results(regions, reg_values, 'feedback_20_nopseudo.png',cmap='Greens')

RESULTS_DATE = '2022-01-25'
SAVE_DETAILS = 'decode_feedback_task_Lasso_align_feedback_times_0_pseudosessions_timeWindow_0_0_1_20eidV0'

FILE_PATH = RESULTS_PATH + ('_'.join([RESULTS_DATE, SAVE_DETAILS])) + '.parquet'
results = pd.read_parquet(FILE_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]
acronyms = np.array(results['region'])
values = np.array(results['Rsquared_test'])
regions = np.unique(acronyms)
reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

all_targets = []
all_preds = []
all_block_pLeft = []
for result_index in results.index:
    target, preds, block_pLeft = get_saved_data(results,
                                        result_index,
                                        RESULTS_PATH,RESULTS_DATE,SAVE_DETAILS)
    all_targets.extend(target)
    all_preds.extend(preds)
    all_block_pLeft.extend(block_pLeft)

best_targets, best_preds, best_block_pLeft = get_saved_data(results,
                                        results['Rsquared_test'].idxmax(),
                                        RESULTS_PATH,RESULTS_DATE,SAVE_DETAILS)
    
# plotting

plot_decoding_results(regions, reg_values, 
                      ('_'.join([RESULTS_DATE, 'brains', SAVE_DETAILS]))+'.png',
                      FILE_PATH = FIGURE_PATH,
                      cmap='Blues')

sns.set_theme(style="whitegrid")

all_df = pd.DataFrame({'Target':all_targets,
                       'Predictions':all_preds,
                       'pLeft':all_block_pLeft})
best_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds,
                       'pLeft':best_block_pLeft})

ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
            data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
            scale='count')
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsByBlock', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions',
            data=all_df,
            scale='count')
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictions', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
            data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
            scale='count')
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions',
            data=best_df,
            scale='count')
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsBestR2', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

#%%
# RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/2022-01-21_decode_prior_expSmoothingPrevActions_Lasso_align_goCue_times_0_pseudosessions_timeWindow_-0_6_-0_2_v0.parquet'

# results = pd.read_parquet(RESULTS_PATH).reset_index()
# results = results.loc[results.loc[:,'fold']==-1]
# acronyms = np.array(results['region'])
# values = np.array(results['Rsquared_test'])

# regions = np.unique(acronyms)
# reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

# plot_decoding_results(regions, reg_values, 'prior_10_nopseudo.png',cmap='Purples')

# # assuming prior was run last, the data will still be stored there.
# mdf = pd.read_pickle('results/decoding/CSHL052/3663d82b-f197-4e8b-b299-7b803a155b84/probe01/2022-01-21_ZI_timeWindow_-0_6_-0_2.pkl')['fit']
# preds = np.concatenate(mdf['predictions'])
# inds = np.concatenate(mdf['idxes_test'])
# preds = preds[np.argsort(inds)]
# inds = inds[np.argsort(inds)]
# target = mdf['target']

# plt.figure(figsize=(6,2.5))
# plt.title('3663d82b-f197-4e8b-b299-7b803a155b84 [ZI] \n$r^2=0.19$')
# plt.plot(inds,target,'k')
# plt.plot(inds,preds,'.',color='indigo')
# # plt.ylim(0,1)
# plt.yticks([0,.5,1])
# plt.legend(['True','Predicted'],frameon=False,loc=(-0.15,1.1))
# plt.xlabel('Trials')
# plt.ylabel('Prior')
# plt.tight_layout()
# plt.savefig('/home/bensonb/IntBrainLab/prior-localization/decoding_figures/prior_example_ZI_3663d82b-f197-4e8b-b299-7b803a155b84_probe01.png',dpi=600)
# plt.show()

RESULTS_DATE = '2022-01-25'
SAVE_DETAILS = 'decode_prior_expSmoothingPrevActions_Lasso_align_goCue_times_0_pseudosessions_timeWindow_-0_6_-0_2_20eidV0'

FILE_PATH = RESULTS_PATH + ('_'.join([RESULTS_DATE, SAVE_DETAILS])) + '.parquet'
results = pd.read_parquet(FILE_PATH).reset_index()
results = results.loc[results.loc[:,'fold']==-1]
acronyms = np.array(results['region'])
values = np.array(results['Rsquared_test'])
regions = np.unique(acronyms)
reg_values = np.array([np.median(values[reg==acronyms]) for reg in regions])

all_targets = []
all_preds = []
all_block_pLeft = []
for result_index in results.index:
    target, preds, block_pLeft = get_saved_data(results,
                                        result_index,
                                        RESULTS_PATH,RESULTS_DATE,SAVE_DETAILS)
    all_targets.extend(target)
    all_preds.extend(preds)
    all_block_pLeft.extend(block_pLeft)

best_targets, best_preds, best_block_pLeft = get_saved_data(results,
                                        results['Rsquared_test'].idxmax(),
                                        RESULTS_PATH,RESULTS_DATE,SAVE_DETAILS)
all_targets, best_targets = np.array(all_targets), np.array(best_targets)
all_targets[all_targets >= 0.5] = 1
all_targets[all_targets < 0.5] = 0
best_targets[best_targets >= 0.5] = 1
best_targets[best_targets < 0.5] = 0
# plotting

plot_decoding_results(regions, reg_values, 
                      ('_'.join([RESULTS_DATE, 'brains', SAVE_DETAILS]))+'.png',
                      FILE_PATH = FIGURE_PATH,
                      cmap='Purples')

sns.set_theme(style="whitegrid")

all_df = pd.DataFrame({'Target':all_targets,
                       'Predictions':all_preds,
                       'pLeft':all_block_pLeft})
best_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds,
                       'pLeft':best_block_pLeft})

ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
            data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
            scale='count')
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsByBlock', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions',
            data=all_df,
            scale='count')
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictions', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
            data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
            scale='count')
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()

ax = sns.violinplot(x='Target', y='Predictions',
            data=best_df,
            scale='count')
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            ('_'.join([RESULTS_DATE, 'predictionsBestR2', SAVE_DETAILS]))+'.png', 
            dpi=600)
plt.show()
