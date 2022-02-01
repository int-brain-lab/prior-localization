#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:54:53 2022

@author: bensonb
"""
import os
from plot_decoding_brain import plot_decoding_results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#cmap='Purples'#,'Blues','Greens','Oranges','Reds'

def get_saved_data(results,result_index,
                              RESULTS_PATH,
                              RESULTS_DATE,
                              SAVE_DETAILS,
                              return_number_of_active_neurons=False):
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
    block_pLeft = block_pLeft[datafit_df['mask']]
    
    if return_number_of_active_neurons:
        all_weights = np.concatenate(datafit_df['weights'])
        average_neurons_active = np.mean(all_weights>0.01) * data_df['N_units']
    
        return target, preds, block_pLeft, average_neurons_active
    return target, preds, block_pLeft

RESULTS_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding/results/decoding/'
FIGURE_PATH = '/home/bensonb/IntBrainLab/prior-localization/decoding_figures/'
FIGURE_SUFFIX = '.pdf'

#%%
VARIABLE_FOLDER = 'stimulus/'
if not os.path.isdir(FIGURE_PATH + VARIABLE_FOLDER):
    os.mkdir(FIGURE_PATH + VARIABLE_FOLDER)
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
best_eid = results.loc[results['Rsquared_test'].idxmax(),'eid']
best_probe = results.loc[results['Rsquared_test'].idxmax(),'probe']
best_region = results.loc[results['Rsquared_test'].idxmax(),'region']

plt.title('stimulus: random example')
plt.plot(all_targets[100:400],'.')
plt.plot(all_preds[100:400],'.')
plt.legend(['targets','predictions'])
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsTraceRandomExample', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.title('stimulus: best example')
plt.plot(best_targets[100:400],'.')
plt.plot(best_preds[100:400],'.')
plt.legend(['targets','predictions'])
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2Example', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)

# plotting

plot_decoding_results(regions, reg_values, 
                      VARIABLE_FOLDER + 
                      ('_'.join([RESULTS_DATE, 'brains', SAVE_DETAILS])) +
                      FIGURE_SUFFIX, 
                      FILE_PATH = FIGURE_PATH,
                      cmap='Blues')

sns.set_theme(style="whitegrid")

all_df = pd.DataFrame({'Target':all_targets,
                       'Predictions':all_preds,
                       'pLeft':all_block_pLeft})
best_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds,
                       'pLeft':best_block_pLeft})

ci = 95

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', 
                 data=all_df, 
                 ci=ci, capsize=.2)
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictions_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft', 
                 data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlock_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Target', y='Predictions', 
                 data=best_df, 
                 ci=ci, capsize=.2)
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsBestR2_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft', 
                 data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

# ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
#             data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
#             scale='count')
# ax.set_ylim(-.2,.2)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlock', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions',
#             data=all_df,
#             scale='count')
# ax.set_ylim(-.2,.2)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictions', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()


# ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
#             data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
#             scale='count')
# ax.set_ylim(-1,1)
# plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions',
#             data=best_df,
#             scale='count')
# ax.set_ylim(-1,1)
# plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsBestR2', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()
#%%

VARIABLE_FOLDER = 'choice/'
if not os.path.isdir(FIGURE_PATH + VARIABLE_FOLDER):
    os.mkdir(FIGURE_PATH + VARIABLE_FOLDER)
RESULTS_DATE = '2022-01-29'
SAVE_DETAILS = 'decode_choice_task_Logistic_align_goCue_times_0_pseudosessions_timeWindow_0_0_1_20eidV0'

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
    # if not np.any(target==0):
    all_targets.extend(target)
    all_preds.extend(preds)
    all_block_pLeft.extend(block_pLeft)

best_targets, best_preds, best_block_pLeft = get_saved_data(results,
                                        results['Rsquared_test'].idxmax(),
                                        RESULTS_PATH,RESULTS_DATE,SAVE_DETAILS)
best_eid = results.loc[results['Rsquared_test'].idxmax(),'eid']
best_probe = results.loc[results['Rsquared_test'].idxmax(),'probe']
best_region = results.loc[results['Rsquared_test'].idxmax(),'region']

# all_targets = np.array(all_targets)
# all_preds = np.array(all_preds)
# all_block_pLeft = np.array(all_block_pLeft)
# best_targets = np.array(best_targets)
# best_preds = np.array(best_preds)
# best_block_pLeft = np.array(best_block_pLeft)

plt.title('choice: random example')
plt.plot(all_targets[100:400],'.')
plt.plot(all_preds[100:400],'.')
plt.legend(['targets','predictions'])
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsTraceRandomExample', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.title('choice: best example')
plt.plot(best_targets[100:400],'.')
plt.plot(best_preds[100:400],'.')
plt.legend(['targets','predictions'])
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2Example', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)

# plotting

plot_decoding_results(regions, reg_values, 
                      VARIABLE_FOLDER + 
                      ('_'.join([RESULTS_DATE, 'brains', SAVE_DETAILS])) +
                      FIGURE_SUFFIX, 
                      FILE_PATH = FIGURE_PATH,
                      cmap='Oranges')

# x, y = all_targets, all_preds
# mu0 = np.mean(y[x==0])
# sig0 = np.std(y[x==0])
# mu1 = np.mean(y[x==1])
# sig1 = np.std(y[x==1])
# plt.errorbar([0,1],[mu0,mu1],[sig0,sig1],marker='o',ls='none')
# plt.show()


sns.set_theme(style="whitegrid")

all_df = pd.DataFrame({'Target':all_targets,
                       'Predictions':all_preds,
                       'pLeft':all_block_pLeft})
best_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds,
                       'pLeft':best_block_pLeft})

# tips = sns.load_dataset("tips")
# ax = sns.barplot(x="day", y="total_bill", hue="sex", data=tips)
ci = 95

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', 
                 data=all_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictions_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft', 
                 data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlock_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Target', y='Predictions', 
                 data=best_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsBestR2_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft', 
                 data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Predictions', y='Target', 
                 data=all_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictions_xpred', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


plt.figure(figsize=(3,5))
ax = sns.barplot(x='Predictions', y='Target', hue='pLeft', 
                 data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlock_xpred', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Predictions', y='Target', 
                 data=best_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsBestR2_xpred', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Predictions', y='Target', hue='pLeft', 
                 data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2_xpred', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


# ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
#             data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
#             scale='count')
# ax.set_ylim(-1,1)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlock', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions',
#             data=all_df,
#             scale='count')
# ax.set_ylim(-1,1)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictions', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
#             data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
#             scale='count')
# ax.set_ylim(-1,1)
# plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions',
#             data=best_df,
#             scale='count')
# ax.set_ylim(-1,1)
# plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsBestR2', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

#%%

VARIABLE_FOLDER = 'feedback/'
if not os.path.isdir(FIGURE_PATH + VARIABLE_FOLDER):
    os.mkdir(FIGURE_PATH + VARIABLE_FOLDER)
RESULTS_DATE = '2022-01-29'
SAVE_DETAILS = 'decode_feedback_task_Logistic_align_feedback_times_0_pseudosessions_timeWindow_0_0_2_20eidV0'

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
best_eid = results.loc[results['Rsquared_test'].idxmax(),'eid']
best_probe = results.loc[results['Rsquared_test'].idxmax(),'probe']
best_region = results.loc[results['Rsquared_test'].idxmax(),'region']

plt.title('feedback: random example')
plt.plot(all_targets[100:400],'.')
plt.plot(all_preds[100:400],'.')
plt.legend(['targets','predictions'])
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsTraceRandomExample', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.title('feedback: best example')
plt.plot(best_targets[100:400],'.')
plt.plot(best_preds[100:400],'.')
plt.legend(['targets','predictions'])
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2Example', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
    
# plotting

plot_decoding_results(regions, reg_values, 
                      VARIABLE_FOLDER + 
                      ('_'.join([RESULTS_DATE, 'brains', SAVE_DETAILS])) +
                      FIGURE_SUFFIX, 
                      FILE_PATH = FIGURE_PATH,
                      cmap='Greens')

sns.set_theme(style="whitegrid")

all_df = pd.DataFrame({'Target':all_targets,
                       'Predictions':all_preds,
                       'pLeft':all_block_pLeft})
best_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds,
                       'pLeft':best_block_pLeft})

ci = 95

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', 
                 data=all_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictions_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft', 
                 data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlock_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Target', y='Predictions', 
                 data=best_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsBestR2_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft', 
                 data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Predictions', y='Target', 
                 data=all_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictions_xpred', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


plt.figure(figsize=(3,5))
ax = sns.barplot(x='Predictions', y='Target', hue='pLeft', 
                 data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlock_xpred', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Predictions', y='Target', 
                 data=best_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsBestR2_xpred', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Predictions', y='Target', hue='pLeft', 
                 data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2_xpred', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

# sns.set_theme(style="whitegrid")

# all_df = pd.DataFrame({'Target':all_targets,
#                        'Predictions':all_preds,
#                        'pLeft':all_block_pLeft})
# best_df = pd.DataFrame({'Target':best_targets,
#                        'Predictions':best_preds,
#                        'pLeft':best_block_pLeft})

# ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
#             data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
#             scale='count')
# ax.set_ylim(-2,2)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlock', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions',
#             data=all_df,
#             scale='count')
# ax.set_ylim(-2,2)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictions', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
#             data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
#             scale='count')
# ax.set_ylim(-2,2)
# plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions',
#             data=best_df,
#             scale='count')
# ax.set_ylim(-2,2)
# plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsBestR2', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

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

VARIABLE_FOLDER = 'prior/'
if not os.path.isdir(FIGURE_PATH + VARIABLE_FOLDER):
    os.mkdir(FIGURE_PATH + VARIABLE_FOLDER)
RESULTS_DATE = '2022-01-27'
SAVE_DETAILS = 'decode_prior_expSmoothingPrevActions_Lasso_align_goCue_times_0_pseudosessions_timeWindow_-0_6_-0_2_alleidV0'

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
all_actn = []
for result_index in results.index:
    target, preds, block_pLeft, actn = get_saved_data(results,
                                        result_index,
                                        RESULTS_PATH,
                                        RESULTS_DATE,
                                        SAVE_DETAILS,
                                        return_number_of_active_neurons=True)
    all_targets.extend(target)
    all_preds.extend(preds)
    all_block_pLeft.extend(block_pLeft)
    all_actn.append(actn)

index_max = results.loc[results['region']=='ORBvl','Rsquared_test'].idxmax()
best_targets, best_preds, best_block_pLeft, best_actn = get_saved_data(results,
                                        index_max,
                                        RESULTS_PATH,
                                        RESULTS_DATE,
                                        SAVE_DETAILS,
                                        return_number_of_active_neurons=True)
best_eid = results.loc[index_max,'eid']
best_probe = results.loc[index_max,'probe']
best_region = results.loc[index_max,'region']

plt.title('prior: random example')
plt.plot(all_targets[100:400])
plt.plot(all_preds[100:400])
plt.legend(['targets','predictions'])
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsTraceRandomExample', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.title('prior: best example')
plt.plot(best_targets[100:400])
plt.plot(best_preds[100:400])
plt.legend(['targets','predictions'])
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2Example', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

# plotting

all_targets, best_targets = np.array(all_targets), np.array(best_targets)
best_targets_continuous = np.copy(best_targets)
edge = np.linspace(0,1,11)
for i in range(len(edge)-1):
    all_targets[(all_targets >= edge[i])&(all_targets < edge[i+1])] = .5*(edge[i]+edge[i+1])
    best_targets[(best_targets >= edge[i])&(best_targets < edge[i+1])] = .5*(edge[i]+edge[i+1])




plot_decoding_results(regions, reg_values, 
                      VARIABLE_FOLDER + 
                      ('_'.join([RESULTS_DATE, 'brains', SAVE_DETAILS])) +
                      FIGURE_SUFFIX, 
                      FILE_PATH = FIGURE_PATH,
                      cmap='Purples')

sns.set_theme(style="whitegrid")

all_df = pd.DataFrame({'Target':all_targets,
                       'Predictions':all_preds,
                       'pLeft':all_block_pLeft})
best_df = pd.DataFrame({'Target':best_targets,
                       'Predictions':best_preds,
                       'pLeft':best_block_pLeft})

ci = 95

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', 
                 data=all_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictions_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft', 
                 data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlock_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Target', y='Predictions', 
                 data=best_df, 
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsBestR2_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()


plt.figure(figsize=(3,5))
plt.title(best_eid+' \n['+best_probe+'] ['+best_region+']')
ax = sns.barplot(x='Target', y='Predictions', hue='pLeft', 
                 data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
                 ci=ci, capsize=.2)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2_xtarget', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

# ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
#             data=all_df.loc[(all_df['pLeft']==0.8)|(all_df['pLeft']==0.2),:],
#             scale='count')
# ax.set_ylim(0,1)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlock', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions',
#             data=all_df,
#             scale='count')
# ax.set_ylim(0,1)
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictions', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions', hue='pLeft', split=True,
#             data=best_df.loc[(best_df['pLeft']==0.8)|(best_df['pLeft']==0.2),:],
#             scale='count')
# ax.set_ylim(0,1)
# plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsByBlockBestR2', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

# ax = sns.violinplot(x='Target', y='Predictions',
#             data=best_df,
#             scale='count')
# ax.set_ylim(0,1)
# plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
# plt.tight_layout()
# plt.savefig(FIGURE_PATH +
#             VARIABLE_FOLDER + 
#             ('_'.join([RESULTS_DATE, 'predictionsBestR2', SAVE_DETAILS])) +
#             FIGURE_SUFFIX, 
#             dpi=600)
# plt.show()

plt.figure(figsize=(8,3.5))
plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
plt.plot(best_targets_continuous,'k')
plt.plot(best_preds,'indigo')
plt.legend(['targets','predictions'])
plt.yticks([0,.5,1])
plt.ylim(0,1)
plt.legend(['True','Predicted'],frameon=True,loc=(-0.15,1.1))
plt.xlabel('Trials')
plt.ylabel('Prior')
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'predictionsTraceBestR2', SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

plt.title(best_eid+' ['+best_probe+'] ['+best_region+']')
vals, bins, _ = plt.hist(all_actn, density=True)
ymax, ymin = np.max(vals), np.min(vals)
plt.plot(best_actn*np.ones(51) ,np.linspace(0,ymax*1.1,51))
plt.ylim(0,ymax*1.1)
plt.legend(['Best','All'],frameon=True,loc=(-0.15,1.15))
plt.xlabel('Average Number of Neurons Active')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(FIGURE_PATH +
            VARIABLE_FOLDER + 
            ('_'.join([RESULTS_DATE, 'averageNeuronsActiveDistribution', 
                       SAVE_DETAILS])) +
            FIGURE_SUFFIX, 
            dpi=600)
plt.show()

