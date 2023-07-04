#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:34:08 2023

@author: bensonb
"""
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style('ticks')

# ref clusters
preamb = 'decoding_results/summary/'
file_all_results = preamb + '12-05-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_1000_pseudosessions_allProbes_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_0_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1] for i in range(xy_table.shape[0])])


ref_sessions = res_table['eid']
ref_xy = xy_table.copy()
ref_xy['eid'] = ref_xy.apply(lambda x: x['eid_region'].split('_')[0], axis=1)
ref_xy = ref_xy[['eid','cluster_uuids']]
ref_clusters = np.concatenate(list(xy_table['cluster_uuids']))

#%%
VARI = 'block'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '12-05-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_1000_pseudosessions_allProbes_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_0_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1] for i in range(xy_table.shape[0])])

# check clusters
cuuids = np.concatenate(list(xy_table['cluster_uuids']))
assert set(cuuids) == set(ref_clusters)

omni_pvalue = scipy.stats.combine_pvalues(res_table['p-value'], 
                                          method='fisher')[1]

plt.title(f'Fisher combined omni p-value is {omni_pvalue:.2e} \n {(res_table["p-value"]<0.05).sum()}/{(res_table["p-value"]<0.05).count()} sessions significant at alpha=0.05',
          fontsize=14)
plt.hist(res_table['p-value'],
         bins=30,
         histtype='step',
         density=True)
plt.xlabel('P-value')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(f'decoding_figures/SI/pvalues_{VARI}_omnibus.png', 
            dpi=200)
plt.show()


#%%

VARI = 'stimside'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '12-05-2023_decode_signcont_task_LogisticsRegression_align_stimOn_times_1000_pseudosessions_allProbes_timeWindow_0_0_0_1_imposterSess_0_balancedWeight_1_RegionLevel_0_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1] for i in range(xy_table.shape[0])])

# check clusters
cuuids = np.concatenate(list(xy_table['cluster_uuids']))
assert set(cuuids) == set(ref_clusters)

omni_pvalue = scipy.stats.combine_pvalues(res_table['p-value'], 
                                          method='fisher')[1]

plt.title(f'Fisher combined omni p-value is {omni_pvalue:.2e} \n {(res_table["p-value"]<0.05).sum()}/{(res_table["p-value"]<0.05).count()} sessions significant at alpha=0.05',
          fontsize=14)
plt.hist(res_table['p-value'],
         bins=30,
         histtype='step',
         density=True)
plt.xlabel('P-value')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(f'decoding_figures/SI/pvalues_{VARI}_omnibus.png', 
            dpi=200)
plt.show()

#%%

VARI = 'choice'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '12-05-2023_decode_choice_task_LogisticsRegression_align_firstMovement_times_1000_pseudosessions_allProbes_timeWindow_-0_1_0_0_imposterSess_0_balancedWeight_1_RegionLevel_0_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1] for i in range(xy_table.shape[0])])

# check clusters
cuuids = np.concatenate(list(xy_table['cluster_uuids']))
assert set(cuuids) == set(ref_clusters)

omni_pvalue = scipy.stats.combine_pvalues(res_table['p-value'], 
                                          method='fisher')[1]

plt.title(f'Fisher combined omni p-value is {omni_pvalue:.2e} \n {(res_table["p-value"]<0.05).sum()}/{(res_table["p-value"]<0.05).count()} sessions significant at alpha=0.05',
          fontsize=14)
plt.hist(res_table['p-value'],
         bins=30,
         histtype='step',
         density=True)
plt.xlabel('P-value')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(f'decoding_figures/SI/pvalues_{VARI}_omnibus.png', 
            dpi=200)
plt.show()

#%%

VARI = 'feedback'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '12-05-2023_decode_feedback_task_LogisticsRegression_align_feedback_times_1000_pseudosessions_allProbes_timeWindow_0_0_0_2_imposterSess_0_balancedWeight_1_RegionLevel_0_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1] for i in range(xy_table.shape[0])])

# check clusters
cuuids = np.concatenate(list(xy_table['cluster_uuids']))
assert set(cuuids) == set(ref_clusters)

omni_pvalue = scipy.stats.combine_pvalues(res_table['p-value'], 
                                          method='fisher')[1]

plt.title(f'Fisher combined omni p-value is {omni_pvalue:.2e} \n {(res_table["p-value"]<0.05).sum()}/{(res_table["p-value"]<0.05).count()} sessions significant at alpha=0.05',
          fontsize=14)
plt.hist(res_table['p-value'],
         bins=30,
         histtype='step',
         density=True)
plt.xlabel('P-value')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(f'decoding_figures/SI/pvalues_{VARI}_omnibus.png', 
            dpi=200)
plt.show()


#%%

VARI = 'wheel-speed'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '12-05-2023_decode_wheel-speed_task_Lasso_align_firstMovement_times_100_pseudosessions_allProbes_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_0_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
file_xy_results = file_all_results[:-4] + '_xy.pkl'

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([11*len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1] for i in range(xy_table.shape[0])])

# check clusters
cuuids = np.concatenate(list(xy_table['cluster_uuids']))
assert set(cuuids) == set(ref_clusters)

omni_pvalue = scipy.stats.combine_pvalues(res_table['p-value'], 
                                          method='fisher')[1]

plt.title(f'Fisher combined omni p-value is {omni_pvalue:.2e} \n {(res_table["p-value"]<0.05).sum()}/{(res_table["p-value"]<0.05).count()} sessions significant at alpha=0.05',
          fontsize=14)
plt.hist(res_table['p-value'],
         bins=30,
         histtype='step',
         density=True)
plt.xlabel('P-value')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(f'decoding_figures/SI/pvalues_{VARI}_omnibus.png', 
            dpi=200)
plt.show()

#%%

N_pseudo = 100

VARI = 'wheel-vel'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '12-05-2023_decode_wheel-vel_task_Lasso_align_firstMovement_times_100_pseudosessions_allProbes_timeWindow_-0_2_1_0_imposterSess_1_balancedWeight_0_RegionLevel_0_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

file_xy_results = file_all_results[:-4] + '_xy.pkl'

# load results
res_table = pd.read_csv(file_all_results)
xy_table = pd.read_pickle(file_xy_results)
assert np.all([11*len(xy_table.iloc[i]['cluster_uuids']) == xy_table.iloc[i]
              ['weights'].shape[-1] for i in range(xy_table.shape[0])])

# check clusters
trunc_cuuids = np.concatenate(list(ref_xy[ref_xy['eid'].isin(res_table['eid'])]['cluster_uuids']))
cuuids = np.concatenate(list(xy_table['cluster_uuids']))
assert set(cuuids) == set(trunc_cuuids)

mask = ref_sessions.isin(res_table['eid'])
pvals = np.ones(len(ref_sessions))
pvals[mask] = res_table['p-value']
pvals[~mask] = N_pseudo/(N_pseudo+1)

omni_pvalue = scipy.stats.combine_pvalues(pvals, 
                                          method='fisher')[1]

plt.title(f'Fisher combined omni p-value is {omni_pvalue:.2e} \n {(pvals<0.05).sum()}/{len(pvals)} sessions significant at alpha=0.05',
          fontsize=14)
plt.hist(pvals,
         bins=30,
         histtype='step',
         density=True)
plt.xlabel('P-value')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(f'decoding_figures/SI/pvalues_{VARI}_omnibus.png', 
            dpi=200)
plt.show()


