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
from one.api import ONE
from brainwidemap.bwm_loading import bwm_units
sns.set(font_scale=1.5)
sns.set_style('ticks')

# get reference cluster dataframe
julias_clusters = bwm_units(ONE(base_url='https://openalyx.internationalbrainlab.org',
                                password='international'))
julias_clusters['sessreg'] = julias_clusters.apply(lambda x: f"{x['eid']}_{x['Beryl']}", axis=1)
ref_clusters = julias_clusters[['uuids','sessreg']]

DATE = '12-05-2023'
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
assert set(cuuids) == set(ref_clusters['uuids'])

omni_pvalue = scipy.stats.combine_pvalues(res_table['p-value'], 
                                          method='fisher')[1]

#%%

plt.title(f'Fisher combined omni p-value is {omni_pvalue:.2e} \n {(res_table["p-value"]<0.05).sum()}/{(res_table["p-value"]<0.05).count()} sessions significant at alpha=0.05')
plt.hist(res_table['p-value'],
         bins=30,
         histtype='step',
         density=True)
plt.xlabel('P-value')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('decoding_figures/SI/pvalues_block_omnibus.png', 
            dpi=200)
plt.show()
