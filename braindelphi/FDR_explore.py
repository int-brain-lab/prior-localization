#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:05:28 2023

@author: bensonb
"""
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from one.api import ONE
from brainwidemap.bwm_loading import bwm_units
from plot_utils import comb_regs_df
sns.set(font_scale=1.5)
sns.set_style('ticks')

julias_clusters = bwm_units(ONE(base_url='https://openalyx.internationalbrainlab.org',
                                password='international'))
julias_clusters['sessreg'] = julias_clusters.apply(lambda x: f"{x['eid']}_{x['Beryl']}", axis=1)
ref_clusters = julias_clusters[['uuids','sessreg']]


DATE = '04-05-2023'
VARI = 'block'
preamb = 'decoding_results/summary/'
file_all_results = preamb + '04-05-2023_decode_pLeft_oracle_LogisticsRegression_align_stimOn_times_1000_pseudosessions_regionWise_timeWindow_-0_4_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'

# load results
res_table = pd.read_csv(file_all_results)
res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(ref_clusters['sessreg'])]

regs_table = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=False)

#%%

# Set the parameters
alpha = 0.05
num_samples = 10000  # Number of samples
x = np.linspace(0, 1, 1001)  # x-axis values for plotting

ps = np.array(regs_table['combined_p-value'])
reg_cdf = np.mean(ps <= x[:, None], axis=1)

n = len(ps)

# Generate the samples
samples = np.random.uniform(size=(num_samples, n))

# Calculate the empirical CDF for each sample
ecdf = np.mean(samples <= x[:, None, None], axis=2)

# Calculate the error bars for the 5th and 95th percentiles
lower_bound = np.percentile(ecdf, 2.5, axis=1)
upper_bound = np.percentile(ecdf, 97.5, axis=1)

# Plot the CDF with error bars
plt.figure(figsize=(5,5))
plt.plot(x, np.mean(ecdf,axis=1), c='k', alpha=0.1)
plt.fill_between(x, lower_bound, upper_bound, 
                 label='95% envelope \nof uniform dist.',
                 color = 'k', alpha=0.1)
# plt.plot(x, ecdf[:,0], 'k--', label='sample uniform')
plt.plot(x, reg_cdf, 'C0', lw=2, label='Region decoding')

plt.plot(x, x/alpha, 'r', lw=1, label='FDR-BH (0.05)')
plt.plot(x, 1-((1-x)*(1-alpha)), 'g', lw=1, label='FDR-BB (0.05)')
plt.xlim(-0.05,.15)
plt.ylim(-0.05,.15)
plt.xlabel('p-value')
plt.ylabel('CDF')
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

#%%

