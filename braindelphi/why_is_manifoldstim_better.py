#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 08:59:14 2023

@author: bensonb
"""
import os
import numpy as np
import pandas as pd
from plot_utils import acronym2name, get_xy_vals, get_res_vals, brain_SwansonFlat_results, bar_results
from plot_utils import heatmap, activity_and_decoding_weights
from plot_utils import comb_regs_df, get_within_region_mean_var
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
from one.api import ONE
from brainwidemap.bwm_loading import bwm_units
sns.set(font_scale=1.5)
sns.set_style('ticks')

# load wv, ws, c region sets

# get reference cluster dataframe
julias_clusters = bwm_units(ONE(base_url='https://openalyx.internationalbrainlab.org',
                                password='international'))
julias_clusters['sessreg'] = julias_clusters.apply(lambda x: f"{x['eid']}_{x['Beryl']}", axis=1)
ref_clusters = julias_clusters[['uuids','sessreg']]

out = np.load('michael_stim_restr.npy', allow_pickle=True).flat[0]
rs = list(out.keys())
ns_man = {r:out[r]['nclus'] for r in list(out.keys())}
vs_man = {r:out[r]['amp_euc_can'] for r in list(out.keys())}
ss_man = {r:out[r]['p_euc_can']<0.05 for r in list(out.keys())}


out = pd.read_csv('decoding_processing/01-04-2023_stimside_regs_nsig42_fsig0.341_wi2ovar1.693.csv')

vs_dec = {r:v for v, r in zip(list(out['valuesminusnull_median']),
                     list(out['region'])) if not np.isnan(v)}
ss_dec = {r:p<0.05 for p, r in zip(list(out['combined_p-value']),
                     list(out['region'])) if not np.isnan(p)}
assert set(vs_dec.keys()) == set(vs_man.keys())

plt.figure(figsize=(11,10))
# for r in rs:
#     plt.text(vs_man[r], vs_dec[r], r, fontsize=7)
# plt.plot([vs_man[r] for r in rs], [vs_dec[r] for r in rs], 
#          'o',
#          ms = 1)

plt.scatter([vs_man[r] for r in rs], 
            [vs_dec[r] for r in rs],
            s = [0.5*ns_man[r] for r in rs],
            c = [f'C{ss_man[r] + (2*ss_dec[r])}' for r in rs],
            alpha=.15)
plt.xlabel('Manifold')
plt.ylabel('Decoding')
plt.tight_layout()
plt.savefig('decoding_fig')
plt.show()

