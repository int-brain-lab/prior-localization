
import os
import numpy as np
import pandas as pd
from plot_utils import acronym2name, get_xy_vals, get_res_vals, brain_SwansonFlat_results, bar_results
from plot_utils import heatmap, activity_and_decoding_weights
from plot_utils import comb_regs_df, get_within_region_mean_var
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

DATE = '17-12-2022'
VARI = 'blocktofelix'
file_all_results = 'decoding_results/summary/17-12-2022_decode_pLeft_optBay_Lasso_align_stimOn_times_200_pseudosessions_regionWise_timeWindow_-0_6_-0_1_imposterSess_0_balancedWeight_1_RegionLevel_1_mergedProbes_1_behMouseLevelTraining_0_constrainNullSess_0.csv'
FIG_SUF = ''

FOCUS_REGIONS = ['SSp-ul']

res_table = pd.read_csv(file_all_results)
save_comb_regs_data = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=True)
regs_table = comb_regs_df(res_table, USE_ALL_BERYL_REGIONS=False)

n_sig = regs_table['combined_sig'].sum()
f_sig = regs_table['combined_sig'].mean()
wi_means, wi_vars = get_within_region_mean_var(res_table)
wi_var = np.mean(wi_vars)
wo_var = np.var(wi_means)
wi2wo_var = wi_var/wo_var
save_comb_regs_data.to_csv(f'decoding_processing/{DATE}_{VARI}_regs_nsig{n_sig}_fsig{f_sig:.3f}_wi2ovar{wi2wo_var:.3f}.csv')
