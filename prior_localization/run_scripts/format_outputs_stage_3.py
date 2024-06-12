import argparse
import numpy as np
import os
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


parser = argparse.ArgumentParser(description='Format outputs stage 3')
parser.add_argument('output_dir')
parser.add_argument('target')
parser.add_argument('--min_units', default=5, required=False)
parser.add_argument('--min_trials', default=250, required=False)
parser.add_argument('--min_sessions_per_region', default=2, required=False)
parser.add_argument('--n_pseudo', default=200, required=False)
parser.add_argument('--alpha_level', default=0.05, required=False)
parser.add_argument('--q_level', default=0.01, required=False)

args = parser.parse_args()
output_dir = str(args.output_dir)
target = str(args.target)

MIN_UNITS = args.min_units
MIN_TRIALS = args.min_trials
MIN_SESSIONS_PER_REGION = args.min_sessions_per_region
N_PSEUDO = args.n_pseudo
ALPHA_LEVEL = args.alpha_level
Q_LEVEL = args.q_level


def significance_by_region(group):
    result = pd.Series()
    # only get p-values for sessions with min number of trials
    if 'n_trials' in group:
        trials_mask = group['n_trials'] >= MIN_TRIALS
    else:
        trials_mask = np.ones(group.shape[0]).astype('bool')
    pvals = group.loc[trials_mask, 'p-value'].values
    pvals = np.array([p if p > 0 else 1.0 / (N_PSEUDO + 1) for p in pvals])
    # count number of good sessions
    n_sessions = len(pvals)
    result['n_sessions'] = n_sessions
    # only compute combined p-value if there are enough sessions
    if n_sessions < MIN_SESSIONS_PER_REGION:
        result['pval_combined'] = np.nan
        result['n_units_mean'] = np.nan
        result['values_std'] = np.nan
        result['values_median'] = np.nan
        result['null_median_of_medians'] = np.nan
        result['valuesminusnull_median'] = np.nan
        result['frac_sig'] = np.nan
        result['values_median_sig'] = np.nan
        result['sig_combined'] = np.nan
    else:
        scores = group.loc[trials_mask, 'score'].values
        result['pval_combined'] = stats.combine_pvalues(pvals, method='fisher')[1]
        result['n_units_mean'] = group.loc[trials_mask, 'n_units'].mean()
        result['values_std'] = np.std(scores)
        result['values_median'] = np.median(scores)
        result['null_median_of_medians'] = group.loc[trials_mask, 'median-null'].median()
        result['valuesminusnull_median'] = result['values_median'] - result['null_median_of_medians']
        result['frac_sig'] = np.mean(pvals < ALPHA_LEVEL)
        result['values_median_sig'] = np.median(scores[pvals < ALPHA_LEVEL])
        result['sig_combined'] = result['pval_combined'] < ALPHA_LEVEL
    return result


pqt_file = os.path.join(output_dir, target, "collected_results_stage2.pqt")

df1 = pd.read_parquet(pqt_file)

# compute combined p-values for each region
df2 = df1[df1.n_units >= MIN_UNITS].groupby(['region']).apply(lambda x: significance_by_region(x)).reset_index()

# run FDR correction on p-values
mask = ~df2['pval_combined'].isna()
_, pvals_combined_corrected, _, _ = multipletests(
    pvals=df2.loc[mask, 'pval_combined'],
    alpha=Q_LEVEL,
    method='fdr_bh',
)
df2.loc[mask, 'pval_combined_corrected'] = pvals_combined_corrected
df2.loc[:, 'sig_combined_corrected'] = df2.pval_combined_corrected < Q_LEVEL

# save out
filename = os.path.join(output_dir, target, "collected_results_stage3.pqt")
print("saving parquet")
df2.to_parquet(filename)
print("parquet saved")
