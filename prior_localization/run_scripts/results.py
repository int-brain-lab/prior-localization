import os
import glob
import pickle

import pandas as pd
import tqdm

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def consolidate_stage1_pkl2pqt(output_dir):
    """
    Process and consolidate stage 1 decoding results from multiple files into a single parquet file.

    This function searches for result files in the specified output directory structure,
    extracts relevant information from each file, and compiles it into a pandas DataFrame.

    Parameters
    ----------
    output_dir : str or Path
        Base directory where output files are stored. This is combined with the target
        parameter to form the complete path.
    target : str
        Target subdirectory within the output_dir where result files are located.
        This typically represents a specific experiment or condition.

    Returns
    -------
    pd.DataFrame
    """
    finished = glob.glob(str(output_dir.joinpath("*", "*", "*.pkl")))

    print("nb files:", len(finished))

    indexers = ["subject", "eid", "probe", "region", "N_units"]

    resultslist = []

    failed_load = 0
    for fn in tqdm.tqdm(finished):
        if os.path.isdir(fn):
            continue
        try:
            fo = open(fn, "rb")
            result = pickle.load(fo)
            fo.close()
            if result["fit"] is None:
                continue
            for i_decoding in range(len(result["fit"])):
                tmpdict = {
                    **{x: result[x] for x in indexers},
                    "fold": -1,
                    "pseudo_id": result["fit"][i_decoding]["pseudo_id"],
                    "run_id": result["fit"][i_decoding]["run_id"] + 1,
                    "score_test": result["fit"][i_decoding]["scores_test_full"],
                    "n_trials": len(result["fit"][i_decoding]["predictions_test"]),
                }
                resultslist.append(tmpdict)
        except Exception as e:
            print(failed_load)
            print(e)
            failed_load += 1
            pass

    print("loading of %i files failed" % failed_load)
    resultsdf = pd.DataFrame(resultslist)
    return resultsdf


def compute_stats_over_pseudo_ids(group, n_pseudo=200):
    """Aggregate info over pseudo_ids."""
    result = pd.Series()
    eid = group["eid"].unique()[0]
    region = group["region"].unique()[0]
    a = group.loc[group['pseudo_id'] == -1, 'score_test'].values
    b = group.loc[group['pseudo_id'] > 0, 'score_test'].values
    if len(b) != n_pseudo:
        print(f'result for {eid}-{region} does not contain {n_pseudo} pseudo-sessions')
    result['n_pseudo'] = len(b)
    if (len(a) == 0) or (len(b) != n_pseudo):
        result['score'] = np.nan
        result['p-value'] = np.nan
        result['median-null'] = np.nan
        result['n_trials'] = np.nan
    else:
        result['score'] = a[0]
        result['p-value'] = np.mean(np.concatenate([np.array(b), [a[0]]]) >= a[0])  # treat real session as pseudo
        result['median-null'] = np.median(b)
        # collect number of trials, only from real session
        c = group.loc[group['pseudo_id'] == -1, 'n_trials'].values
        n_trials = np.unique(c)
        assert len(n_trials) == 1, 'n_trials vals do not agree across all runs'
        result['n_trials'] = int(n_trials[0])
    return result


def aggregate_df_per_session_region(df, n_pseudo=200):
    """Compute mean over runs and median of null distribution; compute p-value."""

    # set regions to be strings instead of lists of strings of length 1
    df['region'] = df['region'].apply(lambda x: x[0])

    # take mean scores over runs
    df_tmp = df.groupby(
        ['subject', 'eid', 'region', 'N_units', 'pseudo_id', 'n_trials'], as_index=False
    )['score_test'].mean()

    df_new = df_tmp.groupby(['subject', 'eid', 'region', 'N_units']).apply(
        lambda x: compute_stats_over_pseudo_ids(x, n_pseudo)
    ).reset_index()
    df_new = df_new.rename(columns={"N_units": "n_units"})

    return df_new


def significance_by_region(group, n_pseudo=200, min_trials=250, min_sessions_per_region=2, alpha_level=.05):
    result = pd.Series()
    # only get p-values for sessions with min number of trials
    if 'n_trials' in group:
        trials_mask = group['n_trials'] >= min_trials
    else:
        trials_mask = np.ones(group.shape[0]).astype('bool')
    pvals = group.loc[trials_mask, 'p-value'].values
    pvals = np.array([p if p > 0 else 1.0 / (n_pseudo + 1) for p in pvals])
    # count number of good sessions
    n_sessions = len(pvals)
    result['n_sessions'] = n_sessions
    # only compute combined p-value if there are enough sessions
    if n_sessions < min_sessions_per_region:
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
        result['frac_sig'] = np.mean(pvals < alpha_level)
        result['values_median_sig'] = np.median(scores[pvals < alpha_level])
        result['sig_combined'] = result['pval_combined'] < alpha_level
    return result


def compute_regional_stats(df1, min_units=5, q_level=.01, **kwargs):
    # compute combined p-values for each region
    df2 = df1[df1.n_units >= min_units].groupby(['region']).apply(
        lambda x: significance_by_region(x, **kwargs)).reset_index()

    # run FDR correction on p-values
    mask = ~df2['pval_combined'].isna()
    _, pvals_combined_corrected, _, _ = multipletests(
        pvals=df2.loc[mask, 'pval_combined'],
        method='fdr_bh',
    )
    df2.loc[mask, 'pval_combined_corrected'] = pvals_combined_corrected
    df2.loc[:, 'sig_combined_corrected'] = df2.pval_combined_corrected < q_level
    return df2
