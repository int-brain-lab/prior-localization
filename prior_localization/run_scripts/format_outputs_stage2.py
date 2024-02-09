import argparse
import numpy as np
import os
import pandas as pd


def custom_func(group):
    """Aggregate info over pseudo_ids."""
    result = pd.Series()
    a = group.loc[group['pseudo_id'] == -1, 'score_test'].values
    b = group.loc[group['pseudo_id'] > 0, 'score_test'].values
    assert len(b) == n_pseudo, f'result for this region-eid pair does not contain {n_pseudo} pseudo-sessions'
    result['score'] = a[0]
    result['p-value'] = np.mean(np.array(b) >= a[0])
    result['median-null'] = np.median(b)
    # collect number of trials, only from real session
    c = group.loc[group['pseudo_id'] == -1, 'n_trials'].values
    n_trials = np.unique(c)
    assert len(n_trials) == 1, 'n_trials vals do not agree across all runs'
    result['n_trials'] = int(n_trials[0])
    return result


def reformat_df(df):
    """Compute mean over runs and median of null distribution; compute p-value."""
    
    # set regions to be strings instead of lists of strings of length 1
    df['region'] = df['region'].apply(lambda x: x[0])

    # take mean scores over runs
    df_tmp = df.groupby(
        ['subject', 'eid', 'region', 'N_units', 'pseudo_id', 'n_trials'], as_index=False
    )['score_test'].mean()

    df_new = df_tmp.groupby(['subject', 'eid', 'region', 'N_units']).apply(lambda x: custom_func(x)).reset_index()
    df_new = df_new.rename(columns={"N_units": "n_units"})

    return df_new


parser = argparse.ArgumentParser(description='Format outputs stage 2')
parser.add_argument('output_dir')
parser.add_argument('target')
parser.add_argument('--n_pseudo', default=200, type=int, required=False)

args = parser.parse_args()
output_dir = str(args.output_dir)
target = str(args.target)
n_pseudo = int(args.n_pseudo)

pqt_file = os.path.join(output_dir, target, "collected_results.pqt")

df_collected = reformat_df(pd.read_parquet(pqt_file))

filename = os.path.join(output_dir, target, "collected_results_compressed.pqt")
print("saving parquet")
df_collected.to_parquet(filename)
print("parquet saved")
