from pathlib import Path
import argparse
import os
import pandas as pd
import yaml

from prior_localization.run_scripts.results import compute_regional_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format outputs stage 3')
    parser.add_argument('target')
    parser.add_argument('--min_units', default=5, required=False)
    parser.add_argument('--min_trials', default=250, required=False)
    parser.add_argument('--min_sessions_per_region', default=2, required=False)
    parser.add_argument('--n_pseudo', default=200, required=False)
    parser.add_argument('--alpha_level', default=0.05, required=False)
    parser.add_argument('--q_level', default=0.01, required=False)

    args = parser.parse_args()
    target = str(args.target)



    # get output dir from config file
    with open(Path(__file__).parent.parent.joinpath('config.yml'), "r") as config_yml:
        config = yaml.safe_load(config_yml)
    output_dir = config['output_dir']


    pqt_file = os.path.join(output_dir, target, "collected_results_stage2.pqt")

    df1 = pd.read_parquet(pqt_file)

    kwargs = dict(min_units=int(args.min_units),
                  min_trials=int(args.min_trials),
                  min_sessions_per_region=int(args.min_sessions_per_region),
                  n_pseudo=int(args.n_pseudo),
                  alpha_level=float(args.alpha_level),
                  q_level=float(args.q_level))
    df2 = compute_regional_stats(df1, **kwargs)

    # save out
    filename = os.path.join(output_dir, target, "collected_results_stage3.pqt")
    print("saving parquet")

    df2.to_parquet(filename)
    print("parquet saved")
