import argparse
import os
from pathlib import Path

import pandas as pd
import yaml

from prior_localization.run_scripts.results import reformat_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format outputs stage 2')
    parser.add_argument('target')
    parser.add_argument('--n_pseudo', default=200, type=int, required=False)

    args = parser.parse_args()
    target = str(args.target)
    n_pseudo = int(args.n_pseudo)

    # get output dir from config file
    with open(Path(__file__).parent.parent.joinpath('config.yml'), "r") as config_yml:
        config = yaml.safe_load(config_yml)
    output_dir = config['output_dir']

    pqt_file = os.path.join(output_dir, target, "collected_results_stage1.pqt")

    df_collected = reformat_df(pd.read_parquet(pqt_file))

    filename = os.path.join(output_dir, target, "collected_results_stage2.pqt")
    print("saving parquet")
    df_collected.to_parquet(filename)
    print("parquet saved")
