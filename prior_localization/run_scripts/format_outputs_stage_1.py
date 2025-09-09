import argparse
import numpy as np
import os
import pickle
import pandas as pd
import glob
import yaml
from tqdm import tqdm
from pathlib import Path

import prior_localization.run_scripts.results


def run(output_dir):
    """
    Process and consolidate stage 1 decoding results from multiple files into a single parquet file.

    This function searches for result files in the specified output directory structure,
    extracts relevant information from each file, and compiles it into a pandas DataFrame.
    The DataFrame is then saved as a parquet file for further analysis.

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
    None
        The function doesn't return any value but saves the consolidated results
        as a parquet file named 'collected_results_stage1.pqt' in the output directory.
        It also prints progress information and statistics about the processing.
    """

    resultsdf = prior_localization.run_scripts.results.consolidate_stage1_pkl2pqt(output_dir)
    fn = str(Path(output_dir).joinpath('collected_results_stage1.pqt'))
    resultsdf.to_parquet(fn)
    print(f"Results saved in {fn}")
    return resultsdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format outputs stage 1')
    parser.add_argument('target')

    args = parser.parse_args()
    target = str(args.target)

    # get output dir from config file
    with open(Path(__file__).parent.parent.joinpath('config.yml'), "r") as config_yml:
        config = yaml.safe_load(config_yml)
    output_dir = Path(config['output_dir']).joinpath(target)

    run(output_dir)