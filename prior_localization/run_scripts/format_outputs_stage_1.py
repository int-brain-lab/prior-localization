import argparse
import numpy as np
import os
import pickle
import pandas as pd
import glob
import yaml
from tqdm import tqdm
from pathlib import Path


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
    finished = glob.glob(str(output_dir.joinpath("*", "*", "*")))

    print("nb files:", len(finished))

    indexers = ["subject", "eid", "probe", "region", "N_units"]

    resultslist = []

    failed_load = 0
    for fn in tqdm(finished):
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