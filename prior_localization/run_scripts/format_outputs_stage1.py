import argparse
import numpy as np
import os
import pickle
import pandas as pd
import glob
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser(description='Format outputs')
parser.add_argument('output_dir')
parser.add_argument('target')

def format_outputs_stage1(output_dir, target, file_output_report=None):
    output_dir = Path(output_dir).joinpath(target)
    finished = glob.glob(str(output_dir.joinpath("*", "*", "*")))

    print("nb files:", len(finished))

    indexers = ["subject", "eid", "probe", "region", "N_units"]

    resultslist = []

    failed_load = 0
    for fn in tqdm(finished):
        if os.path.isdir(fn):
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
                    "n_trials": None if result['fit'][i_decoding]['mask'] is None else sum(result['fit'][i_decoding]['mask'][0]) ,
                }
                resultslist.append(tmpdict)
        except Exception as e:
            print(failed_load)
            print(e)
            failed_load += 1
            pass

    print("loading of %i files failed" % failed_load)
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

    if file_output_report is None:
        file_output_report = Path(output_dir).joinpath('collected_results_stage1.pqt')

    print("saving parquet")
    resultsdf.to_parquet(file_output_report)
    print("parquet saved")

target = 'wheel-velocity'
output_dir = '/mnt/s0/bwm/wheel_rerun/wheel-velocity'
file_output_report = '/mnt/s0/bwm/wheel_rerun/parede_N100.pqt'

output_dir = '/mnt/s0/bwm/wheel_rerun/wheel-velocity_N200'
file_output_report = '/mnt/s0/bwm/wheel_rerun/parede_N200.pqt'

target = 'wheel-velocity_N200'
output_dir = '/mnt/home/owinter/ceph/bwm/wheel_rerun'
file_output_report = '/mnt/home/owinter/ceph/bwm/wheel_rerun/sdsc_N200.pqt'

target = 'wheel-velocity'
output_dir = '/mnt/home/owinter/ceph/bwm/wheel_rerun'
file_output_report = '/mnt/home/owinter/ceph/bwm/wheel_rerun/sdsc_partial.pqt'
format_outputs_stage1(output_dir, target, file_output_report)

# scp popeye:/mnt/home/owinter/ceph/bwm/wheel_rerun/*.pqt /mnt/s0/bwm/wheel_rerun

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format outputs stage 1')
    parser.add_argument('output_dir')
    parser.add_argument('target')
    parser.add_argument('--file_output_report', type=str, default=None)

    args = parser.parse_args()
    output_dir = str(args.output_dir)
    target = str(args.target)
