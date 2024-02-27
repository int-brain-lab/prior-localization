import argparse
import os
import pickle
import pandas as pd
import glob
from tqdm import tqdm
import gc
from pathlib import Path

parser = argparse.ArgumentParser(description='Format outputs')
parser.add_argument('output_dir')
parser.add_argument('target')

args = parser.parse_args()
output_dir = str(args.output_dir)
target = str(args.target)

output_dir = Path(output_dir).joinpath(target)
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
                "n_trials": sum(result['fit'][i_decoding]['mask'][0]),
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
print("saving parquet")
resultsdf.to_parquet(fn)
print("parquet saved")
