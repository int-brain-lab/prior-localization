import argparse
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

finished = glob.glob(str(Path(output_dir).joinpath(target, "*", "*", "*")))

print("nb files:", len(finished))

indexers = ["subject", "eid", "probe", "region", "N_units"]

resultslist = []

failed_load = 0
for fn in tqdm(finished):
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
            }
            resultslist.append(tmpdict)
    except:
        print(failed_load)
        failed_load += 1
        pass
print("loading of %i files failed" % failed_load)

resultsdf = pd.DataFrame(resultslist)

fn = str(Path(output_dir).joinpath('collected_results.pqt'))
print("saving parquet")
resultsdf.to_parquet(fn)
print("parquet saved")
