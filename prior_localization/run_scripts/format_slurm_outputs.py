import pickle
import pandas as pd
import glob
from tqdm import tqdm
import gc
from prior_localization.functions.utils import check_config
from pathlib import Path

config = check_config()
output_dir = "/home/users/f/findling/prior-public-code/prior_localization/results"

finished = glob.glob(
    str(Path(output_dir).joinpath("*", "*", "*"))
)
print("nb files:", len(finished))

indexers = ["subject", "eid", "probe", "region"]

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
                "N_units": result["N_units"],
                "run_id": result["fit"][i_decoding]["run_id"] + 1,
                "R2_test": result["fit"][i_decoding]["Rsquared_test_full"],
            }
            resultslist.append(tmpdict)
    except:
        print(failed_load)
        failed_load += 1
        pass
print("loading of %i files failed" % failed_load)

resultsdf = pd.DataFrame(resultslist)

fn = str(
    Path(output_dir).joinpath(
        "_".join(
            [
                "ephys",
                "decode",
                "Pleft",
                config['estimator'].__name__,
                "align",
                'onset',
                "timeWindow",
                "0_6",
                "0_1",
            ]
        ),
    )
)

fn = fn + ".parquet"

print("saving parquet")
resultsdf.to_parquet(fn)
print("parquet saved")

