import pickle
import pandas as pd
import glob
from prior_pipelines.decoding.settings import *
from prior_pipelines.params import FIT_PATH
from prior_pipelines.decoding.settings import modeldispatcher
from tqdm import tqdm
import gc

SAVE_KFOLDS = False

date = "30-01-2023"
finished = glob.glob(
    str(FIT_PATH.joinpath(kwargs["neural_dtype"], "*", "*", "*", "*%s*" % date))
)
print("nb files:", len(finished))

indexers = ["subject", "eid", "probe", "region"]
indexers_neurometric = [
    "low_slope",
    "high_slope",
    "low_range",
    "high_range",
    "shift",
    "mean_range",
    "mean_slope",
]

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
                "mask": "".join(
                    [
                        str(item)
                        for item in list(result["fit"][i_decoding]["mask"].values * 1)
                    ]
                ),
                "R2_test": result["fit"][i_decoding]["Rsquared_test_full"],
            }
            if result["fit"][i_decoding]["full_neurometric"] is not None:
                tmpdict = {
                    **tmpdict,
                    **{
                        idx_neuro: result["fit"][i_decoding]["full_neurometric"][
                            idx_neuro
                        ]
                        for idx_neuro in indexers_neurometric
                    },
                }
            resultslist.append(tmpdict)

            if SAVE_KFOLDS:
                for kfold in range(result["fit"][i_decoding]["n_folds"]):
                    tmpdict = {
                        **{x: result[x] for x in indexers},
                        "fold": kfold,
                        "pseudo_id": result["fit"][i_decoding]["pseudo_id"],
                        "N_units": result["N_units"],
                        "run_id": result["fit"][i_decoding]["run_id"] + 1,
                        "R2_test": result["fit"][i_decoding]["scores_test"][kfold],
                        "Best_regulCoef": result["fit"][i_decoding]["best_params"][
                            kfold
                        ],
                    }
                    if result["fit"][i_decoding]["fold_neurometric"] is not None:
                        tmpdict = {
                            **tmpdict,
                            **{
                                idx_neuro: result["fit"][i_decoding][
                                    "fold_neurometric"
                                ][kfold][idx_neuro]
                                for idx_neuro in indexers_neurometric
                            },
                        }
                    resultslist.append(tmpdict)
    except:
        print(failed_load)
        failed_load += 1
        pass
print("loading of %i files failed" % failed_load)

resultsdf = pd.DataFrame(resultslist)

if NEURAL_DTYPE == "ephys":
    start_tw, end_tw = TIME_WINDOW
if NEURAL_DTYPE == "widefield":
    start_tw = WFI_NB_FRAMES_START
    end_tw = WFI_NB_FRAMES_END

fn = str(
    FIT_PATH.joinpath(
        kwargs["neural_dtype"],
        "_".join(
            [
                date,
                "decode",
                TARGET,
                modeldispatcher[MODEL] if TARGET in ["prior", "pLeft"] else "task",
                strlut[ESTIMATOR],
                "align",
                ALIGN_TIME,
                "timeWindow",
                str(start_tw).replace(".", "_"),
                str(end_tw).replace(".", "_"),
            ]
        ),
    )
)

if COMPUTE_NEUROMETRIC:
    fn = fn + "_".join(["", "neurometricPLeft", modeldispatcher[MODEL]])

if ADD_TO_SAVING_PATH != "":
    fn = fn + "_" + ADD_TO_SAVING_PATH

fn = fn + ".parquet"

metadata_df = pd.Series({"filename": fn, "date": date, **fit_metadata})
metadata_fn = ".".join([fn.split(".")[0], "metadata", "pkl"])
print("saving parquet")
resultsdf.to_parquet(fn)
print("parquet saved")
print("saving metadata")
metadata_df.to_pickle(metadata_fn)
print("metadata saved")
