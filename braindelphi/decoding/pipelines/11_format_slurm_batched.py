import pickle
from behavior_models.models.utils import format_data as format_data_mut
import pandas as pd
import glob
from braindelphi.decoding.settings import *
import models.utils as mut
from braindelphi.params import FIT_PATH
from braindelphi.decoding.settings import modeldispatcher
from tqdm import tqdm


date = "50-10-2022"
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
        for i_nb_decodings in range(len(result["fit"])):
            tmpdict = {
                **{x: result[x] for x in indexers},
                "fold": -1,
                "pseudo_id": result["fit"][i_nb_decodings]["pseudo_id"],
                "N_units": result["N_units"],
                "run_id": result["fit"][i_nb_decodings]["run_id"] + 1,
                "R2_test": result["fit"][i_nb_decodings]["Rsquared_test_full"],
            }
            if result["fit"][i_nb_decodings]["full_neurometric"] is not None:
                tmpdict = {
                    **tmpdict,
                    **{
                        idx_neuro: result["fit"][i_nb_decodings]["full_neurometric"][
                            idx_neuro
                        ]
                        for idx_neuro in indexers_neurometric
                    },
                }
            resultslist.append(tmpdict)
    except:
        failed_load += 1
        pass
print("loading of %i files failed" % failed_load)
resultsdf = pd.DataFrame(resultslist)

estimatorstr = strlut[ESTIMATOR]

if NEURAL_DTYPE == "ephys":
    start_tw, end_tw = TIME_WINDOW
if NEURAL_DTYPE == "widefield":
    start_tw = WFI_NB_FRAMES_START
    end_tw = WFI_NB_FRAMES_END

model_str = "interIndividual" if isinstance(MODEL, str) else modeldispatcher[MODEL]

fn = str(
    FIT_PATH.joinpath(
        kwargs["neural_dtype"],
        "_".join(
            [
                date,
                "decode",
                TARGET,
                model_str if TARGET in ["prior", "pLeft"] else "task",
                estimatorstr,
                "align",
                ALIGN_TIME,
                str(N_PSEUDO),
                "pseudosessions",
                "regionWise" if SINGLE_REGION else "allProbes",
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
