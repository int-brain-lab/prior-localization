import pickle
import pandas as pd
import glob
from braindelphi.decoding.settings import *
from braindelphi.params import FIT_PATH
from braindelphi.decoding.settings import modeldispatcher
from tqdm import tqdm

date = "30-01-2023"
finished = glob.glob(
    str(FIT_PATH.joinpath(kwargs["neural_dtype"], "*", "*", "*", "*%s*" % date))
)
print("nb files:", len(finished))

indexers = ["subject", "eid", "probe", "region"]

resultslist = []
weights, predictions, R2_test, intercepts, targets, masks = [], [], [], [], [], []
nb_runs = 0
failed_load = 0
for fn in tqdm(finished):
    try:
        fo = open(fn, "rb")
        result = pickle.load(fo)
        fo.close()
        if result["fit"] is None:
            continue
        for i_decoding in range(len(result["fit"])):
            if i_decoding == 0:
                pseudo_id = result["fit"][i_decoding]["pseudo_id"]

            if result["fit"][i_decoding]["pseudo_id"] == pseudo_id:
                weights.append(np.vstack(result["fit"][i_decoding]["weights"]).mean(axis=0))
                predictions.append(result["fit"][i_decoding]["predictions_test"])
                intercepts.append(np.mean(result["fit"][i_decoding]["intercepts"]))
                targets.append(result["fit"][i_decoding]["target"])
                R2_test.append(result["fit"][i_decoding]["Rsquared_test_full"])
                masks.append(np.array([str(item)
                                       for item in list(result["fit"][i_decoding]["mask"].values * 1)], dtype=float))
                N_units = result["N_units"]
                nb_runs += 1
            else:
                tmpdict = {
                    **{x: result[x] for x in indexers},
                    "fold": -1,
                    "pseudo_id": pseudo_id,
                    "N_units": N_units,
                    "nb_runs": nb_runs,
                    "mask": np.array(masks).mean(axis=0).tolist(),
                    "R2_test": np.array(R2_test).mean(),
                    "prediction": np.array(predictions).squeeze().mean(axis=0).tolist(),
                    "target": np.array(targets).squeeze().mean(axis=0).tolist(),
                    "weights": np.array(weights).mean(axis=0).tolist(),
                    "intercepts": np.array(intercepts).mean(),
                }
                resultslist.append(tmpdict)
                weights, predictions, R2_test, intercepts, targets, masks = [], [], [], [], [], []
                pseudo_id = result["fit"][i_decoding]["pseudo_id"]
                nb_runs = 0
    except:
        print(failed_load)
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
