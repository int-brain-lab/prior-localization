import pickle
import pandas as pd
import glob
from braindelphi.decoding.settings import *
from braindelphi.params import FIT_PATH
from braindelphi.decoding.settings import modeldispatcher
from tqdm import tqdm

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
            # side, stim, act, _ = format_data_mut(result["fit"][i_decoding]["df"])
            # mask = result["fit"][i_decoding]["mask"]  # np.all(result["fit"][i_decoding]["target"] == stim[mask])
            # full_test_prediction = np.zeros(np.array(result["fit"][i_decoding]["target"]).size)
            # for k in range(len(result["fit"][i_decoding]["idxes_test"])):
            #    full_test_prediction[result["fit"][i_decoding]['idxes_test'][k]] = result["fit"][i_decoding]['predictions_test'][k]
            # neural_act = np.sign(full_test_prediction)
            # perf_allcontrasts = (side.values[mask][neural_act != 0] == neural_act[neural_act != 0]).mean()
            # perf_allcontrasts_prevtrial = (side.values[mask][1:] == neural_act[:-1])[neural_act[:-1] != 0].mean()
            # perf_0contrasts = (side.values[mask] == neural_act)[(stim[mask] == 0) * (neural_act != 0)].mean()
            # nb_trials_act_is_0 = (neural_act == 0).mean()
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
                "prediction": list(result["fit"][i_decoding]["predictions_test"]),
                "target": list(result["fit"][i_decoding]["target"]),
                "weights": np.vstack(result["fit"][i_decoding]["weights"]).mean(axis=0).tolist(),
                "intercepts": np.mean(result["fit"][i_decoding]["intercepts"]),
                # 'perf_allcontrast': perf_allcontrasts,
                # 'perf_allcontrasts_prevtrial': perf_allcontrasts_prevtrial,
                # 'perf_0contrast': perf_0contrasts,
                # 'nb_trials_act_is_0': nb_trials_act_is_0,
            }
            #            if 'predictions_test' in result['fit'][i_decoding].keys():
            #                tmpdict = {**tmpdict,
            #                           'prediction': result['fit'][i_decoding]['predictions_test'],
            #                           'target': result['fit'][i_decoding]['target']}
            #            if 'acc_test_full' in result['fit'][i_decoding].keys():
            #                tmpdict = {**tmpdict, 'acc_test': result['fit'][i_decoding]['acc_test_full'],
            #                           'balanced_acc_test': result['fit'][i_decoding]['balanced_acc_test_full']}
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

"""
resultsdf = resultsdf[resultsdf.subject == 'NYU-12']
resultsdf = resultsdf[resultsdf.eid == 'a8a8af78-16de-4841-ab07-fde4b5281a03']
resultsdf.region = resultsdf.region.apply(lambda x:x[0])
resultsdf = resultsdf[resultsdf.region == 'CA1']
resultsdf = resultsdf[resultsdf.probe == 'probe00']
resultsdf = resultsdf[resultsdf.run_id == 1]
subdf = resultsdf.set_index(['subject', 'eid', 'probe', 'region']).drop('fold', axis=1)
"""

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
