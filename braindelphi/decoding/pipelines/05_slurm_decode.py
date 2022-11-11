import pandas as pd
import sys
from braindelphi.decoding.settings import kwargs, N_PSEUDO_PER_JOB, N_PSEUDO
from braindelphi.decoding.functions.decoding import fit_eid
import numpy as np
from braindelphi.params import CACHE_PATH, IMPOSTER_SESSION_PATH
from braindelphi.decoding.functions.utils import load_metadata
import pickle

try:
    index = int(sys.argv[1]) - 1
except:
    index = 129
    pass

# import most recent cached data
bwmdf, _ = load_metadata(
    CACHE_PATH.joinpath("*_%s_metadata.pkl" % kwargs["neural_dtype"]).as_posix()
)

eids = [
    "07dc4b76-5b93-4a03-82a0-b3d9cc73f412",
    "09b2c4d1-058d-4c84-9fd4-97530f85baf6",
    "0c828385-6dd6-4842-a702-c5075f5f5e81",
    "111c1762-7908-47e0-9f40-2f2ee55b6505",
    "413a6825-2144-4a50-b3fc-cf38ddd6fd1a",
    "41431f53-69fd-4e3b-80ce-ea62e03bf9c7",
    "58b1e920-cfc8-467e-b28b-7654a55d0977",
    "5d01d14e-aced-4465-8f8e-9a1c674f62ec",
    "6274dda8-3a59-4aa1-95f8-a8a549c46a26",
    "8a3a0197-b40a-449f-be55-c00b23253bbf",
    "90d1e82c-c96f-496c-ad4e-ee3f02067f25",
    "bda2faf5-9563-4940-a80f-ce444259e47b",
    "e0928e11-2b86-4387-a203-80c77fab5d52",
    "e0928e11-2b86-4387-a203-80c77fab5d52",
]
bwmdf["dataset_filenames"] = bwmdf["dataset_filenames"][
    bwmdf["dataset_filenames"].eid.isin(eids)
]

if kwargs["use_imposter_session"]:
    kwargs["imposterdf"] = pd.read_parquet(
        IMPOSTER_SESSION_PATH.joinpath("imposterSessions_beforeRecordings.pqt")
    )
else:
    kwargs["imposterdf"] = None

pid_id = index % bwmdf["dataset_filenames"].index.size
job_id = index // bwmdf["dataset_filenames"].index.size

pid = bwmdf["dataset_filenames"].iloc[pid_id]
metadata = pickle.load(open(pid.meta_file, "rb"))
regressors = pickle.load(open(pid.reg_file, "rb"))

if kwargs["neural_dtype"] == "widefield":
    trials_df, neural_dict = regressors
else:
    trials_df, neural_dict = regressors["trials_df"], regressors

# metadata['probe_name'] = 'probe00'
if (job_id + 1) * N_PSEUDO_PER_JOB <= N_PSEUDO:
    print(f"pid_id: {pid_id}")
    pseudo_ids = (
        np.arange(job_id * N_PSEUDO_PER_JOB, (job_id + 1) * N_PSEUDO_PER_JOB) + 1
    )
    if 1 in pseudo_ids:
        pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype("int64")
    results_fit_eid = fit_eid(
        neural_dict=neural_dict,
        trials_df=trials_df,
        metadata=metadata,
        pseudo_ids=pseudo_ids,
        **kwargs,
    )
print("Slurm job successful")
