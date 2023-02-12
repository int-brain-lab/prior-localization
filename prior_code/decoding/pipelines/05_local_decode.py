import pandas as pd
import sys
from code.decoding.settings import kwargs, N_PSEUDO_PER_JOB, N_PSEUDO
from code.decoding.functions.decoding import fit_eid
import numpy as np
from code.params import CACHE_PATH, IMPOSTER_SESSION_PATH
from code.decoding.functions.utils import load_metadata
import pickle
from tqdm import tqdm

# import most recent cached data
bwmdf, _ = load_metadata(
    CACHE_PATH.joinpath("*_%s_metadata.pkl" % kwargs["neural_dtype"]).as_posix()
)

for index in tqdm(np.arange(bwmdf["dataset_filenames"].index.size)):
    kwargs["imposterdf"] = None

    pid_id = index % bwmdf["dataset_filenames"].index.size

    pid = bwmdf["dataset_filenames"].iloc[pid_id]

    metadata = pickle.load(open(pid.meta_file, "rb"))
    regressors = pickle.load(open(pid.reg_file, "rb"))

    if kwargs["neural_dtype"] == "widefield":
        trials_df, neural_dict = regressors
    else:
        trials_df, neural_dict = regressors["trials_df"], regressors

    pseudo_ids = -np.ones(1).astype("int64")
    results_fit_eid = fit_eid(
        neural_dict=neural_dict,
        trials_df=trials_df,
        metadata=metadata,
        pseudo_ids=pseudo_ids,
        **kwargs,
    )
    print("Slurm job successful")
