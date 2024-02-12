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

#index = np.array([40, 41, 42, 59, 60, 61, 86, 87, 88])[index]

# import most recent cached data
#bwmdf, _ = load_metadata(
#    CACHE_PATH.joinpath("*_%s_metadata.pkl" % kwargs["neural_dtype"]).as_posix()
#)
mypath = "/home/share/pouget_lab/cache_final/2023-12-22 09:43:59.319410_ephys_metadata_BWM_merged_final.pkl"
#mypath = "/home/share/pouget_lab/cache_final/2023-12-25 10:10:17.809664_ephys_metadata_BWM_unmerged_final.pkl"
bwmdf = pickle.load(open(mypath, "rb")) 

#eids = [
#    "1ec23a70-b94b-4e9c-a0df-8c2151da3761",
#]
#bwmdf["dataset_filenames"] = bwmdf["dataset_filenames"][
#    bwmdf["dataset_filenames"].eid.isin(eids)
#]

if kwargs["use_imposter_session"]:
    kwargs["imposterdf"] = pd.read_parquet(
        IMPOSTER_SESSION_PATH.joinpath("imposterSessions_beforeRecordings.pqt")
	#IMPOSTER_SESSION_PATH.joinpath("imposterSessions_ephysWorld.pqt")
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

metadata['my_seed'] = index

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
