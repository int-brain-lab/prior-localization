# Standard library
import logging
import pickle
from datetime import datetime as dt
from pathlib import Path
from prior_pipelines.pipelines.utils_common_pipelines import load_ephys
from prior_pipelines.pipelines.utils_common_pipelines import cache_regressors


# Third party libraries
import dask
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dask.distributed import LocalCluster

# IBL libraries
from one.api import ONE
from brainwidemap import bwm_query
from prior_pipelines.params import CACHE_PATH

CACHE_PATH.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger("prior_pipelines")

one = ONE()

# Parameters
ALGN_RESOLVED = True
DATE = str(dt.today())
QC = True
TYPE = "primaries"
MERGE_PROBES = False
# End parameters

# Construct params dict from above
params = {
    "ret_qc": QC,
}

dataset_futures = []

one = ONE()
one.alyx.clear_rest_cache()
bwm_df = bwm_query().set_index(["subject", "eid"])  # freeze="2022_10_update"
from tqdm import tqdm

for eid in tqdm(bwm_df.index.unique(level="eid")):
    session_df = bwm_df.xs(eid, level="eid")
    subject = session_df.index[0]
    # If there are two probes, there are two options:
    # load and save data from each probe independently, or merge the data from both probes
    pids = session_df.pid.to_list()
    probe_names = session_df.probe_name.to_list()
    if MERGE_PROBES:
        load_outputs = load_ephys(eid, pids, one=one, **params)
        save_future = cache_regressors(
            subject, eid, "merged_probes", {**params, "type": TYPE}, load_outputs
        )
        dataset_futures.append([subject, eid, "merged_probes", save_future])
    else:
        for (pid, probe_name) in zip(pids, probe_names):
            load_outputs =  load_ephys(eid, [pid], one=one, **params)
            save_future = cache_regressors(
                subject, eid, probe_name, {**params, "type": TYPE}, load_outputs
            )
            dataset_futures.append([subject, eid, probe_name, save_future])

dataset = [
    {
        "subject": x[0],
        "eid": x[1],
        "probe_name": x[2],
        "meta_file": dataset_futures[-1][0],
        "reg_file": dataset_futures[-1][1],
    }
    for i, x in enumerate(dataset_futures)
]
dataset = pd.DataFrame(dataset)

outdict = {"params": params, "dataset_filenames": dataset}
with open(Path(CACHE_PATH).joinpath(DATE + "_ephys_metadata.pkl"), "wb") as fw:
    pickle.dump(outdict, fw)
