# Standard library
import logging
import pickle
from datetime import datetime as dt
from pathlib import Path
from prior_code.caching_pipelines.utils_pipelines import load_ephys
from prior_code.caching_pipelines.utils_pipelines import cache_regressors
import numpy as np

# Third party libraries
import pandas as pd

# IBL libraries
from one.api import ONE
from brainwidemap import bwm_query
from prior_code.params import CACHE_PATH

from tqdm import tqdm

CACHE_PATH.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger("code")


# Parameters
ALGN_RESOLVED = True
DATE = str(dt.today())
QC = True
TYPE = "primaries"
# End parameters

# Construct params dict from above
params = {
    "ret_qc": QC,
}

dataset_futures = []

one = ONE()
one.alyx.clear_rest_cache()
bwm_df = bwm_query().set_index(["subject", "eid"])

dataset_ = []
fails = []
for eid in tqdm(bwm_df.index.unique(level="eid")):
    session_df = bwm_df.xs(eid, level="eid")
    subject = session_df.index[0]
    # If there are two probes, there are two options:
    # load and save data from each probe independently, or merge the data from both probes
    pids = session_df.pid.to_list()
    probe_names = session_df.probe_name.to_list()
    for (pid, probe_name) in zip(pids, probe_names):
        try:
            outputs = load_ephys(eid, [pid], one=one, **params)
            meta_file, reg_file = cache_regressors(
                subject, eid, probe_name, {**params, "type": TYPE}, outputs
            )
            dataset_.append([subject, eid, probe_name, meta_file, reg_file])
        except:
            fails.append((pid, probe_name))
            pass

for eid, eid_probe_name in np.array(fails):
    if eid not in bwm_df.index.get_level_values(1):
        print(eid)
        continue
    session_df = bwm_df.xs(eid, level="eid")
    subject = session_df.index[0]

    # If there are two probes, there are two options:
    # load and save data from each probe independently, or merge the data from both probes
    pids = session_df.pid.to_list()
    for (pid, probe_name) in zip(pids, probe_names):
        if probe_name == eid_probe_name:
            outputs = load_ephys(eid, [pid], one=one, **params)
            meta_file, reg_file = cache_regressors(
                subject, eid, probe_name, {**params, "type": TYPE}, outputs
            )
            dataset_.append([subject, eid, probe_name, meta_file, reg_file])

dataset = pd.DataFrame(
    dataset_, columns=["subject", "eid", "probe_name", "meta_file", "reg_file"]
)

outdict = {"params": params, "dataset_filenames": dataset}
with open(Path(CACHE_PATH).joinpath(DATE + "_ephys_metadata.pkl"), "wb") as fw:
    pickle.dump(outdict, fw)
