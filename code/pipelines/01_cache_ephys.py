# Standard library
import logging
import pickle
from datetime import datetime as dt
from pathlib import Path
from code.pipelines.utils_common_pipelines import load_ephys
from code.pipelines.utils_common_pipelines import cache_regressors


# Third party libraries
import pandas as pd

# IBL libraries
from one.api import ONE
from brainwidemap import bwm_query
from code.params import CACHE_PATH

from tqdm import tqdm

CACHE_PATH.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger("code")


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
    if MERGE_PROBES:
        raise NotImplementedError("There is problem here")
        load_outputs = delayed_load(eid, pids, params)
        save_future = delayed_save(
            subject, eid, "merged_probes", {**params, "type": TYPE}, load_outputs
        )
        # dataset_futures.append([subject, eid, "merged_probes", save_future])
    else:
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

import numpy as np

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
"""
fails
[('04690e35-ab38-41db-982c-50cbdf8d0dd1', 'probe01'),
 ('e10a7a75-4740-41d1-82bb-7696ed14c442', 'probe01'),
 ('100433fa-2c59-4432-8295-aa27657fe3fb', 'probe00'),
 ('d591a59c-b49b-46ba-a914-df379ada9813', 'probe01')]
"""

dataset = pd.DataFrame(
    dataset_, columns=["subject", "eid", "probe_name", "meta_file", "reg_file"]
)

outdict = {"params": params, "dataset_filenames": dataset}
with open(Path(CACHE_PATH).joinpath(DATE + "_ephys_metadata.pkl"), "wb") as fw:
    pickle.dump(outdict, fw)

"""
with open(
    Path(CACHE_PATH).joinpath("2023-01-18 08:50:47.774747_ephys_metadata.pkl"), "rb"
) as fw:
    outdict = pickle.load(fw)
dataset = outdict['dataset_filenames']
dataset_ = dataset.values.tolist()
"""
