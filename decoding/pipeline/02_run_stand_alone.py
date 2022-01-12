"""
On the second part of the pipeline example, we loop over the dataframe
The analysis tools load the downloaded and cached version of the data.
"""
from pathlib import Path
import pandas as pd
from decode_prior import fit_eid
import numpy as np
import decoding_utils as dut

DECODING_PATH = Path("/home/users/f/findling/ibl/prior-localization/decoding")
# create necessary empty directories if not existing
DECODING_PATH.joinpath('results').mkdir(exist_ok=True)
DECODING_PATH.joinpath('results', 'behavioral').mkdir(exist_ok=True)
DECODING_PATH.joinpath('results', 'neural').mkdir(exist_ok=True)
insdf = pd.read_parquet(DECODING_PATH.joinpath('insertions.pqt'))
insdf = insdf[insdf.spike_sorting!=""]

eids = insdf['eid'].unique()
# sessdf = insdf.sort_values('subject').set_index(['subject', 'eid'])
# todo loop over eids

excludes = [
    'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',  # inconsistent trials object: relaunched task on 31-12-2021
    'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',  # same same
    '7a887357-850a-4378-bd2a-b5bc8bdd3aac',  # same same
    '56b57c38-2699-4091-90a8-aba35103155e',  # load obect pickle error
    '09394481-8dd2-4d5c-9327-f2753ede92d7',  # same same
]
IMIN = 0
nb_common_regions = np.zeros(len(eids))
for i, eid in enumerate(eids):
    if i < IMIN:
        continue
    if eid in excludes:
        continue
    if np.any(insdf[insdf['eid'] == eid]['spike_sorting'] == ""):
        print(f"dud {eid}")
        continue
    print(f"{i}, session: {eid}")
    regions = dut.return_regions(eid, insdf, QC_CRITERIA=1, NUM_UNITS=10)
    if len(regions.keys()) > 1:
        nb_common_regions[i] = len(list(set(regions['probe01']).intersection(regions['probe00'])))
    """
    fns = fit_eid(eid,
                  insdf,
                  modelfit_path=DECODING_PATH.joinpath('results', 'behavioral'),
                  output_path=DECODING_PATH.joinpath('results', 'neural'),
                  pseudo_id=-1,
                  nb_runs=3,
                  one=one)
    """