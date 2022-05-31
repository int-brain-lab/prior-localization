"""
On the second part of the pipeline example, we loop over the dataframe
The analysis tools load the downloaded and cached version of the data.
"""
import logging
import numpy as np
from pathlib import Path
import pandas as pd
import yaml

from braindelphi.params import braindelphi_PATH, SETTINGS_PATH, FIT_PATH
from braindelphi.decoding.functions.decoding import fit_eid
from braindelphi.decoding.functions.utils import check_settings

# sessions to be excluded (by Olivier Winter)
excludes = [
    'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',  # inconsistent trials object: relaunched on 31-12-2021
    'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',  # same same
    '7a887357-850a-4378-bd2a-b5bc8bdd3aac',  # same same
    '56b57c38-2699-4091-90a8-aba35103155e',  # load obect pickle error
    '09394481-8dd2-4d5c-9327-f2753ede92d7',  # same same
]


# load settings as a dict
settings = yaml.safe_load(open(SETTINGS_PATH))
kwargs = check_settings(settings)

# set up logging
log_file = FIT_PATH.joinpath('decoding_%s_%s.log' % (kwargs['target'], kwargs['date']))
logging.basicConfig(
    filename=log_file, filemode='w', level=logging.DEBUG,
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # add logging to console

# load insertion data
insdf = pd.read_parquet(braindelphi_PATH.joinpath('decoding', 'insertions.pqt'))
insdf = insdf[insdf.spike_sorting != ""]
eids = insdf['eid'].unique()

# load imposter session df
imposter_path = braindelphi_PATH.joinpath('decoding', 'imposterSessions_%s.pqt' % params['target'])
imposter_df = pd.read_parquet(imposter_path)

# loop over sessions: load data and decode
IMIN = 0
for i, eid in enumerate(eids[:1]):

    # determine if we should proceed with decoding session
    if i < IMIN:
        continue
    if eid in excludes:
        continue
    if np.any(insdf[insdf['eid'] == eid]['spike_sorting'] == ""):
        print(f"dud {eid}")
        continue

    print(f"{i}, session: {eid}")

    # load data

    # fit model
    fns = fit_eid(
        eid,
        insdf,
        modelfit_path=DECODING_PATH.joinpath('results', 'behavioral'),
        output_path=DECODING_PATH.joinpath('results', 'neural'),
        pseudo_id=-1,
        nb_runs=3,
        one=one
    )
