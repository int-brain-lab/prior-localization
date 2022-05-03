"""
On the second part of the pipeline example, we loop over the dataframe
The analysis tools load the downloaded and cached version of the data.
"""
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path

from one.api import ONE

import sys
sys.path.append('/home/mattw/Dropbox/github/int-brain-lab/prior-localization/brainwide/decoding')
from functions import utils as dut
from functions import decoding_continuous as decode
from settings.settings_continuous import fit_metadata


# set up logging
log_file = fit_metadata['output_path'].joinpath(
    'results', 'neural', 'decoding_%s.log' % fit_metadata['target'])
logging.basicConfig(
    filename=log_file, filemode='w', level=logging.DEBUG,
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # add logging to console

# create necessary empty directories if not existing
fit_metadata['output_path'].joinpath('results').mkdir(exist_ok=True)
fit_metadata['output_path'].joinpath('results', 'neural').mkdir(exist_ok=True)
insdf = pd.read_parquet(fit_metadata['output_path'].joinpath('insertions.pqt'))
insdf = insdf[insdf.spike_sorting != ""]

eids = insdf['eid'].unique()
# sessdf = insdf.sort_values('subject').set_index(['subject', 'eid'])

one = ONE()

# session to be excluded (by Olivier Winter)
excludes = [
    'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',  # inconsistent trials object: relaunched on 31-12-2021
    'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',  # same same
    '7a887357-850a-4378-bd2a-b5bc8bdd3aac',  # same same
    '56b57c38-2699-4091-90a8-aba35103155e',  # load obect pickle error
    '09394481-8dd2-4d5c-9327-f2753ede92d7',  # same same
]
IMIN = 0
IMAX = 1000
nb_common_regions = np.zeros(len(eids))

output_path = fit_metadata.pop('output_path').joinpath('results', 'neural')

for i, eid in enumerate(eids):

    if i >= IMAX:
        continue
    if i < IMIN:
        continue
    if eid in excludes:
        continue
    if np.any(insdf[insdf['eid'] == eid]['spike_sorting'] == ""):
        print(f"dud {eid}")
        continue

    logging.log(logging.DEBUG, f"{i}, session: {eid}")
    regions = dut.return_regions(
        eid, insdf, QC_CRITERIA=fit_metadata['qc_criteria'], NUM_UNITS=fit_metadata['min_units'])
    if len(regions.keys()) > 1:
        nb_common_regions[i] = len(list(set(regions['probe01']).intersection(regions['probe00'])))

    fns = decode.fit_eid(
        eid,
        insdf,
        output_path=output_path,
        pseudo_ids=[-1],
        one=one,
        **fit_metadata,
    )
