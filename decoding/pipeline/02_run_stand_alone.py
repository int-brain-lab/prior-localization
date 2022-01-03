from pathlib import Path
import pandas as pd
from decode_prior import fit_eid
import numpy as np

output_path = Path("/datadisk/Data/taskforces/bwm")
output_path.joinpath('models').mkdir(exist_ok=True)
output_path.joinpath('results').mkdir(exist_ok=True)


insdf = pd.read_parquet(output_path.joinpath('insertions.pqt'))


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
for i, eid in enumerate(eids):
    if i < IMIN:
        continue
    if eid in excludes:
        continue
    if np.any(insdf[insdf['eid'] == eid]['spike_sorting'] == ""):
        print(f"dud {eid}")
        continue
    print(f"{i}, session: {eid}")
    fns = fit_eid(eid, insdf, modelfit_path=output_path.joinpath('models'), output_path=output_path.joinpath('results'))
