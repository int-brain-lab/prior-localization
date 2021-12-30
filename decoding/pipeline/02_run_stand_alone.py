from pathlib import Path
import pandas as pd
from decode_prior import fit_eid

output_path = Path("/datadisk/Data/taskforces/bwm")
output_path.joinpath('models').mkdir(exist_ok=True)
output_path.joinpath('results').mkdir(exist_ok=True)


insdf = pd.read_parquet(output_path.joinpath('insertions.pqt'))

eids = insdf['eid'].unique()
# sessdf = insdf.sort_values('subject').set_index(['subject', 'eid'])
# todo loop over eids
for eid in eids:
    fns = fit_eid(eid, insdf, modelfit_path=output_path.joinpath('models'), output_path=output_path.joinpath('results'))
