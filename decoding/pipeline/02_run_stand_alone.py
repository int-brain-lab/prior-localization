from pathlib import Path
import pandas as pd
from decode_prior import fit_eid

output_path = Path("/datadisk/Data/taskforces/bwm")

insdf = pd.read_parquet(output_path.joinpath('insertions.pqt'))

# sessdf = insdf.sort_values('subject').set_index(['subject', 'eid'])
# todo loop over eids
fns = fit_eid(eid, insdf, modelfit_path=output_path.joinpath('models'), output_path=output_path.joinpath('results'))
