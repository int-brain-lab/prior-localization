import pandas as pd
import sys
from settings.settings import *
from functions.decoding import fit_eid
import numpy as np

try:
    index = int(sys.argv[1]) - 1
except:
    index = 0
    pass

# import cached data
insdf = pd.read_parquet(DECODING_PATH.joinpath('insertions.pqt')).reset_index(drop=True)
insdf = insdf[insdf.spike_sorting != '']
eids = insdf['eid'].unique()

# create necessary empty directories if not existing
DECODING_PATH.joinpath('results').mkdir(exist_ok=True)
DECODING_PATH.joinpath('results', 'behavioral').mkdir(exist_ok=True)
DECODING_PATH.joinpath('results', 'neural').mkdir(exist_ok=True)
DECODING_PATH.joinpath('logs').mkdir(exist_ok=True)
DECODING_PATH.joinpath('logs', 'slurm').mkdir(exist_ok=True)

if USE_IMPOSTER_SESSION:
    imposterdf = pd.read_parquet(DECODING_PATH.joinpath('imposterSessions_beforeRecordings.pqt'))
else:
    imposterdf_future = None

kwargs = {'imposterdf': None, 'nb_runs': N_RUNS, 'single_region': SINGLE_REGION, 'merged_probes': MERGED_PROBES,
          'modelfit_path': DECODING_PATH.joinpath('results', 'behavioral'), 'continuous_target': CONTINUOUS_TARGET,
          'output_path': DECODING_PATH.joinpath('results', 'neural'), 'one': None,
          'estimator_kwargs': ESTIMATOR_KWARGS, 'hyperparam_grid': HPARAM_GRID,
          'save_binned': SAVE_BINNED, 'shuffle': SHUFFLE, 'balanced_weight': BALANCED_WEIGHT,
          'normalize_input': NORMALIZE_INPUT, 'normalize_output': NORMALIZE_OUTPUT,
          'compute_on_each_fold': COMPUTE_NEURO_ON_EACH_FOLD,
          'force_positive_neuro_slopes': FORCE_POSITIVE_NEURO_SLOPES,
          'estimator': ESTIMATOR, 'target': TARGET, 'model': MODEL, 'align_time': ALIGN_TIME,
          'no_unbias': NO_UNBIAS, 'min_rt': MIN_RT, 'min_behav_trials': MIN_BEHAV_TRIAS,
          'qc_criteria': QC_CRITERIA, 'min_units': MIN_UNITS, 'time_window': TIME_WINDOW,
          'use_imposter_session': USE_IMPOSTER_SESSION, 'compute_neurometric': COMPUTE_NEUROMETRIC,
          'border_quantiles_neurometric': BORDER_QUANTILES_NEUROMETRIC, 'today': DATE,
          'add_to_saving_path': ADD_TO_SAVING_PATH
          }


eid_id = index % eids.size
job_id = index // eids.size

eid = eids[eid_id]
if (eid in excludes or np.any(insdf[insdf['eid'] == eid]['spike_sorting'] == "")):
    print(f"dud {eid}")
else:
    print(f"session: {eid}")
    pseudo_ids = np.arange(job_id * N_PSEUDO_PER_JOB, (job_id + 1) * N_PSEUDO_PER_JOB) + 1
    if 1 in pseudo_ids:
        pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype('int64')
    fit_eid(eid=eid, sessdf=insdf, pseudo_ids=pseudo_ids, **kwargs)

print('Slurm job successful')