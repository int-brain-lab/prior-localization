import pandas as pd
import sys
from braindelphi.decoding.settings import kwargs, N_PSEUDO_PER_JOB, N_PSEUDO
from braindelphi.decoding.functions.decoding import fit_eid
import numpy as np
from braindelphi.params import CACHE_PATH, IMPOSTER_SESSION_PATH
import pickle

try:
    index = int(sys.argv[1]) - 1
except:
    index = 5
    pass

# import cached data
bwmdf = pickle.load(open(CACHE_PATH.joinpath('2022-05-30 09:33:57.655433_%s_metadata.pkl' % kwargs['neural_dtype']), 'rb'))

if kwargs['use_imposter_session']:
    imposterdf = pd.read_parquet(IMPOSTER_SESSION_PATH.joinpath('imposterSessions_beforeRecordings.pqt'))
else:
    imposterdf = None

kwargs = {**kwargs, 'imposterdf': imposterdf}

pid_id = index % bwmdf['dataset_filenames'].index.size
job_id = index // bwmdf['dataset_filenames'].index.size

pid = bwmdf['dataset_filenames'].iloc[pid_id]
metadata = pickle.load(open(pid.meta_file, 'rb'))
trials_df, neural_dict = pickle.load(open(pid.reg_file, 'rb'))


if (job_id + 1) * N_PSEUDO_PER_JOB <= N_PSEUDO:
    print(f"pid_id: {pid_id}")
    pseudo_ids = np.arange(job_id * N_PSEUDO_PER_JOB, (job_id + 1) * N_PSEUDO_PER_JOB) + 1
    if 1 in pseudo_ids:
        pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype('int64')
    results_fit_eid = fit_eid(neural_dict=neural_dict, trials_df=trials_df, metadata=metadata,
                              pseudo_ids=pseudo_ids, **kwargs)
print('Slurm job successful')

