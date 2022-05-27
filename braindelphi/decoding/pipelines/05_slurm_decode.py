import pandas as pd
import sys
from braindelphi.decoding.settings import kwargs
from braindelphi.decoding.functions.decoding import fit_eid
import numpy as np
from braindelphi.params import IMPOSTER_SESSION_PATH

try:
    index = int(sys.argv[1]) - 1
except:
    index = 185
    pass

# import cached data
bwmdf = pd.read_parquet(DECODING_PATH.joinpath('insertions.pqt')).reset_index(drop=True)
bwmdf = bwmdf[bwmdf.spike_sorting != '']
eids = bwmdf['eid'].unique()

if kwargs['imposter_session']:
    imposterdf = pd.read_parquet(DECODING_PATH.joinpath('imposterSessions_beforeRecordings.pqt'))
else:
    imposterdf = None

kwargs = {**kwargs, 'imposterdf': imposterdf}

if WIDE_FIELD_IMAGING:
    import glob
    subjects = glob.glob('wide_field_imaging/CSK-im-*')
    eids = np.array([np.load(s + '/behavior.npy', allow_pickle=True).size for s in subjects])
    eid_id = index % eids.sum()
    job_id = index // eids.sum()
    subj_id = np.sum(eid_id >= eids.cumsum())
    sess_id = eid_id - np.hstack((0, eids)).cumsum()[:-1][subj_id]
    if sess_id < 0:
        raise ValueError('There is an error in the code')
    sessiondf, wideFieldImaging_dict = wut.load_wfi_session(subjects[subj_id], sess_id)
    eid = sessiondf.eid[sessiondf.session_to_decode].unique()[0]
else:
    eid_id = index % eids.size
    job_id = index // eids.size
    eid = eids[eid_id]
    sessiondf, wideFieldImaging_dict = None, None

if (job_id + 1) * N_PSEUDO_PER_JOB <= N_PSEUDO:
    if WIDE_FIELD_IMAGING and eid in excludes or np.any(bwmdf[bwmdf['eid'] == eid]['spike_sorting'] == ""):
        print(f"dud {eid}")
    else:
        print(f"session: {eid}")
        pseudo_ids = np.arange(job_id * N_PSEUDO_PER_JOB, (job_id + 1) * N_PSEUDO_PER_JOB) + 1
        if 1 in pseudo_ids:
            pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype('int64')
        fit_eid(eid=eid, bwmdf=bwmdf, pseudo_ids=pseudo_ids,
                sessiondf=sessiondf, wideFieldImaging_dict=wideFieldImaging_dict, **kwargs)
    print('Slurm job successful')
else:
    print('index is too high, which would lead to generating more sessions than expected')

