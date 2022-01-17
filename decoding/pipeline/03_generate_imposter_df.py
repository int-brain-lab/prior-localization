from pathlib import Path
import pandas as pd
from one.api import ONE
import brainbox.io.one as bbone
import numpy as np
from decode_prior import MIN_BEHAV_TRIAS

DECODING_PATH = Path("/Users/csmfindling/Documents/Postdoc-Geneva/IBL/behavior/prior-localization/decoding")
#DECODING_PATH = Path("/home/users/f/findling/ibl/prior-localization/decoding")
insdf = pd.read_parquet(DECODING_PATH.joinpath('insertions.pqt'))
insdf = insdf[insdf.spike_sorting != '']
eids = insdf['eid'].unique()
columns = ['probabilityLeft', 'contrastRight', 'feedbackType', 'choice', 'contrastLeft', 'eid', 'template_sess']

one = ONE(mode='local')
all_trialsdf = []
for u in eids:
    try:
        trialsdf = bbone.load_trials_df(u, one=one)
        trialsdf['eid'] = u
        trialsdf['trial_id'] = trialsdf.index
        if trialsdf['eid'].size > MIN_BEHAV_TRIAS:
            all_trialsdf.append(trialsdf)
    except Exception as e:
        print(e)

all_trialsdf = pd.concat(all_trialsdf)
pLeft_MIN_BEHAV_TRIAS = np.vstack([all_trialsdf[(all_trialsdf.trial_id < MIN_BEHAV_TRIAS) & (all_trialsdf.eid == u)]
                                   .probabilityLeft.values
                                   for u in all_trialsdf.eid.unique()])
pLeft_MIN_BEHAV_TRIAS_uniq = np.unique(pLeft_MIN_BEHAV_TRIAS, axis=0)

if pLeft_MIN_BEHAV_TRIAS_uniq.shape[0] != 12:
    raise ValueError('these is most likely a bug in this pipeline')

template_sess = np.argmax(np.all(pLeft_MIN_BEHAV_TRIAS[None] == pLeft_MIN_BEHAV_TRIAS_uniq[:, None], axis=-1), axis=0)
all_trialsdf['template_sess'] = template_sess[np.argmax(all_trialsdf.eid.values[:, None]
                                                        == all_trialsdf.eid.unique()[None], axis=-1)]

# save imposter sessions
all_trialsdf[columns].to_parquet(DECODING_PATH.joinpath('imposterSessions.pqt'))


'''
# ?todo add eid template https://github.com/int-brain-lab/iblenv/issues/117
# ?todo change this with good performing behavioral sessions? not ephys sessions
#  get eids of behavioral sessions
one = ONE()
# task_protocol='ephysChoiceWorld' for template sessions with neural activity
eids_behav = one.search(project='ibl_neuropixel_brainwide_01',
                        task_protocol='biasedChoiceWorld',
                        )  # no template, no neural activity
for u in eids:
    det = one.get_details(u, full=True)
    if 'ephys' in det['json']['PYBPOD_BOARD']: # mice are on the ephys rig but no neural recordings
        # do stuff
        det = one.get_details(u, full=True)
        session_id = det['json']['SESSION_ORDER'][det['json']['SESSION_IDX']]
'''