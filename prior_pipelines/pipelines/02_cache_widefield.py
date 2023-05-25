# Standard library
import logging
import pickle
from datetime import datetime as dt
from pathlib import Path
import numpy as np

# Third party libraries
import pandas as pd
import glob
from tqdm import tqdm

# IBL libraries
from prior_pipelines.params import CACHE_PATH, WIDE_FIELD_PATH
from prior_pipelines.wide_field_imaging.wfi_utils import _load_wfi_session as load_wfi_session
CACHE_PATH.mkdir(parents=True, exist_ok=True)
_logger = logging.getLogger('prior_pipelines')

DATE = str(dt.today())
HEMISPHERES = ['left','right'] # 'left', 'right'
if np.any([k not in ['left', 'right'] for k in HEMISPHERES]):
    raise ValueError('Hemispheres must be left or right')
hemisphere_specif = 'both_hemispheres' if np.unique(HEMISPHERES).size > 1 else HEMISPHERES[0]

subjects = glob.glob(WIDE_FIELD_PATH.joinpath('CSK-im-*').as_posix())
nb_eids_per_subject = np.array([np.load(s + '/behavior.npy', allow_pickle=True).size for s in subjects])

dataset_files = []
for index in tqdm(range(nb_eids_per_subject.sum())):
    subj_id = np.sum(index >= nb_eids_per_subject.cumsum())
    sess_id = index - np.hstack((0, nb_eids_per_subject)).cumsum()[:-1][subj_id]
    if sess_id < 0:
        raise ValueError('There is an error in the code')
    sessiondf, wideFieldImaging_dict, metadata = load_wfi_session(subjects[subj_id], sess_id, HEMISPHERES, WIDE_FIELD_PATH.as_posix())

    sesspath = Path(CACHE_PATH).joinpath('widefield').joinpath(metadata['subject']).joinpath(metadata['eid']).joinpath(hemisphere_specif)
    sesspath.mkdir(parents=True, exist_ok=True)
    fnbase = str(dt.now().date())
    metadata_fn = sesspath.joinpath(fnbase + '_widefield_metadata.pkl')
    data_fn = sesspath.joinpath(fnbase + '_widefield_regressors.pkl')
    metadata = {
        **metadata,
        'hemispheres': hemisphere_specif,
    }
    # If you've reached here, there's no matching file
    with open(metadata_fn, 'wb') as fw:
        pickle.dump(metadata, fw)
    with open(data_fn, 'wb') as fw:
        pickle.dump([sessiondf, wideFieldImaging_dict], fw)

    dataset_files.append([metadata['subject'], metadata['eid'], metadata['hemispheres'], metadata_fn, data_fn])

dataset = [{
    'subject': x[0],
    'eid': x[1],
    'hemisphere': x[2],
    'meta_file': x[3],
    'reg_file': x[4],
} for x in dataset_files]
dataset = pd.DataFrame(dataset)

outdict = {'params': {'hemisphere_specif': hemisphere_specif}, 'dataset_filenames': dataset}
with open(Path(CACHE_PATH).joinpath(DATE + '_widefield_metadata.pkl'), 'wb') as fw:
    pickle.dump(outdict, fw)
