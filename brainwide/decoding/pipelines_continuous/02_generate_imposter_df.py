"""Create imposter sessions that are used for the construction of null distributions."""

from pathlib import Path
import pandas as pd

import brainbox.io.one as bbone
from one.api import ONE


GENERATE_FROM_EPHYS = True  # the number of ephys session template is too small

import sys
sys.path.append('/home/mattw/Dropbox/github/int-brain-lab/prior-localization/brainwide/decoding')
import functions.utils as dut
import functions.utils_continuous as dutc
import functions.decoding_continuous as decode
from settings.settings_continuous import fit_metadata

one = ONE()

if GENERATE_FROM_EPHYS:
    insdf = pd.read_parquet(fit_metadata['output_path'].joinpath('insertions.pqt'))
    insdf = insdf[insdf.spike_sorting != '']
    eids = insdf['eid'].unique()
else:
    eids = one.search(project='ibl_neuropixel_brainwide_01',
                      task_protocol='biasedChoiceWorld')  # no template, no neural activity

columns = [
    'probabilityLeft', 'contrastRight', 'feedbackType', 'choice', 'contrastLeft', 'eid',
    'template_sess', fit_metadata['target']
]

all_trialsdf = []
for i, u in enumerate(eids):
    try:
        det = one.get_details(u, full=True)
        print(i)
        # mice on the rig and more than 400 trials and better then 90% on highest contrasts trials
        # (BWM criteria)
        if 'ephys' in det['json']['PYBPOD_BOARD']:
            trialsdf = bbone.load_trials_df(u, one=one)
            if ((trialsdf.index.size > 400) and
                    ((trialsdf[(trialsdf.contrastLeft == 1) |
                               (trialsdf.contrastRight == 1)].feedbackType == 1).mean() > 0.9) and
                    ((trialsdf.probabilityLeft == 0.5).sum() == 90) and (
                            trialsdf.probabilityLeft.values[0] == 0.5)):
                session_id = i  # if not GENERATE_FROM_EPHYS else det['json']['SESSION_ORDER'][det['json']['SESSION_IDX']]
                trialsdf_w_var = dutc.get_target_variable_in_df(one, u, fit_metadata)
                trialsdf_w_var['eid'] = u
                trialsdf_w_var['trial_id'] = trialsdf_w_var.index
                trialsdf_w_var['template_sess'] = session_id
                all_trialsdf.append(trialsdf_w_var)
    except Exception as e:
        print(e)

all_trialsdf = pd.concat(all_trialsdf)

# save imposter sessions
save_file = fit_metadata['output_path'].joinpath('imposterSessions_%s.pqt' % fit_metadata['target'])
all_trialsdf[columns].to_parquet(save_file)
