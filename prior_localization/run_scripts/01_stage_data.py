from pathlib import Path
from one.api import ONE
from brainwidemap.bwm_loading import bwm_query

from prior_localization.fit_data import fit_session_ephys

#### USER SETTINGS, ADAPT THESE TO YOUR CLUSTER ###
input_dir = Path('/home/share/pouget_lab/').joinpath('prior', 'inputs')
###################################################

# Here we use and online ONE instance because we need to actually access the database
# We set it up to download to our chosen input_dir)
#ONE.setup(base_url='https://openalyx.internationalbrainlab.org', cache_dir=Path('/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/ONE_cache'))
one = ONE()
one.alyx.clear_rest_cache()

# Get brainwidemap dataframe
bwm_df = bwm_query() #, freeze='2022_10_initial')

# Loop over all sessions to stage ephys data
failed = [] # ['a92c4b1d-46bd-457e-a1f4-414265f0e2d4']
for i, session_id in enumerate(bwm_df.eid.unique()):
    try:
        print(f'Staging data for session {i}/{len(bwm_df.eid.unique())}:{session_id}')
        subject = bwm_df[bwm_df.eid == session_id].subject.unique()[0]
        # We are merging probes per session, therefore using a list of all probe names of a session as input
        probe_name = list(bwm_df[bwm_df.eid == session_id].probe_name)
        probe_name = probe_name[0] if len(probe_name) == 1 else probe_name
        # Run the fit session function with stage only flag to download all needed data, but not perform the decoding
        # No data will be output, but since an output_dir is required, we just use the input dir as a front
        _ = fit_session_ephys(one, session_id, subject, probe_name, output_dir=input_dir, stage_only=True)
    except:
        print('failed')
        failed.append(session_id)
        pass
