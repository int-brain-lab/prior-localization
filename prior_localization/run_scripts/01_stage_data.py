from pathlib import Path
from one.api import ONE
from brainwidemap.bwm_loading import bwm_query

from prior_localization.fit_data import fit_session_ephys

#### USER SETTINGS, ADAPT THESE TO YOUR CLUSTER ###
input_dir = Path.home().joinpath('prior', 'inputs')
###################################################

# Here we use and online ONE instance because we need to actually access the database
# We set it up to download to our chosen input_dir
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', cache_dir=input_dir, silent=True)
one = ONE(base_url='https://openalyx.internationalbrainlab.org')

# Get brainwidemap dataframe
bwm_df = bwm_query(one=one) #, freeze='2022_10_initial')

# Loop over all sessions to stage ephys data
for i, session_id in enumerate(bwm_df.eid.unique()):
    print(f'Staging data for session {i}/{len(bwm_df.eid.unique())}:{session_id}')
    subject = bwm_df[bwm_df.eid == session_id].subject.unique()[0]
    # We are merging probes per session, therefore using a list of all probe names of a session as input
    probe_name = list(bwm_df[bwm_df.eid == session_id].probe_name)
    # Run the fit session function with stage only flag to download all needed data, but not perform the decoding
    # No data will be output, but since an output_dir is required, we just use the input dir as a front
    _ = fit_session_ephys(one, session_id, subject, probe_name, output_dir=input_dir, stage_only=True)
