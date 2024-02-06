from one.api import ONE
from brainwidemap.bwm_loading import bwm_query
from prior_localization.fit_data import fit_session_ephys
import sys
import numpy as np
from pathlib import Path

# Parse input arguments
try:
    index = int(sys.argv[1]) - 1
except:
    index = 129
    pass


N_PSEUDO_PER_JOB = 10
output_dir = "/home/users/f/findling/prior-public-code/prior_localization/results"

# Create an offline ONE instance, we don't want to hit the database when running so many jobs in parallel and have
# downloaded the data before
one = ONE(mode='local') #base_url='https://openalyx.internationalbrainlab.org', mode='local')

# Get info for respective eid from bwm_dataframe
bwm_df = bwm_query()  #, freeze='2022_10_initial')

nb_uniq_eids = bwm_df.eid.unique().size

session_idx = index % nb_uniq_eids
job_id = index // nb_uniq_eids

session_id = bwm_df.eid.unique()[session_idx]
subject = bwm_df[bwm_df.eid == session_id].subject.unique()[0]
# We are merging probes per session, therefore using a list of all probe names of a session as input
probe_name = list(bwm_df[bwm_df.eid == session_id].probe_name)
probe_name = probe_name[0] if len(probe_name) == 1 else probe_name

pseudo_ids = (
        np.arange(job_id * N_PSEUDO_PER_JOB, (job_id + 1) * N_PSEUDO_PER_JOB) + 1
)

if 1 in pseudo_ids:
    pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype("int64")

# Run the decoding for the current set of pseudo ids.
results = fit_session_ephys(
        one, session_id, subject, probe_name, output_dir=output_dir, pseudo_ids=pseudo_ids, target=target,
        align_event='stimOn_times', time_window=(-0.6, -0.1), model='optBay', n_runs=10, compute_neurometrics=False,
        motor_residuals=False
)
