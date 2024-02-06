import argparse
import numpy as np

from one.api import ONE
from brainwidemap.bwm_loading import bwm_query
from prior_localization.fit_data import fit_session_ephys

# Parse input arguments
parser = argparse.ArgumentParser(description='Run decoding')
parser.add_argument('job_idx')
parser.add_argument('n_pseudo')
parser.add_argument('n_per_job')
parser.add_argument('output_dir')
parser.add_argument('target')

args = parser.parse_args()
job_idx = int(args.job_idx)
n_pseudo = int(args.n_pseudo)
n_per_job = int(args.n_per_job)
output_dir = str(args.output_dir)
target = str(args.output_dir)

# TODO: settings for the particular target

# Get session idx
session_idx = int(np.ceil(job_idx / (n_pseudo / n_per_job)) - 1)

# Set of pseudo sessions for one session, first session is always the real one, indicated by -1
all_pseudo = list(range(n_pseudo))
all_pseudo[0] = -1
# Select relevant pseudo sessions for this job
pseudo_idx = int((job_idx - 1) % (n_pseudo / n_per_job) * n_per_job)
pseudo_ids = all_pseudo[pseudo_idx:pseudo_idx+n_per_job]

# Create an offline ONE instance, we don't want to hit the database when running so many jobs in parallel and have
# downloaded the data before
one = ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local')

# Get info for respective eid from bwm_dataframe
bwm_df = bwm_query(one=one, freeze='2023_12_bwm_release')
session_id = bwm_df.eid.unique()[session_idx]
subject = bwm_df[bwm_df.eid == session_id].subject.unique()[0]
# We are merging probes per session, therefore using a list of all probe names of a session as input
probe_name = list(bwm_df[bwm_df.eid == session_id].probe_name)
probe_name = probe_name[0] if len(probe_name) == 1 else probe_name

# Run the decoding for the current set of pseudo ids.
results = fit_session_ephys(
        one, session_id, subject, probe_name, output_dir=output_dir, pseudo_ids=pseudo_ids, target=target,
        align_event='stimOn_times', time_window=(-0.6, -0.1), model='optBay', n_runs=10, compute_neurometrics=False,
        motor_residuals=False
)
