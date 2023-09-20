import argparse

from one.api import ONE
from brainwidemap.bwm_loading import bwm_query
from prior_localization.fit_data import fit_session_ephys


# Parse input arguments
parser = argparse.ArgumentParser(description='Run decoding')
parser.add_argument('output_dir')
parser.add_argument('session_idx')
parser.add_argument('pseudo_ids')
args = parser.parse_args()
output_dir = str(args.output_dir)
session_idx = int(args.session_idx)
pseudo_ids = list(args.pseudo_id)

# Create an offline ONE instance, we don't want to hit the database when running so many jobs in parallel and have
# downloaded the data before
one = ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local')

# Get info for respective eid from bwm_dataframe
bwm_df = bwm_query(one=one, freeze='2022_10_initial')
session_id = bwm_df.eid.unique()[session_idx]
subject = bwm_df[bwm_df.eid == session_id].subject.unique()[0]
# We are merging probes per session, therefore using a list of all probe names of a session as input
probe_name = list(bwm_df[bwm_df.eid == session_id].probe_name)

# Run the decoding for the current set of pseudo ids.
results = fit_session_ephys(
        one, session_id, subject, probe_name, output_dir=output_dir, pseudo_ids=pseudo_ids, target='pLeft',
        align_event='stimOn_times', time_window=(-0.6, -0.1), model='optBay', n_runs=10, compute_neurometrics=False,
        motor_residuals=False
)
