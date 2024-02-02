import argparse

from one.api import ONE
from brainwidemap.bwm_loading import bwm_query
from prior_localization.fit_data import fit_session_ephys


# Parse input arguments
parser = argparse.ArgumentParser(description='Run decoding')
parser.add_argument('output_dir')
parser.add_argument('session_idx')
parser.add_argument('pseudo_ids', nargs='+')
parser.add_argument('target')
args = parser.parse_args()
output_dir = str(args.output_dir)
session_idx = int(args.session_idx)
pseudo_ids = [int(p) for p in args.pseudo_id]
target = str(args.target)

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

# set BWM defaults here
binsize = None
n_bins_lag = None
n_runs = 10

if target == 'stimside':
    align_event = 'stimOn_times'
    time_window = (0.0, 0.1)
    model = 'oracle'
    estimator = 'LogisticRegression'

elif target == 'signcont':
    align_event = 'stimOn_times'
    time_window = (0.0, 0.1)
    model = 'oracle'
    estimator = 'Lasso'

elif target == 'choice':
    align_event = 'firstMovement_times'
    time_window = (-0.1, 0.0)
    model = 'actKernel'
    estimator = 'LogisticRegression'

elif target == 'feedback':
    align_event = 'feedback_times'
    time_window = (0.0, 0.2)
    model = 'actKernel'
    estimator = 'LogisticRegression'

elif target == 'pLeft':
    align_event = 'stimOn_times'
    time_window = (-0.6, -0.1)
    model = 'optBay'
    estimator = 'LogisticRegression'

elif target in ['wheel-speed', 'wheel-velocity']:
    align_event = 'firstMovement_times'
    time_window = (-0.2, 1.0)
    model = 'oracle'
    estimator = 'Lasso'
    binsize = 0.02
    n_bins_lag = 10
    n_runs = 2

else:
    raise ValueError(f'{target} is an invalid target value')

# Run the decoding for the current set of pseudo ids.
results = fit_session_ephys(
    one, session_id, subject, probe_name, output_dir=output_dir, pseudo_ids=pseudo_ids, target=target,
    align_event=align_event, time_window=time_window, model=model, n_runs=n_runs, estimator=estimator,
    compute_neurometrics=False, motor_residuals=False,
)
