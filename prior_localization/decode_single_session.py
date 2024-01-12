from pathlib import Path
from one.api import ONE
from prior_localization.functions.decoding import fit_session_ephys
import tempfile

# Instantiate ONE to connect to the public IBL database
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international')

# UUID of an example session
session_id = '56956777-dca5-468c-87cb-78150432cc57'

# Get some other required information through the ONE API, like subject nickname and probe names
subject = one.eid2ref(session_id)['subject']
probe_names = one.eid2pid(session_id)[1]

# Create a temporary directory for the outputs (you can replace this with permanent path on your disk)
output_dir = Path(tempfile.TemporaryDirectory().name)

# For this session, we decode the Bayes Optimal prior from the neural activity. More specifically, we identify
# the regions along the probe(s) probe_names, and perform a region-level decoding (as defined by the
# "regions" argument).
# For each region, we decode from all units the variable "target" (pLeft here - so the prior that the
# stimulus will appear on the left side) from the model "model" (optimal Bayesian here) using Lasso linear
# regression (see Methods of the paper for more information).
# Pseudo_ids gives the information of whether the decoding of the session's Bayes optimal prior is decoded
# (pseudo_id=-1) or whether we decode a counterfactual Bayes optimal prior, from an unobserved pseudo-session
# (pseudo_id > 0). pseudo_ids=[-1, 1, 2] indicates that we decode the session's prior as well as
# counterfactual prior from 2 pseudo-sessions (in the paper, we actually decode priors from 200
# pseudo-sessions).
# Neural activity is defined by a time window (corresponding to the "time_window" argument, here -600 to
# -100ms) aligned to a particular event (defined by the "stimOn_times" argument, here stimulus onset)
# The function outputs, for each region and for each pseudo_id, the path file of the decoding results.
results_fit_session = fit_session_ephys(
    one, session_id, subject, probe_names, model='optBay', pseudo_ids=[-1, 1, 2], target='pLeft',
    align_event='stimOn_times', time_window=(-0.6, -0.1), output_dir=output_dir, regions='single_regions'
)

