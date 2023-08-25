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

# For this session, ...
results_fit_session = fit_session_ephys(one, session_id, subject, probe_names, model='optBay', pseudo_ids=[-1, 1, 2],
                                        target='pLeft', align_event='stimOn_times', time_window=(-0.6, -0.1),
                                        output_dir=output_dir, regions='single_regions'
                                        )
