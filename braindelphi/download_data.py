from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import bwm_query, load_good_units

one = ONE()
bwm_df = bwm_query(one)

# Download trials data
for eid in bwm_df['eid'].unique():
    print(f'Downloading trials for {eid}')
    sess_loader = SessionLoader(one, eid)
    sess_loader.load_trials()

# Download ephys data
for pid in bwm_df['pid']:
    print(f'Downloading spike sorting for {pid}')
    spikes, clusters = load_good_units(one, pid)
