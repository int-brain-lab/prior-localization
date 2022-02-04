"""
Downloads trials, wheel, spikes and clusters for the BWM
We compute clusters locations and metrics within the clusters objects and save it as a parquet table
The output is a pandas dataframe of insertions, containing eids, pids, subjects, and the location of
the cached datasets on disk for future loading
"""
from json import JSONDecodeError
from pathlib import Path
from one.api import ONE
from ibllib.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader

import decoding_utils as dut
from decode_prior import SESS_CRITERION

try:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, LocalCluster
except:
    import warnings

    warnings.warn('dask import failed')
    pass

DECODING_PATH = Path('C:/Users/fphub\Documents/int-brain-lab/prior-localization/decoding')
# DECODING_PATH = Path("/home/users/h/hubertf/int-brain-lab/prior-localization/decoding")
# DECODING_PATH = Path("/home/users/f/findling/ibl/prior-localization/decoding")

one = ONE()
ba = AllenAtlas()

# Generate cluster interface and map eids to workers via dask.distributed.Client
insdf = dut.query_sessions(selection=SESS_CRITERION, one=one)

excludes = [
    'd8c7d3f2-f8e7-451d-bca1-7800f4ef52ed',  # key error in loading histology from json
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',  # same same
    'c2184312-2421-492b-bbee-e8c8e982e49e',  # same same
    '58b271d5-f728-4de8-b2ae-51908931247c',  # same same
    'f86e9571-63ff-4116-9c40-aa44d57d2da9',  # 404 not found
    '1a276285-8b0e-4cc9-9f0a-a3a002978724'
]

insdf['spike_sorting'] = ''
insdf['session_path'] = ''
insdf['histology'] = ''

errors_pid = []
IMIN = 0

###### if I only care about V1 sessions #######
# get all eid with region corresponding to primary visual cortex
df = pd.read_parquet('C:/Users/fphub/Documents/int-brain-lab/decoding_results/decoding_results2022-01-10_decode_signcont_task_Lasso_align_goCue_times_2_pseudosessions_timeWindow_0_0_1.parquet')# '/home/users/h/hubertf/int-brain-lab/decoding_results/2022-01-11_decode_prior_expSmoothingPrevActions_Lasso_align_goCue_times_2_pseudosessions_timeWindow_0_0_1.parquet')
df_sess_reg = df[df['fold'] == -1]
region = 'VISp'
V1_mask = df_sess_reg.index.get_level_values(3).str.startswith((region))
df_sess_reg_V1 = df_sess_reg[V1_mask]
V1_eids = df_sess_reg_V1.index.unique('eid')
print("number of sessions with primary visual cortex recordings : ",len(V1_eids))
################################################

for i, rec in insdf.iterrows():
    pid = rec['pid']
    eid = rec['eid']
    if pid in excludes:
        continue
    if eid not in V1_eids:
        continue
    else:
        print(i)
    if i < IMIN:
        continue
    print(i, pid)
    try:
        ssl = SpikeSortingLoader(pid, one=one, atlas=ba)
        spikes, clusters, channels = ssl.load_spike_sorting()
        # this will cache both the metrics if they don't exist or don't match and also write a clusters.pqt dataframe
        if not ssl.spike_sorting_path.joinpath('clusters.pqt').exists():
            SpikeSortingLoader.merge_clusters(spikes, clusters, channels, cache_dir=ssl.spike_sorting_path)
        # saves the spike sorting path into the dataframe fossl.load_spike_sorting()r future loading, as well as the histology source
        insdf['histology'][i] = ssl.histology
        insdf['session_path'][i] = str(ssl.session_path)
        insdf['spike_sorting'][i] = ssl.collection
        one.load_object(ssl.eid, 'trials', collection='alf', download_only=True)
        one.load_object(ssl.eid, 'wheel', collection='alf', download_only=True)
    except Exception as e:
        print(e)
        errors_pid.append(rec.eid)
        pass


insdf.to_parquet(DECODING_PATH.joinpath('insertions_V1.pqt'))

print('\n')
print('errors')
print(errors_pid)