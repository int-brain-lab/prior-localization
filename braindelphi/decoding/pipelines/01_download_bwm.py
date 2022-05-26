"""
Downloads trials, wheel, spikes and clusters for the BWM
We compute clusters locations and metrics within the clusters objects and save it as a parquet table
The output is a pandas dataframe of insertions, containing eids, probes, subjects, and the location of
the cached datasets on disk for future loading
"""
from pathlib import Path
from one.api import ONE
from ibllib.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
import tracemalloc
tracemalloc.start()
from functions import utils as dut
from settings import SESS_CRITERION

#DECODING_PATH = Path("/Users/csmfindling/Documents/Postdoc-Geneva/IBL/behavior/prior-localization/decoding")
DECODING_PATH = Path("/home/users/f/findling/scratch")
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
for i, rec in insdf.iterrows():
    pid = rec['pid']
    if pid in excludes:
        continue
    if i < IMIN:
        continue
    print(i, pid)
    try:
        ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
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
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    except Exception as e:
        print(e)
        errors_pid.append(rec.eid)
        pass

insdf.to_parquet(DECODING_PATH.joinpath('insertions.pqt'))
tracemalloc.stop()

print('\n')
print('errors')
print(errors_pid)