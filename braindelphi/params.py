import os
from pathlib import Path
try:
    if os.getlogin() == 'mattw':
        braindelphi_PATH = Path('/media/mattw/ibl/')
    else:
        braindelphi_PATH = Path(
            '/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/braindelphi')
except OSError:
    braindelphi_PATH = Path(
        '/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/braindelphi')
    pass

CACHE_PATH = braindelphi_PATH.joinpath('cache')
FIT_PATH = braindelphi_PATH.joinpath('decoding', 'results', 'neural')
BEH_MOD_PATH = braindelphi_PATH.joinpath('decoding', 'results', 'behavior')
IMPOSTER_SESSION_PATH = braindelphi_PATH.joinpath('decoding')
CACHE_PATH.mkdir(parents=True, exist_ok=True)
FIT_PATH.mkdir(parents=True, exist_ok=True)
BEH_MOD_PATH.mkdir(parents=True, exist_ok=True)
