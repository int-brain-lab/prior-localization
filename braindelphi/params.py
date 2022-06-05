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

# path to user-specific settings file
SETTINGS_PATH = braindelphi_PATH.joinpath('decoding', 'settings.yaml')

# store cached data for simpler loading
CACHE_PATH = braindelphi_PATH.joinpath('cache')
CACHE_PATH.mkdir(parents=True, exist_ok=True)

# store neural decoding models
FIT_PATH = braindelphi_PATH.joinpath('decoding', 'results', 'neural')
FIT_PATH.mkdir(parents=True, exist_ok=True)

# store behavioral models
BEH_MOD_PATH = braindelphi_PATH.joinpath('decoding', 'results', 'behavior')
BEH_MOD_PATH.mkdir(parents=True, exist_ok=True)

# store imposter session data used for creating null distributions
IMPOSTER_SESSION_PATH = braindelphi_PATH.joinpath('decoding')
