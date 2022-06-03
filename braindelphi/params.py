import os
from pathlib import Path

if os.getlogin() == 'mattw':
    braindelphi_PATH = Path('/media/mattw/ibl/')
elif os.getlogin() == 'mw3323':
    braindelphi_PATH = Path('/home/mw3323/ibl/')
else:
    braindelphi_PATH = Path(
        '/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/braindelphi')

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
