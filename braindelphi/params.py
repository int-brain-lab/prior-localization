import os
from pathlib import Path

username = os.environ["USER"]  # os.getlogin()
if username == 'mattw':
    braindelphi_PATH = Path('/media/mattw/ibl/')
elif username == 'findling':
    braindelphi_PATH = Path('/home/users/f/findling/scratch/ibl/prior-localization/braindelphi')
elif username == 'root':
    braindelphi_PATH = Path('/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/braindelphi')

# path to user-specific settings file
SETTINGS_PATH = braindelphi_PATH.joinpath('decoding', 'settings.yaml')

# store cached data for simpler loading
CACHE_PATH = braindelphi_PATH.joinpath('cache')
CACHE_PATH.mkdir(parents=True, exist_ok=True)

# store neural decoding models
FIT_PATH = braindelphi_PATH.joinpath('decoding', 'results', 'neural')
FIT_PATH.mkdir(parents=True, exist_ok=True)

# store behavioral models
BEH_MOD_PATH = braindelphi_PATH.joinpath('decoding', 'results', 'behavioral')
BEH_MOD_PATH.mkdir(parents=True, exist_ok=True)

# store imposter session data used for creating null distributions
IMPOSTER_SESSION_PATH = braindelphi_PATH.joinpath('decoding')

# widefield imaging path
if os.getlogin() in ['findling', 'hubert']:
    WIDE_FIELD_PATH = Path('/home/share/pouget_lab/wide_field_imaging/')
else:
    WIDE_FIELD_PATH = Path('wide_field_imaging/')
