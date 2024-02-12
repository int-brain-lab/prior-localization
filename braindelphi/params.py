import os
from pathlib import Path

username = os.environ["USER"]  # os.getlogin()
if username == 'mattw':
    out_dir = Path('/media/mattw/ibl/')
elif username == 'mw3323':
    out_dir = Path('/home/mw3323/ibl/')
elif username == 'findling':
    out_dir = Path('/home/users/f/findling/scratch/ibl/prior-localization/braindelphi')
elif username == 'csmfindling':
    out_dir = Path('/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/braindelphi')
elif username == 'hubertf':
    out_dir = Path('/home/users/h/hubertf/scratch/')

# widefield imaging path
if username in ['findling', 'hubertf']:
    WIDE_FIELD_PATH = Path('/home/share/pouget_lab/wide_field_imaging/')
else:
    WIDE_FIELD_PATH = Path('wide_field_imaging/')

# path to user-specific settings file
SETTINGS_PATH = out_dir.joinpath('decoding', 'settings.yaml')

# store cached data for simpler loading
CACHE_PATH = out_dir.joinpath('cache')
CACHE_PATH = Path('/home/share/pouget_lab/cache_final/')

# store neural decoding models
FIT_PATH = out_dir.joinpath('decoding', 'results', 'neural')

# store behavioral models
BEH_MOD_PATH = out_dir.joinpath('decoding', 'results', 'behavioral')

# store imposter session data used for creating null distributions
IMPOSTER_SESSION_PATH = out_dir.joinpath('decoding')

# store imposter session data used for creating null distributions
INTER_INDIVIDUAL_PATH = out_dir.joinpath('decoding')

