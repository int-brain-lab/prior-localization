import os
from pathlib import Path

#out_dir = Path(os.environ["PRIOR_DECODING_OUT_DIRECTORY"])
out_dir = Path('/home/julia/data/prior_review')

# widefield imaging path
WIDE_FIELD_PATH = out_dir.joinpath('wide_field_imaging/')

# path to user-specific settings file
SETTINGS_PATH = out_dir.joinpath('decoding', 'settings.yaml')

# store cached data for simpler loading
CACHE_PATH = out_dir.joinpath('cache')

# store neural decoding models
FIT_PATH = out_dir.joinpath('decoding', 'results', 'neural')

# store behavioral models
BEH_MOD_PATH = out_dir.joinpath('decoding', 'results', 'behavioral')

# store imposter session data used for creating null distributions
IMPOSTER_SESSION_PATH = out_dir.joinpath('decoding')

# store imposter session data used for creating null distributions
INTER_INDIVIDUAL_PATH = out_dir.joinpath('decoding')

