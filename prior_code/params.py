import os
from pathlib import Path

out_dir = Path(
    "/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/code"
)

WIDE_FIELD_PATH = Path(
    "/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/prior_code/wide_field_imaging"
)

# store cached data for simpler loading
CACHE_PATH = out_dir.joinpath("cache")

# store neural decoding models
FIT_PATH = out_dir.joinpath("decoding", "results", "neural")

# store behavioral models
BEH_MOD_PATH = out_dir.joinpath("decoding", "results", "behavioral")

# store imposter session data used for creating null distributions
IMPOSTER_SESSION_PATH = out_dir.joinpath("decoding")

# store imposter session data used for creating null distributions
INTER_INDIVIDUAL_PATH = out_dir.joinpath("decoding")
