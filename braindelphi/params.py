from pathlib import Path
braindelphi_PATH = Path('/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/braindelphi')
CACHE_PATH = braindelphi_PATH.joinpath('cache')
FIT_PATH = braindelphi_PATH.joinpath('decoding', 'results', 'neural')
BEH_MOD_PATH = braindelphi_PATH.joinpath('decoding', 'results', 'behavior')
IMPOSTER_SESSION_PATH = braindelphi_PATH.joinpath('decoding')
CACHE_PATH.mkdir(parents=True, exist_ok=True)
FIT_PATH.mkdir(parents=True, exist_ok=True)
BEH_MOD_PATH.mkdir(parents=True, exist_ok=True)
