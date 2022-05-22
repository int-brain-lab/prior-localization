from pathlib import Path
braindelphi_PATH = Path.cwd()
CACHE_PATH = braindelphi_PATH.joinpath('decoding', 'cache')
FIT_PATH = braindelphi_PATH.joinpath('decoding', 'results', 'neural')
BEH_MOD_PATH = braindelphi_PATH.joinpath('decoding', 'results', 'behavior')
CACHE_PATH.mkdir(parents=True, exist_ok=True)
FIT_PATH.mkdir(parents=True, exist_ok=True)
BEH_MOD_PATH.mkdir(parents=True, exist_ok=True)