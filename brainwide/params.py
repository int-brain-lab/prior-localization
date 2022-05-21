from pathlib import Path
BRAINWIDE_PATH = Path.cwd()
CACHE_PATH = BRAINWIDE_PATH.joinpath('decoding', 'cache')
FIT_PATH = BRAINWIDE_PATH.joinpath('decoding', 'results', 'neural')
BEH_MOD_PATH = BRAINWIDE_PATH.joinpath('decoding', 'results', 'behavior')
CACHE_PATH.mkdir(parents=True, exist_ok=True)
FIT_PATH.mkdir(parents=True, exist_ok=True)
BEH_MOD_PATH.mkdir(parents=True, exist_ok=True)