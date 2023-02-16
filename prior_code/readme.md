## Code documentation

This code was used to generate the analysis of ... by ...

#### Steps to setup the code:

- create an environment and install the packages from requirements.txt
- `conda develop` the folder in which prior_code is
- modify the `out_dir` in params.py. `out_dir` is the path where cached files will be saved (see next paragraph) 
- and `WIDE_FIELD_PATH` in params.py. `WIDE_FIELD_PATH` is the path where the widefield data is saved. 

#### Steps to run the data download and format code:

Before running the decoding analysis, you will have to download and transform the data in expected format.
These transformations are defined in the `caching_pipelines` folder. You will find three pipelines in the 
`cachine_pipelines` folder. 
- `01_cache_ephys.py` caches the electrophysiology sessions and its associated behavior (choice, feedback, ...)
- `02_cache_widefield.py` caches the widefield sessions and its associated behavior (choice, feedback, ...)
- `03_cache_motor.py` caches the deep lab cut (DLC) data. These sessions are the same ones as in the `ephys` caching.
two different pipelines were built because the DLC data is not systematically used.

Running these pipelines will generate cache files in the `out_dir` file. You should count a few hours for 
each cache generation.

#### Steps to run the decoding code:

You will find in the `decoding/pipelines` the necessary functions to run the decoding script. There
are three scripts:
- `local_decoding.py` this script enables you to run the decoding on your local machine. This script
serves more as an example toy script as running all the decoding locally would take weeks if not months
- `slurm_decoding_one_session.py` this script is called to run the decoding on one session on a slurm cluster.
This script goes in pair with the `slurm_run_decoding_on_each_session.sh` in the decoding folder. This pipeline
is used to launch the decoding on each session through slurm
- `slurm_pull_decoding_results.py`. this script enable you to pull the decoding results across
sessions that we obtained by launching the `slurm_run_decoding_on_each_session.sh` slurm job

