#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=logs/slurm/decoding.%A.%a.out
#SBATCH --error=logs/slurm/decoding.%A.%a.err
#SBATCH --partition=public-bigmem
#SBATCH --array=1-1:1
#SBATCH --mem=50000
#SBATCH --time=24:00:00

# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
# change to conda  => which python
~/.conda/envs/iblenv/bin/python  pipelines/06_download_bwm.py
