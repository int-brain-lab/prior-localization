#!/bin/bash
#SBATCH --job-name=format_decoding
#SBATCH --output=logs/slurm/format_decoding.%A.%a.out
#SBATCH --error=logs/slurm/format_decoding.%A.%a.err
#SBATCH --partition=public-bigmem
#SBATCH --array=1-1:1
#SBATCH --mem=50000
#SBATCH --time=50:00:00

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
# change to conda  => which python
python  pipelines/06_format_slurm.py
