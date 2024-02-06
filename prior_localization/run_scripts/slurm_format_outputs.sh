#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=logs/slurm/decoding.%A.%a.out
#SBATCH --error=logs/slurm/decoding.%A.%a.err
#SBATCH --partition=public-bigmem
#SBATCH --array=1-1:1
#SBATCH --mem=120000
#SBATCH --time=72:00:00

source /home/users/f/findling/.bash_profile
conda activate iblenv2

# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
# change to conda  => which python
#python pipelines/01_cache_ephys_slurm.py
#python pipelines/11_format_slurm_batched.py
python  format_slurm_outputs.py
