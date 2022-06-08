#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=logs/slurm/decoding.%A.%a.out
#SBATCH --error=logs/slurm/decoding.%A.%a.err
#SBATCH --partition=shared-cpu
#SBATCH --array=1-5
#SBATCH --mem=7000
#SBATCH --time=12:00:00

source /home/users/f/findling/.bash_profile
mamba activate iblenv
# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
python pipelines/05_slurm_decode.py $SLURM_ARRAY_TASK_ID
