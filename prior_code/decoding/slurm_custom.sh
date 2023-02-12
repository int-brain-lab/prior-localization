#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=logs/slurm/decoding.%A.%a.out
#SBATCH --error=logs/slurm/decoding.%A.%a.err
#SBATCH --partition=public-bigmem,shared-bigmem
#SBATCH --array=1-1:1
#SBATCH --mem=120000
#SBATCH --time=12:00:00

#source /home/users/f/findling/.bash_profile
#mamba activate iblenv

# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
# change to conda  => which python
python  pipelines/06_format_slurm.py
