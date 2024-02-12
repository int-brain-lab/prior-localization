#!/bin/bash
#SBATCH --job-name=format_decoding
#SBATCH --output=logs/slurm/format_decoding.%A.%a.out
#SBATCH --error=logs/slurm/format_decoding.%A.%a.err
#SBATCH --partition=public-cpu,public-bigmem
#SBATCH --array=1-1:1
#SBATCH --mem=10000
#SBATCH --time=50:00:00

source /home/users/f/findling/.bash_profile
mamba activate iblenv

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

echo
python pipelines/01_cache_ephys_slurm.py

#python  pipelines/06_format_slurm.py $SLURM_ARRAY_TASK_ID
