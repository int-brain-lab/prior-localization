#!/bin/bash
#SBATCH --job-name=format_decoding
#SBATCH --output=logs/slurm/format_decoding.%A.%a.out
#SBATCH --error=logs/slurm/format_decoding.%A.%a.err
#SBATCH --partition=public-bigmem
#SBATCH --array=1-11
#SBATCH --mem=50000
#SBATCH --time=50:00:00

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

echo
python  pipelines/06_format_slurm.py $SLURM_ARRAY_TASK_ID
