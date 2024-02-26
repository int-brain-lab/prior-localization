#!/bin/bash
#SBATCH --job-name=formatting
#SBATCH --output=logs/slurm/formatting.%A.%a.out
#SBATCH --error=logs/slurm/formatting.%A.%a.err
#SBATCH --partition=shared-cpu
#SBATCH --mem=128G
#SBATCH --ntasks=1  # run one thing per job
#SBATCH --time=1:00:00

# Potentially activate env here

################################################################################################################
# ADAPT SETTINGS ABOVE THIS
################################################################################################################

out_dir=/mnt/ibl/quarantine/prior/ephys
target=pLeft  # target to fit

# Launch slurm job
python format_outputs.py "$out_dir" "$target"
