#!/bin/bash
#SBATCH --account=stats
#SBATCH -c 8
#SBATCH --job-name=formatting
#SBATCH --output=/moto/stats/users/mw3323/logs/slurm/formatting.%A.%a.out
#SBATCH --error=/moto/stats/users/mw3323/logs/slurm/formatting.%A.%a.err
#SBATCH --mem=128GB
#SBATCH --time=2:00:00

module load anaconda

################################################################################################################
# ADAPT SETTINGS ABOVE THIS
################################################################################################################

out_dir=/moto/stats/users/m23323/results
target=stimside  # target to fit

# Launch slurm job
sbatch python .format_outputs "$out_dir" "$target"
