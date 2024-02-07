#!/bin/bash
#SBATCH --account=stats
#SBATCH -c 8
#SBATCH --job-name=decoding
#SBATCH --output=/moto/stats/users/mw3323/logs/slurm/decoding.%A.%a.out
#SBATCH --error=/moto/stats/users/mw3323/logs/slurm/decoding.%A.%a.err
#SBATCH --array=1-51
#SBATCH --mem=32GB
#SBATCH --time=1:00:00

module load anaconda

out_dir=/moto/stats/users/m23323/results
n_sessions=354  # number of unique eids in the dataframe that we will index in
n_pseudo=200  # number of pseudo sessions to generate for each real session
n_per_job=10  # number of (pseudo)sessions to fit per job on the cluster
target=stimside  # target to fit

# Potentially activate env here

################################################################################################################
# ADAPT SETTINGS ABOVE THIS
################################################################################################################

# Make sure output dir exists
mkdir -p $out_dir

# Get the total number of jobs
n_jobs=n_jobs=$(($n_sessions * $n_pseudo / $n_per_job))

# Launch slurm job array
sbatch --array=1-$n_jobs python .run_ephys_decoding "$SLURM_ARRAY_TASK_ID" "$n_pseudo" "$n_per_job" "$out_dir" "$target"