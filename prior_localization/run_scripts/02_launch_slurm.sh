#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=logs/slurm/decoding.%A.%a.out
#SBATCH --error=logs/slurm/decoding.%A.%a.err
#SBATCH --partition=shared-cpu
#SBATCH --mem=128G
#SBATCH --ntasks=1  # run one thing per job
#SBATCH --time=12:00:00

out_dir=/mnt/ibl/quarantine/prior/ephys
n_sessions=354  # number of unique eids in the dataframe that we will index in
n_pseudo=200  # number of pseudo sessions to generate for each real session
n_per_job=10  # number of (pseudo)sessions to fit per job on the cluster
base_idx=0    # add this to the task id; some clusters don't allow array values >1000
target=pLeft  # target to fit

# Potentially activate env here

################################################################################################################
# ADAPT SETTINGS ABOVE THIS
################################################################################################################

# Make sure output dir exists
mkdir -p $out_dir

# After the job is submitted, run the python script
python run_ephys_decoding.py "$SLURM_ARRAY_TASK_ID" "$n_pseudo" "$n_per_job" "$base_idx" "$out_dir" "$target"
