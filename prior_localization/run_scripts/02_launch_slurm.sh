#!/bin/bash
#SBATCH --account=stats
#SBATCH -c 8
#SBATCH --job-name=decoding
#SBATCH --output=/moto/stats/users/mw3323/logs/slurm/decoding.%A.%a.out
#SBATCH --error=/moto/stats/users/mw3323/logs/slurm/decoding.%A.%a.err
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --array=1-4

# --array=1-708

module load anaconda

out_dir=/moto/stats/users/mw3323/results
n_sessions=354  # number of unique eids in the dataframe that we will index in
n_pseudo=5  # 200 # number of pseudo sessions to generate for each real session
n_per_job=5  # 50  # number of (pseudo)sessions to fit per job on the cluster
base_idx=0  #708  # add this to the task id; terremoto doesn't allow array values >1000

target=wheel-speed  # target to fit

################################################################################################################
# ADAPT SETTINGS ABOVE THIS
################################################################################################################

# Make sure output dir exists
mkdir -p $out_dir
 
# After the job is submitted, run the python script
python run_ephys_decoding.py "$SLURM_ARRAY_TASK_ID" "$n_pseudo" "$n_per_job" "$base_idx" "$out_dir" "$target"
