#!/bin/bash
#SBATCH --account=stats
#SBATCH -c 8
#SBATCH --job-name=decoding
#SBATCH --output=/moto/stats/users/mw3323/logs/slurm/decoding.%A.%a.out
#SBATCH --error=/moto/stats/users/mw3323/logs/slurm/decoding.%A.%a.err
#SBATCH --mem=32GB
#SBATCH --time=11:59:00
#SBATCH --array=1-475

# --array=1-918

module load anaconda

n_sessions=459  # number of unique eids in the dataframe that we will index in
n_pseudo=100  # 200; number of pseudo sessions to generate for each real session
n_per_job=4  # 100; number of (pseudo)sessions to fit per job on the cluster
base_idx=11000  # add this to the task id; terremoto doesn't allow array values >1000

target=wheel-speed  # target to fit

################################################################################################################
# ADAPT SETTINGS ABOVE THIS
################################################################################################################

# Make sure output dir exists
mkdir -p $out_dir
 
# After the job is submitted, run the python script
python run_ephys_decoding.py "$SLURM_ARRAY_TASK_ID" "$n_pseudo" "$n_per_job" "$base_idx" "$target"
