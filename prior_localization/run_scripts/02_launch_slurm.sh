#!/bin/bash
#SBATCH --account=stats
#SBATCH -c 8
#SBATCH --job-name=decoding
#SBATCH --output=/moto/stats/users/mw3323/logs/slurm/decoding.%A.%a.out
#SBATCH --error=/moto/stats/users/mw3323/logs/slurm/decoding.%A.%a.err
#SBATCH --mem=32GB
#SBATCH --time=119:59:00
#SBATCH --array=100,170,173,174,41,44,47,48,49,50,51,53,54,57,58,59,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,83,85,86,87,88,89,90,91,92,93,94,95,96,97

# --array=1-918

module load anaconda

out_dir=/moto/stats/users/mw3323/results
n_sessions=459  # number of unique eids in the dataframe that we will index in
n_pseudo=100  # 200 # number of pseudo sessions to generate for each real session
n_per_job=5  # 100  # number of (pseudo)sessions to fit per job on the cluster
base_idx=9000  # add this to the task id; terremoto doesn't allow array values >1000

target=wheel-velocity  # target to fit

################################################################################################################
# ADAPT SETTINGS ABOVE THIS
################################################################################################################

# Make sure output dir exists
mkdir -p $out_dir
 
# After the job is submitted, run the python script
python run_ephys_decoding.py "$SLURM_ARRAY_TASK_ID" "$n_pseudo" "$n_per_job" "$base_idx" "$out_dir" "$target"
