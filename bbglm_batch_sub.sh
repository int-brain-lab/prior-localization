#!/bin/sh
#SBATCH --job-name=brainwide
#SBATCH --time=04:00:00
#SBATCH --partition=shared-cpu
#SBATCH --output=/home/gercek/logs/bwm_fit.%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
 
# run one task which use all the cpus of the node
conda activate iblenv
python /home/gercek/Projects/prior-localization/bbglm_cluster_worker.py \
 /home/gercek/scratch/glm_cache/2022-01-19_dataset_metadata.pkl \
 /home/gercek/scratch/results/glms/2022-02-03_glm_fit_pars.pkl \
 ${SLURM_ARRAY_TASK_ID} \
 2022-02-03
