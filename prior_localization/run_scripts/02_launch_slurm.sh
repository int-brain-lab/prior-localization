#!/bin/bash

#### USER SETTINGS, ADAPT THESE TO YOUR CLUSTER ###
out_dir=/mnt/ibl/quarantine/prior/ephys
###################################################

n_sessions=354  # number of unique eids in the dataframe that we will index in
n_pseudo=200  # number of pseudo sessions to generate for each real session
n_per_job=10  # number of (pseudo)sessions to fit per job on the cluster

mkdir -p $out_dir

# Create a file in which each line has an index for the eid to use, and a set of pseudo sessions to use
idx_list="$out_dir/idx_list.txt"

# TODO: write to idx_list.txt:
#0 (-1 1 2 3 4 5 6 7 8 9 10)
#0 (11 12 13 14 15 16 17 18 19 20)
#...
#0 (191 192 193 194 195 196 197 198 199 200)
#1 (-1 1 2 3 4 5 6 7 8 9 10)
#1 (11 12 13 14 15 16 17 18 19 20)
#...
#353 (191 192 193 194 195 196 197 198 199 200)

# Get the total number of jobs
n_jobs=$(wc -l $idx_list)

# Launch a Slurm job array with $nfiles entries
sbatch --array=1-$n_jobs job.slurm $idx_list $out_dir