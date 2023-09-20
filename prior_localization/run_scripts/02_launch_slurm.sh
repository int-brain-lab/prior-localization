# Create a file in which each line has an index for the eid to use, and a set of pseudo sessions to use
# Launch slurm task array calling slurm.job script with this file as input


#!/bin/bash


n_pseudo_sessions: 200  # Number of pseudo sessions to decode per real session for statistical analysis


# Recommendation: keep scripts in $HOME, and data in ceph
projdir="$HOME/ceph/myproj/"  # dir with data*.hdf5
jobname="job1"  # change for new jobs
jobdir="$projdir/$jobname"

mkdir -p $jobdir

# Use the "find" command to write the list of files to process, 1 per line
fn_list="$jobdir/fn_list.txt"
find $projdir -name 'data*.hdf5' | sort > $fn_list
nfiles=$(wc -l $fn_list)

# Launch a Slurm job array with $nfiles entries
sbatch --array=1-$nfiles job.slurm $fn_list