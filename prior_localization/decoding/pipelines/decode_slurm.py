try:
    index = int(sys.argv[1]) - 1
except:
    index = 32
    pass



pid_idx = index % bwm_df.index.size
job_id = index // bwm_df.index.size

pseudo_ids = (
        np.arange(job_id * N_PSEUDO_PER_JOB, (job_id + 1) * N_PSEUDO_PER_JOB) + 1
)
if 1 in pseudo_ids:
    pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype("int64")

print('Slurm job successful')
