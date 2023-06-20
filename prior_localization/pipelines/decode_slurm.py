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


# FROM MOTOR
N_CORES = 4

cluster = SLURMCluster(cores=N_CORES,
                       memory='32GB',
                       processes=1,
                       queue="shared-cpu",
                       walltime="01:15:00",
                       log_directory='/srv/beegfs/scratch/users/h/hubertf/dask-worker-logs',
                       interface='ib0',
                       extra=["--lifetime", "60m", "--lifetime-stagger", "10m"],
                       job_cpu=N_CORES,
                       env_extra=[
                           f'export OMP_NUM_THREADS={N_CORES}',
                           f'export MKL_NUM_THREADS={N_CORES}',
                           f'export OPENBLAS_NUM_THREADS={N_CORES}'
                       ])

# cluster = LocalCluster()
cluster.scale(20)
