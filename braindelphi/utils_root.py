"""
Utility functions for the prior-localization repository
"""
# Standard library
import logging
import pickle
from pathlib import Path

# Third party libraries
import numpy as np

# IBL libraries
from one.api import ONE

log = logging.getLogger('braindelphi')


# def sessions_with_region(acronym, one=None):
#     if one is None:
#         one = ONE()
#     query_str = (f'channels__brain_region__acronym__icontains,{acronym},'
#                  'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,'
#                  'probe_insertion__session__qc__lt,50,'
#                  '~probe_insertion__json__qc,CRITICAL')
#     traj = one.alyx.rest('trajectories',
#                          'list',
#                          provenance='Ephys aligned histology track',
#                          django=query_str)
#     eids = np.array([i['session']['id'] for i in traj])
#     sessinfo = [i['session'] for i in traj]
#     probes = np.array([i['probe_name'] for i in traj])
#     return eids, sessinfo, probes


def make_batch_slurm(filename,
                     scriptpath,
                     job_name='brainwide',
                     partition='shared-cpu',
                     time='01:00:00',
                     condapath=Path('~/mambaforge/'),
                     envname='iblenv',
                     logpath=Path('~/worker-logs/'),
                     cores_per_job=4,
                     memory='16GB',
                     array_size='1-100',
                     f_args=[]):
    fw = open(filename, 'wt')
    fw.write('#!/bin/sh\n')
    fw.write(f'#SBATCH --job-name={job_name}\n')
    fw.write(f'#SBATCH --time={time}\n')
    fw.write(f'#SBATCH --partition={partition}\n')
    fw.write(f'#SBATCH --array={array_size}\n')
    fw.write(f'#SBATCH --output={logpath.joinpath(job_name)}.%a.out\n')
    fw.write('#SBATCH --ntasks=1\n')
    fw.write(f'#SBATCH --cpus-per-task={cores_per_job}\n')
    fw.write(f'#SBATCH --mem={memory}\n')
    fw.write('\n')
    fw.write(f'source {condapath}/bin/activate\n')
    fw.write(f'conda activate {envname}\n')
    fw.write(f'python {scriptpath} {" ".join(f_args)}\n')
    fw.close()
    return


def load_pickle_data(pkl_path):
    '''
    Loads pkl file and returns data in a dictionary
    '''
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data
