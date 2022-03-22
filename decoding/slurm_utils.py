# created by Brandon Benson 2022-02-12
import os

def slurm_submit_python_file(py_file, inputstr, 
                             ADD_TO_SAVING_PATH = '_',
                             n_days = 2,
                             n_hours = 0,
                             n_gigs_ram = 4,
                             partition_name = 'normal',
                             SLURM_DIRECTORY = '~/slurm/',
                             SUBDIRECTORY = ''):
    '''
    ADD_TO_SAVING_PATH must not be empty
    '''
    slurm_sub_directory = os.path.join(SLURM_DIRECTORY,SUBDIRECTORY)
    if not os.path.exists(slurm_sub_directory):
        os.mkdir(slurm_sub_directory)
        
    job_file = os.path.join(slurm_sub_directory, inputstr+'.job')
    with open(job_file,'w') as fj:
        fj.writelines("#!/bin/bash\n")
        fj.writelines("#SBATCH --job-name=%s\n" % (os.path.join(slurm_sub_directory, inputstr+'.job')))
        fj.writelines("#SBATCH --output=%s\n" % (os.path.join(slurm_sub_directory, inputstr+'.out')))
        fj.writelines("#SBATCH --error=%s\n" % (os.path.join(slurm_sub_directory, inputstr+'.err')))
        fj.writelines("#SBATCH --time=%d-%d\n" % (n_days,n_hours))
        fj.writelines("#SBATCH --mem=%dG\n" % n_gigs_ram)
        fj.writelines("#SBATCH --qos=%s\n" % partition_name)
    #         fj.writelines("#SBATCH --mail-type=ALL\n")
    #         fj.writelines("#SBATCH --mail-user=$USER@stanford.edu\n")
        fj.writelines("python %s %s %s\n" %(py_file,inputstr,ADD_TO_SAVING_PATH))

    os.system("sbatch %s" %job_file)
    return

def slurm_outs(label='', SLURM_DIRECTORY='~/slurm/',SUBDIRECTORY=''):
    for file in os.listdir(os.path.join(SLURM_DIRECTORY,SUBDIRECTORY)):
        if len(file) >= 4 and len(file) >= len(label) and file[-4:] == '.out' and label in file:
            yield file
            
def get_decoding_output_files(label = '', 
                              SLURM_DIRECTORY = '~/slurm/',
                              SUBDIRECTORY = ''):
    decoding_output_files = []
    for file in slurm_outs(label=label, SLURM_DIRECTORY=SLURM_DIRECTORY, SUBDIRECTORY=SUBDIRECTORY):
        with open(os.path.join(SLURM_DIRECTORY,SUBDIRECTORY,file),'r') as fr:
            for line in fr:
                if line[:16] == 'saving to files:':
                    decoding_output_files.extend([line_part for line_part in line.split('\'') if (label in line_part) and ('.pkl' in line_part)])

    return decoding_output_files