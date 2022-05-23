# created by Brandon Benson 2022-02-12
import os
import pickle
import pandas as pd
import numpy as np
from datetime import date

def submit_python_file(py_file, inputstr, 
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

def gather_save_outputs(SUBDIRECTORY, SLURM_DIRECTORY, OUTPUT_PATH, DATE = str(date.today())):
    finished = get_decoding_output_files(SLURM_DIRECTORY = SLURM_DIRECTORY,
                                     SUBDIRECTORY = SUBDIRECTORY)
    #%%
    indexers = ['subject', 'eid', 'probe', 'region']
    indexers_neurometric = ['low_slope', 'high_slope', 'low_range', 'high_range', 'shift']
    resultslist = []
    for fn in finished:
        try:
            fo = open(fn, 'rb')
            result = pickle.load(fo)
            N_PSEUDO = len(result['pseudosessions'])
            fo.close()
        except:
            print('failed to open file %s'%fn)
            continue
        if np.any(np.array([('_neuralplustarget' in k) for k in result['fit'].keys()])):
            print('successfully ran relative target')
        tmpdict = {**{x: result[x] for x in indexers},
                   'fold': -1,
                   'mask': ''.join([str(item) for item in list(result['fit']['mask'].values * 1)]),
                   'Score_test': result['fit']['Score_test_full'],
                   **{f'Score_test_pseudo{i}': result['pseudosessions'][i]['Score_test_full']
                      for i in range(N_PSEUDO)}}
        if result['fit']['full_neurometric'] is not None \
                and np.all([result['pseudosessions'][i]['full_neurometric'] is not None for i in range(N_PSEUDO)]):
            print(len(result['pseudosessions']))
            tmpdict = {**tmpdict,
                       **{idx_neuro: result['fit']['full_neurometric'][idx_neuro]
                          for idx_neuro in indexers_neurometric},
                       **{str(idx_neuro) + f'_pseudo{i}': result['pseudosessions'][i]['full_neurometric'][idx_neuro]
                          for i in range(N_PSEUDO) for idx_neuro in indexers_neurometric}}
        resultslist.append(tmpdict)
        for kfold in range(result['fit']['nFolds']):
            tmpdict = {**{x: result[x] for x in indexers},
                       'fold': kfold,
                       'Score_test': result['fit']['Scores_test'][kfold],
                       'Best_regulCoef': result['fit']['best_params'][kfold],
                       **{f'Score_test_pseudo{i}': result['pseudosessions'][i]['Scores_test'][kfold]
                          for i in range(N_PSEUDO)},
                       }
            if result['fit']['fold_neurometric'] is not None:
                tmpdict = {**tmpdict,
                           **{idx_neuro: result['fit']['fold_neurometric'][kfold][idx_neuro]
                              for idx_neuro in indexers_neurometric}}
            if np.all([result['pseudosessions'][i]['fold_neurometric'] is not None for i in range(N_PSEUDO)]):
                tmpdict = {**tmpdict,
                           **{str(idx_neuro) + f'_pseudo{i}': result['pseudosessions'][i][
                               'fold_neurometric'][kfold][idx_neuro]
                              for i in range(N_PSEUDO) for idx_neuro in indexers_neurometric}
                           }
            resultslist.append(tmpdict)
    resultsdf = pd.DataFrame(resultslist).set_index(indexers)

    fn = os.path.join(OUTPUT_PATH,SUBDIRECTORY,DATE+'_results')
    fn = fn + '.parquet'
    resultsdf.to_parquet(fn)

