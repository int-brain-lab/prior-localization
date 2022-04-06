import pandas as pd
import sys
from settings.settings import *
from functions.decoding import fit_eid
import numpy as np
from wide_field_imaging import utils as wut

try:
    index = int(sys.argv[1]) - 1
except:
    index = 4
    pass

# import cached data
bwmdf = pd.read_parquet(DECODING_PATH.joinpath('insertions.pqt')).reset_index(drop=True)
bwmdf = bwmdf[bwmdf.spike_sorting != '']
eids = bwmdf['eid'].unique()

eids = eids[:10]

# create necessary empty directories if not existing
DECODING_PATH.joinpath('results').mkdir(exist_ok=True)
DECODING_PATH.joinpath('results', 'behavioral').mkdir(exist_ok=True)
DECODING_PATH.joinpath('results', 'neural').mkdir(exist_ok=True)
DECODING_PATH.joinpath('logs').mkdir(exist_ok=True)
DECODING_PATH.joinpath('logs', 'slurm').mkdir(exist_ok=True)

if USE_IMPOSTER_SESSION:
    imposterdf = pd.read_parquet(DECODING_PATH.joinpath('imposterSessions_beforeRecordings.pqt'))
else:
    imposterdf = None

kwargs = {'imposterdf': imposterdf, 'nb_runs': N_RUNS, 'single_region': SINGLE_REGION, 'merged_probes': MERGED_PROBES,
          'modelfit_path': DECODING_PATH.joinpath('results', 'behavioral'), 'continuous_target': CONTINUOUS_TARGET,
          'output_path': DECODING_PATH.joinpath('results', 'neural'), 'one': None, 'decoding_path': DECODING_PATH,
          'estimator_kwargs': ESTIMATOR_KWARGS, 'hyperparam_grid': HPARAM_GRID,
          'save_binned': SAVE_BINNED, 'shuffle': SHUFFLE, 'balanced_weight': BALANCED_WEIGHT,
          'normalize_input': NORMALIZE_INPUT, 'normalize_output': NORMALIZE_OUTPUT,
          'compute_on_each_fold': COMPUTE_NEURO_ON_EACH_FOLD,
          'force_positive_neuro_slopes': FORCE_POSITIVE_NEURO_SLOPES,
          'estimator': ESTIMATOR, 'target': TARGET, 'model': MODEL, 'align_time': ALIGN_TIME,
          'no_unbias': NO_UNBIAS, 'min_rt': MIN_RT, 'min_behav_trials': MIN_BEHAV_TRIAS,
          'qc_criteria': QC_CRITERIA, 'min_units': MIN_UNITS, 'time_window': TIME_WINDOW,
          'use_imposter_session': USE_IMPOSTER_SESSION, 'compute_neurometric': COMPUTE_NEUROMETRIC,
          'border_quantiles_neurometric': BORDER_QUANTILES_NEUROMETRIC, 'today': DATE,
          'add_to_saving_path': ADD_TO_SAVING_PATH, 'use_openturns': USE_OPENTURNS,
          'bin_size_kde': BIN_SIZE_KDE, 'wide_field_imaging': WIDE_FIELD_IMAGING, 'wfi_hemispheres': WFI_HEMISPHERES,
          'wfi_nb_frames': WFI_NB_FRAMES, }

if WIDE_FIELD_IMAGING:
    import glob
    subjects = glob.glob('wide_field_imaging/CSK-im-*')
    eids = np.array([np.load(s + '/behavior.npy', allow_pickle=True).size for s in subjects])
    eid_id = index % eids.sum()
    job_id = index // eids.sum()
    subj_id = np.sum(eid_id >= eids.cumsum())
    sess_id = eid_id - np.hstack((0, eids)).cumsum()[:-1][subj_id]
    if sess_id < 0:
        raise ValueError('There is an error in the code')
    sessiondf, wideFieldImaging_dict = wut.load_wfi_session(subjects[subj_id], sess_id)
    eid = sessiondf.eid[sessiondf.session_to_decode].unique()[0]
else:
    eid_id = index % eids.size
    job_id = index // eids.size
    eid = eids[eid_id]
    sessiondf, wideFieldImaging_dict = None, None

if WIDE_FIELD_IMAGING and eid in excludes or np.any(bwmdf[bwmdf['eid'] == eid]['spike_sorting'] == ""):
    print(f"dud {eid}")
else:
    print(f"session: {eid}")
    pseudo_ids = np.arange(job_id * N_PSEUDO_PER_JOB, (job_id + 1) * N_PSEUDO_PER_JOB) + 1
    if 1 in pseudo_ids:
        pseudo_ids = np.concatenate((-np.ones(1), pseudo_ids)).astype('int64')
    fit_eid(eid=eid, bwmdf=bwmdf, pseudo_ids=pseudo_ids,
            sessiondf=sessiondf, wideFieldImaging_dict=wideFieldImaging_dict, **kwargs)

    print('Slurm job successful')
