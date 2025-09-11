import argparse
import joblib
from pathlib import Path
import yaml

import numpy as np
from one.api import ONE

from brainwidemap.bwm_loading import bwm_query
from prior_localization.fit_data import fit_session_ephys
import prior_localization

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

n_jobs = 20  # number of jobs to run concurrently
N_PSEUDO = 16
N_PER_JOB = 4
TARGET = 'choice'

OUTPUT_DIR = Path(f'/datadisk/Data/2025/09_unit-refine/decoding/{TARGET}')

TOTAL_JOBS = N_PSEUDO / N_PER_JOB * 459  # 459 sessions

def run_single_job(job_idx, output_dir, config=None):
    # Get session idx
    session_idx = int(np.ceil(job_idx / (N_PSEUDO / N_PER_JOB)) - 1)

    # Set of pseudo sessions for one session
    all_pseudo = list(range(N_PSEUDO))
    # Select relevant pseudo sessions for this job
    pseudo_idx = int((job_idx - 1) % (N_PSEUDO / N_PER_JOB) * N_PER_JOB)
    pseudo_ids = all_pseudo[pseudo_idx:pseudo_idx + N_PER_JOB]
    # Shift by 1; old array starts at 0, pseudo ids should start at 1
    pseudo_ids = list(np.array(pseudo_ids) + 1)
    # Add real session to first pseudo block
    if pseudo_idx == 0:
        pseudo_ids = [-1] + pseudo_ids

    # Create an offline ONE instance, we don't want to hit the database when running so many jobs in parallel and have
    # downloaded the data before
    one = ONE(base_url='https://openalyx.internationalbrainlab.org')

    # Get info for respective eid from bwm_dataframe
    bwm_df = bwm_query(one=one, freeze='2023_12_bwm_release')
    session_id = bwm_df.eid.unique()[session_idx]
    subject = bwm_df[bwm_df.eid == session_id].subject.unique()[0]
    # We are merging probes per session, therefore using a list of all probe names of a session as input
    probe_name = list(bwm_df[bwm_df.eid == session_id].probe_name)
    probe_name = probe_name[0] if len(probe_name) == 1 else probe_name

    # set BWM defaults here
    min_rt = 0.08
    max_rt = 4
    binsize = None
    n_bins_lag = None
    n_bins = None
    n_runs = 4

    # loads in the specific target parameters
    config_target_file = Path(prior_localization.__file__).parent.joinpath('configs', f'{TARGET}.yml')
    if config_target_file.exists():
        with open(config_target_file, "r") as config_yml:
            config = yaml.safe_load(config_yml) | config

    if TARGET == 'stimside':
        align_event = 'stimOn_times'
        time_window = (0.0, 0.1)
        saturation_intervals = 'saturation_stim_plus01'
        model = 'oracle'
        estimator = 'LogisticRegression'

    elif TARGET == 'signcont':
        align_event = 'stimOn_times'
        time_window = (0.0, 0.1)
        saturation_intervals = 'saturation_stim_plus01'
        model = 'oracle'
        estimator = 'Lasso'

    elif TARGET == 'choice':
        align_event = 'firstMovement_times'
        time_window = (-0.1, 0.0)
        saturation_intervals = 'saturation_move_minus02'
        model = 'actKernel'
        estimator = 'LogisticRegression'

    elif TARGET == 'feedback':
        align_event = 'feedback_times'
        time_window = (0.0, 0.2)
        saturation_intervals = 'saturation_feedback_plus04'
        model = 'actKernel'
        estimator = 'LogisticRegression'

    elif TARGET == 'pLeft':
        align_event = 'stimOn_times'
        saturation_intervals = 'saturation_stim_minus06_plus06'
        time_window = (-0.6, -0.1)
        saturation_intervals = 'saturation_stim_minus06_plus06'
        model = 'optBay'
        estimator = 'LogisticRegression'

    elif TARGET in ['wheel-speed', 'wheel-velocity']:
        align_event = 'firstMovement_times'
        time_window = (-0.2, 1.0)
        saturation_intervals = 'saturation_move_minus02'
        model = 'oracle'
        estimator = 'Lasso'
        binsize = 0.02
        n_bins_lag = 10
        n_bins = 60
        n_runs = 2

    else:
        raise ValueError(f'{TARGET} is an invalid target value')

    # Run the decoding for the current set of pseudo ids.
    results = fit_session_ephys(
        one, session_id, subject, probe_name, output_dir=output_dir, pseudo_ids=pseudo_ids, target=TARGET,
        align_event=align_event, min_rt=min_rt, max_rt=max_rt, time_window=time_window,
        saturation_intervals=saturation_intervals,
        model=model, n_runs=n_runs, binsize=binsize, n_bins_lag=n_bins_lag, n_bins=n_bins,
        compute_neurometrics=False, motor_residuals=False, config=config,
    )

    # Print out success string so we can easily sweep through error logs
    print('Job successful')

# run_single_job(job_idx=1)
def run_job(job_idx, output_dir, config):
    try:
        run_single_job(job_idx=job_idx, output_dir=output_dir, config=config)
    except Exception as e:
        import traceback
        print(f'ERROR: {str(e)}')
        error_file = output_dir.joinpath(f'ERROR_jobid_{int(job_idx):05}.txt')
        with open(error_file, 'w') as f:
            f.write(f"Error occurred: {str(e)}\n")
            f.write(traceback.format_exc())
    finally:
        output_dir.joinpath('.00_joblist', f'jobid_{int(job_idx):05}.txt').unlink()

jobs = []
for uqc in [-4, -3, -2, -1, 1]:  # we don´t  run all units for now
    config = {'unit_qc': uqc, 'estimator': 'LogisticRegression', 'use_native_sklearn_for_hyperparam_estimation': False}
    match config['unit_qc']:
        case 1:
            output_dir = OUTPUT_DIR.joinpath('vanilla')
        case -1:
            output_dir = OUTPUT_DIR.joinpath('sirena')
        case -2:
            output_dir = OUTPUT_DIR.joinpath('sirena-control')
        case -3:
            output_dir = OUTPUT_DIR.joinpath('unit-refine')
        case -4:
            output_dir = OUTPUT_DIR.joinpath('unit-refine-control')
    jobs2run = [int(f.stem.split('_')[-1]) for f in output_dir.joinpath('.00_joblist').glob('jobid_*.txt')]
    print(output_dir, len(jobs2run))
    jobs.extend([joblib.delayed(run_job)(job_idx=jid, output_dir=output_dir, config=config) for jid in jobs2run])
joblib.Parallel(n_jobs=n_jobs)(jobs)
