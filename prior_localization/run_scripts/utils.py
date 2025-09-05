# %%
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from brainwidemap import bwm_query
from one.api import ONE

N_PSEUDO = 4
N_PER_JOB = 4
TOTAL_JOBS = N_PSEUDO / N_PER_JOB * 459  # 459 sessions


def get_job_array():
    one = ONE(base_url='https://openalyx.internationalbrainlab.org')
    bwm_df = bwm_query(one=one)
    eids = bwm_df.eid.unique()

    ijob = 1
    df_pseudo_sessions = []
    for eid in eids:
        for pseudo_id in range(1, N_PSEUDO + 1):
            ijob_session = int(np.floor((pseudo_id - 1)/ N_PER_JOB))
            df_pseudo_sessions.append({'eid': eid, 'pseudo_id': pseudo_id, 'job_id': ijob + ijob_session})
        ijob += int(N_PSEUDO / N_PER_JOB)

    df_pseudo_sessions = pd.DataFrame(df_pseudo_sessions)
    df_jobs = df_pseudo_sessions.groupby(['eid', 'job_id']).agg(
        first=pd.NamedAgg(column='pseudo_id', aggfunc='first'),
        last=pd.NamedAgg(column='pseudo_id', aggfunc='last'),
    )
    return df_jobs


def get_remaining_jobs(output_dir):
    df_jobs = get_job_array()
    # df_jobs.reset_index(inplace=True)
    # df_jobs.set_index(['eid', 'first_session', 'last_session'], inplace=True)
    errored_jobs = np.array([float(error_file.stem.split('_')[-1]) for error_file in output_dir.glob('ERROR_jobid_*.txt')]).astype(int)

    file_results = list(output_dir.rglob('*merged_probes_pseudo_ids*.pkl'))
    print(f'Found {len(file_results)} merged_probes_pseudo_ids.pkl files')
    df_results = []
    for file_path in file_results:
        # Extract components from the path
        # Path structure: /mnt/s1/2025/unit-refine/decoding/vanilla/{subject}/{eid}/{region}_merged_probes_pseudo_ids_{first}_{last}.pkl

        subject = file_path.parts[-3]  # Third-to-last part is the subject
        eid = file_path.parts[-2]  # Second-to-last part is the eid

        # Parse the filename to get region, first, and last
        filename = file_path.name

        # Extract region (everything before '_merged_probes_pseudo_ids_')
        region = filename.split('_merged_probes_pseudo_ids_')[0]

        # Extract first and last from the end of the filename
        # Format: {region}_merged_probes_pseudo_ids_{first}_{last}.pkl
        first_last = filename.split('_merged_probes_pseudo_ids_')[1].replace('.pkl', '')
        first, last = first_last.split('_')
        # Add to parsed data
        df_results.append({
            # 'subject': subject,
            'eid': eid,
            'region': region,
            'first': abs(int(first)),
            'last': int(last),
            'path': str(file_path)
        })

    # Create a DataFrame from the parsed data
    df_results = pd.DataFrame(df_results)
    if df_results.empty:
        return None, np.arange(1, TOTAL_JOBS + 1).astype(int)
    # Display the DataFrame
    df_complete = df_results.groupby(['eid', 'first', 'last']).agg(
        region_count=pd.NamedAgg(column='region', aggfunc='count'),
    )

    df_status = df_jobs.join(df_complete, on=['eid', 'first', 'last'], how='left').reset_index()
    n_complete = TOTAL_JOBS - df_status['region_count'].isna().sum()
    df_status.loc[df_status['job_id'].isin(errored_jobs), 'region_count'] = -1
    print(f'Number of complete jobs: {n_complete} over {TOTAL_JOBS} total jobs, {n_complete / TOTAL_JOBS * 100:.2f}%')
    print(f'Number of errored jobs: {errored_jobs.size} over {TOTAL_JOBS} total jobs, { errored_jobs.size / TOTAL_JOBS * 100:.2f}%')

    jobs2run = df_status.loc[df_status.region_count.isna(), 'job_id'].values


    return df_status, jobs2run


def delete_all_results(output_dir, dry=True):
    c = 0
    for file in output_dir.rglob('*_merged_probes_pseudo_ids*.pkl'):
        if not dry:
            file.unlink()
        c += 1
    print(f'Deleted {c} merged_probes_pseudo_ids.pkl files')


def make_jobs(output_dir):

    shutil.rmtree(output_dir, ignore_errors=True)
    df_status, jobs2run = get_remaining_jobs(output_dir)
    print(f'Making {jobs2run.size} jobs in {output_dir}')
    output_dir.joinpath('joblist').mkdir(parents=True, exist_ok=True)
    for job_id in jobs2run:
        output_dir.joinpath('joblist', f'jobid_{int(job_id):05}.txt').touch()
    print('Copy behavior model folder to the output directory')
    shutil.copytree(output_dir.parent.joinpath('behavior'), output_dir.joinpath('behavior'))

# ['vanilla', 'unit-refine', 'unit-refine-dummy']
output_dir = Path('/mnt/s1/2025/unit-refine/decoding/vanilla')
output_dir = Path('/mnt/s1/2025/unit-refine/decoding/unit-refine')
output_dir = Path('/mnt/s1/2025/unit-refine/decoding/unit-refine-control')
make_jobs(output_dir)

# df_status, jobs2run = get_remaining_jobs(output_dir)
# delete_all_results(output_dir, dry=True)

# source ~/PycharmProjects/ibl-task-forces/.venv/bin/activate
# cd ~/PycharmProjects/ibl-task-forces/prior-localization/prior_localization/run_scripts
# python run_ephys_decoding_joblib.py


# %%
from prior_localization.run_scripts import format_outputs_stage_1
from pathlib import Path
import pandas as pd
all_results = []
LABELS = ['vanilla', 'unit-refine', 'unit-refine-control']
LABELS = ['vanilla', 'unit-refine']
for label in LABELS:
    output_dir = Path(f'/mnt/s1/2025/unit-refine/decoding/{label}')
    fn_results = output_dir.joinpath('collected_results_stage1.pqt')
    if fn_results.exists():
        df_results = pd.read_parquet(fn_results)
    else:
        df_results = format_outputs_stage_1.run(output_dir)
    df_results['label'] = label
    all_results.append(df_results)

df_results = pd.concat(all_results)
df_results.to_parquet('/mnt/s1/2025/unit-refine/decoding/all_results.pqt')
