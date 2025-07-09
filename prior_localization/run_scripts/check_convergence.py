import os
from pathlib import Path
import re
import sys

job_name = sys.argv[1]
SLURM_DIR = Path(sys.argv[2])


def check_filename(filename, job_id, string):
    pattern = r'\b{}\b.*\b{}\b'.format(job_id, string)
    return bool(re.search(pattern, filename))


fs_err = [f for f in os.listdir(SLURM_DIR) if check_filename(f, job_name, "err")]
fs_out = [f for f in os.listdir(SLURM_DIR) if check_filename(f, job_name, "out")]
print(f'found {len(fs_err)} matching error files in {SLURM_DIR}')
print(f'found {len(fs_out)} matching output files in {SLURM_DIR}')

fs_err.sort()
fs_out.sort()

cancel_files = []
for f in fs_err:
    with open(SLURM_DIR.joinpath(f), "r") as fo:
        s = fo.read().replace("\n", "")
        if re.match(".*CANCELLED.*", s):
            cancel_files.append(f)

non_success_files = []
for f in fs_out:
    with open(SLURM_DIR.joinpath(f), "r") as fo:
        s = fo.read().replace("\n","")
        if not re.match(".*Job successful.*", s):
            non_success_files.append(f)

print()
print("Convergence warning files:")
print('\n'.join(conv_warn_files))
print()
print("Cancelled files:")
print('\n'.join(cancel_files))
print()
print("Non-successful files:")
print('\n'.join(non_success_files))
