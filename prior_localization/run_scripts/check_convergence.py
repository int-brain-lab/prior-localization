import os
from pathlib import Path
import re
import sys


SLURM_DIR = Path(sys.argv[1])
job_name = sys.argv[2]

def check_filename(filename, job_id, string):
    pattern = r'\b{}\b.*\b{}\b'.format(job_id, string)
    return bool(re.search(pattern, filename))

fs = [f for f in os.listdir(SLURM_DIR) if check_filename(f, job_name, "err")]
fs_out = [f for f in os.listdir(SLURM_DIR) if check_filename(f, job_name, "out")]
print(f'found {len(fs)} matching error files in {SLURM_DIR}')
print(f'found {len(fs_out)} matching output files in {SLURM_DIR}')

fs.sort()
fs_out.sort()

conv_warn_files = []
cancel_files = []
for f in fs:
    with open(SLURM_DIR.joinpath(f), "r") as fo:
        s = fo.read().replace("\n", "")
        if re.match(".*ConvergenceWarning.*", s):
            conv_warn_files.append(f)
        if re.match(".*CANCELLED.*", s):
            cancel_files.append(f)

non_success_files = []
failed_fold_files = []
failed_nonpseudo_files = []
failed_pseudo_files = []
for f in fs_out:
    with open(SLURM_DIR.joinpath(f), "r") as fo:
        s = fo.read().replace("\n","")
        if not (re.match(".*Job successful.*", s) or re.match(".*ended job because this job_repeat.*", s)):
            non_success_files.append(f)
        if re.match(".*sampled outer folds.*", s):
            failed_fold_files.append(f)
        if re.match(".*sampled pseudo sessions.*", s):
            failed_pseudo_files.append(f)
        if re.match(".*decoding could not be done.*", s): # target failed logistic regression criteria
            failed_nonpseudo_files.append(f)

print()
print("Convergence warning files:")
print('\n'.join(conv_warn_files))
print()
print("Failed target files:")
print('\n'.join(failed_nonpseudo_files))
print()
print("Sampled pseudo session files:")
print('\n'.join(failed_pseudo_files))
print()
print("Sampled outer fold files:")
print('\n'.join(failed_fold_files))
print()
print("Cancelled files:")
print('\n'.join(cancel_files))
print()
print("Non-successful files:")
print('\n'.join(non_success_files))
