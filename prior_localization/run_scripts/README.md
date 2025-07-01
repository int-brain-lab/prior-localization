# Run Decoding at Scale on a Cluster

This directory contains scripts for performing brainwide decoding analyses using the Brainwide Map (BWM) dataset.

---

## Step 0: Setup

1. **Install Dependencies**:
   Follow the setup instructions in the [`prior-localization` README](https://github.com/int-brain-lab/prior-localization/blob/main/README.md) to:
   - Install the `prior-localization` repo
   - Connect to the IBL database
   During this setup, you'll also specify the data directory.

   **Data size**: ~100 GB

2. **Configure Output Directory**:
   In `prior_localization/config.yml`, update the `output_dir` field to where decoding results should be stored.

   **Result size**: ~1 TB

---

## Step 1: Download the Data

Download the BWM dataset by running:

```bash
python prior_localization/run_scripts/01_stage_data.py
```

This will download data from all 459 sessions, including:

- Spike-sorted neural data
- Trial information (stimulus onset, contrast, feedback, etc.)
- Behavioral data (wheel movement, choice, etc.)

**Note**: Ensure a stable high-speed connection. This step may take several hours depending on speed.

Once cached, future scripts can be run offline.

---

## Step 2: Create Imposter Sessions (for Wheel Decoding Only)

To assess decoding significance for wheel variables, generate null distributions using "imposter sessions" (wheel movements from other mice).

If you're not decoding `wheel-speed` or `wheel-velocity`, skip this step.

```bash
python prior_localization/run_scripts/create_imposter_df.py --target=wheel-speed --save_dir=/path/to/folder
```

Repeat with `--target=wheel-velocity` for velocity.

Update the `imposter_df_path` field in `config.yml` with your `save_dir`.

---

## Step 3: Run Decoding

Use the SLURM job scheduler to launch decoding:

```bash
sbatch prior_localization/run_scripts/02_launch_slurm.sh
```

### SLURM Configuration Guidelines

- **Memory**: ~32GB per job
- **CPUs**: 1 (code is single-threaded)
- **Runtime**: Up to 12 hours per job

Example SLURM settings:

```bash
#SBATCH --mem=32GB
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --distribution=cyclic
```

### Script Variables

```bash
n_sessions=459
n_pseudo=100         # Number of pseudo-sessions per real session
n_per_job=4          # Number of (pseudo)sessions per job
base_idx=0           # Used to offset job array index
target=wheel-speed   # Variable to decode
```

### Config File (`config.yml`)

```yaml
min_units: 1
save_predictions: False
```

- `min_units`: Include all regions; filter downstream by neuron count.
- `save_predictions`: Set `True` to save predictions from pseudo-sessions (will drastically increase file size).
  > ⚠️ Required to be `True` for running unit tests.

---

### Decoding Target-Specific Settings

#### 1. `stimside`, `choice`, `feedback` (Binary Classification)

In `config.yml`:

```yaml
estimator: LogisticRegression
estimator_kwargs: {tol: 0.0001, max_iter: 20000, fit_intercept: True}
balanced_weighting: True
```

In `02_launch_slurm.sh`:

```bash
#SBATCH --array=1-918

n_pseudo=200
n_per_job=100
target=stimside  # or choice or feedback
```

> Each session requires 2 jobs (200 pseudo-sessions ÷ 100 per job × 459 sessions = 918 jobs total).

Typical runtime: 5–6 hours max per job. Running 100 jobs in parallel takes ~12 hours per variable.

---

#### 2. `wheel-speed`, `wheel-velocity` (Regression)

In `config.yml`:

```yaml
estimator: Lasso
estimator_kwargs: {tol: 0.0001, max_iter: 1000, fit_intercept: True}
balanced_weighting: False
imposter_df_path: /absolute/path/to/imposter/df
```

In `02_launch_slurm.sh`:

```bash
#SBATCH --array=1-1000

n_pseudo=100
n_per_job=4
target=wheel-speed  # or wheel-velocity
```

> Each session requires 25 jobs. Total: 459 × 25 = 11,475 jobs.

If your cluster limits array size (e.g., max 1000 jobs), use `base_idx` to submit multiple SLURM runs:

First batch:

```bash
#SBATCH --array=1-1000
base_idx=0
```

Second batch:

```
#SBATCH --array=1-1000
base_idx=1000
```

Final batch:

```
#SBATCH --array=1-475
base_idx=11000
```

Typical runtime: a few minutes to a few hours per job. 
Full decoding takes ~2 weeks with 100 jobs in parallel.

---

## Step 4: Logging

Update `02_launch_slurm.sh` to save logs:

```bash
#SBATCH --output=/path/to/logs/slurm/decoding.%A.%a.out
#SBATCH --error=/path/to/logs/slurm/decoding.%A.%a.err
```

To inspect logs:

```bash
cat /path/to/logs/slurm/decoding.<run_id>.<job_idx>.out
```

To check all logs from a single run:

```bash
python prior_localization/run_scripts/check_convergence.py /path/to/logs/slurm <run_id>
```

Convergence warnings are expected. Investigate any "Cancelled" or "Non-successful" messages.

---

## Step 5: Format Outputs — Stage 1

Merge decoding results across jobs:

```bash
python prior_localization/run_scripts/format_outputs_stage_1.py <target>
```

**Output**:
`<output_dir>/<target>/collected_results_stage1.pqt`

Takes 10–60 minutes depending on disk speed.

---

## Step 6: Format Outputs — Stage 2

Compute means, medians, and initial p-values:

```bash
python prior_localization/run_scripts/format_outputs_stage_2.py <target>
```

**Output**:
`<output_dir>/<target>/collected_results_stage2.pqt`
Takes ~1 minute

> For wheel decoding, add `--n_pseudo=100`.

---

## Step 7: Format Outputs — Stage 3

Aggregate session-level p-values into region-level significance using Fisher's method to combine p-values across sessions and the Benjamini Hochberg correction for multiple comparisons:

```bash
python prior_localization/run_scripts/format_outputs_stage_3.py <target>
```

**Output**:  
`<output_dir>/<target>/collected_results_stage3.pqt`

> For wheel decoding, add `--n_pseudo=100`.

> This script is where you can apply filters (e.g., min neurons per region, trial counts). See the script for details.
