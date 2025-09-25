# Parallel Execution with Hydra Launchers

This project uses Hydra Launchers to enable parallel execution of parameter sweeps locally or on a SLURM cluster via the `hydra-submitit-launcher` plugin.

## Overview

When you run an experiment using the `--multirun` (or `-m`) flag, Hydra generates a list of jobs based on the parameter combinations you specify.

```bash
# Example: This generates 4 jobs
uv run lambda-hat-m model=small,base data=small,base
```

By default, Hydra runs these jobs sequentially. To run them in parallel, you need to select a different launcher.

## Configuration

Launcher configurations are defined in `conf/hydra/launcher/`.

### SLURM Cluster Execution

To dispatch jobs to a SLURM cluster, select the `submitit_slurm` launcher.

**Important:** You must specify the SLURM partition required by your cluster using the `hydra.launcher.partition` override.

```bash
uv run lambda-hat --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition=YOUR_PARTITION_NAME \
  data.n_data=1000,5000 \
  model=small,base
```

This command submits 4 independent jobs to the specified SLURM partition.

#### Resource Management

The default resources requested per job are defined in the `submitit_slurm.yaml` configuration file. You can override these settings via the command line:

```bash
uv run lambda-hat --multirun hydra/launcher=submitit_slurm \
  hydra.launcher.partition=YOUR_PARTITION_NAME \
  hydra.launcher.timeout_min=480 \
  hydra.launcher.cpus_per_task=8 \
  hydra.launcher.gpus_per_node=1 \
  +hydra.launcher.additional_parameters.gpus=1
```

If your cluster requires an account, use `hydra.launcher.account=YOUR_ACCOUNT`.


### Setting slurm account name

#### One-off on the CLI


Example with GPU on a cluster that needs `--gpus=1`:

```bash
uv run lambda-hat --multirun  sampler=hmc \
  hydra/launcher=submitit_slurm \
  hydra.launcher.gpus_per_node=1 \
  +hydra.launcher.additional_parameters.gpus=1 \
  hydra.launcher.account=OD-228158
```

Notes:

* `hydra/launcher=submitit_slurm` selects the Slurm launcher.
* `hydra.launcher.account=...` maps to `sbatch --account=...`.
* If your cluster uses `--gpus=N` instead of `--gres=gpu:N`, pass it exactly as shown via `additional_parameters.gpus`.

#### If you prefer not to repeat it

Create a tiny config file and use it by name:

`lambda_hat/conf/hydra/launcher/slurm_account.yaml`

```yaml
# Merges into the submitit_slurm launcher
# Usage: hydra/launcher=submitit_slurm,hydra/launcher=slurm_account
slurm:
  additional_parameters:
    account: OD-228158
```

Then:

```bash
uv run lambda-hat-build-target hydra/launcher=submitit_slurm,hydra/launcher=slurm_account
```

### Local Parallel Execution

To run jobs in parallel locally (e.g., utilizing multiple cores on a workstation), use the `submitit_local` launcher.

```bash
uv run lambda-hat --multirun hydra/launcher=submitit_local model=small,base
```

This runs the jobs concurrently using local processes.

## GPU Management

The codebase uses JAX. When using Submitit (SLURM or Local), it is crucial to ensure that each job has access to the resources it requested.

1. **SLURM:** When `gpus_per_node=1` is set, SLURM typically handles the allocation and ensures the job only sees that GPU (e.g., via `CUDA_VISIBLE_DEVICES`).
2. **Local:** For CPU-only local execution, ensure `hydra.launcher.gpus_per_node=0` is set in the config or via override.

## Logging and Monitoring

Submitit creates log files (stdout/stderr) for each job. These are stored in a directory specified by `submitit_folder` within the Hydra sweep output directory (e.g., `multirun/YYYY-MM-DD/HH-MM-SS/.submitit_slurm/`).