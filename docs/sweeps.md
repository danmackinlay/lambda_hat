# Parameter Sweeps with Hydra Multirun

This document explains how to design and execute parameter sweeps using Hydra's `--multirun` functionality. This allows you to explore the parameter space efficiently by running multiple configurations in a single command.

## Core Concepts

Hydra allows you to define variations of your configuration along multiple dimensions. When you use the `--multirun` (or `-m`) flag, Hydra generates a Cartesian product of all specified variations and executes the application once for each combination.

**Note on Execution Flow:** In the current implementation (`train.py`), each Hydra job executes all primary samplers (SGLD, HMC, MCLMC) sequentially.

## Basic Sweeps

### Sweeping over Configuration Groups

You can sweep over predefined configuration groups (e.g., different model sizes or sampler settings).

```bash
# Sweep over 'small' and 'base' models
python train.py -m model=small,base
```

This command launches two jobs.

### Sweeping over Scalar Parameters

You can sweep over individual parameters defined in your configuration using comma-separated lists.

```bash
# Sweep over different data sizes and noise scales
python train.py -m data.n_data=1000,5000,10000 data.noise_scale=0.01,0.1
```

This command launches 3 * 2 = 6 jobs.

## Advanced Sweeps

### Range Sweeps

Hydra supports range functions for numerical sweeps.

```bash
# Sweep seeds from 1 to 5 (inclusive)
python train.py -m seed=range\(1,6\)
```

### Defining Sweeps in YAML (Experiments)

For complex experimental designs, it can be cumbersome to specify everything on the command line. Hydra allows you to define reusable sweep configurations using the `+experiment` convention.

Create a YAML file defining the sweep parameters in `conf/experiment/` (you may need to create this directory).

```yaml
# Example: conf/experiment/data_scaling_sweep.yaml
# @package _global_
defaults:
  - /config # Inherit defaults from the main config

# Define the sweep parameters
hydra:
  mode: MULTIRUN
  sweeper:
    params:
      data.n_data: 1000,5000,10000
      model: small,base
```

Execute the sweep using the `+experiment` syntax:

```bash
python train.py +experiment=data_scaling_sweep
```

## Parallel Execution

By default, `--multirun` executes jobs sequentially. To execute them in parallel (e.g., on a SLURM cluster), you must combine `--multirun` with a Hydra Launcher.

See [Parallel Execution](./parallelism.md) for details.

```bash
# Example: Running a sweep on SLURM
python train.py -m model=small,base \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition=YOUR_PARTITION
```

## Analyzing Results

When a sweep completes, the results are stored in the `multirun/` directory. See [Output Management](./output_management.md) for details on the directory structure and tips on aggregating results.