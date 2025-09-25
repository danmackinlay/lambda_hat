# Lambda-Hat: Local Learning Coefficient (LLC) Estimation

Lambda-Hat provides a streamlined framework for estimating the Local Learning Coefficient (LLC) using various MCMC samplers implemented in JAX (via BlackJAX). It uses Hydra for configuration management and Haiku for neural network definitions.

## Installation

Requires Python 3.11+.

```bash
# Using uv (recommended):
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra cpu   # For CPU/macOS
uv sync --extra cuda12  # For CUDA 12 (Linux)

# Or using pip:
pip install .[cpu]     # For CPU/macOS
pip install .[cuda12]  # For CUDA 12 (Linux)
```

## Running Experiments

Lambda-Hat provides two entry points for running experiments. Configuration is managed by Hydra.

### Basic Usage

Run the default configuration (MLP target, all samplers):

```bash
# Console script (recommended)
uv run lambda-hat

# Module entry
uv run python -m lambda_hat
```

Outputs (logs, plots, metrics) are automatically saved in a timestamped directory under `outputs/`.

### Using Configuration Presets

The configuration is composable. You can select presets defined in the `conf/` directory.

Run a quick, small experiment using the `fast` sampler settings and `small` model/data:

```bash
uv run lambda-hat sampler=fast model=small data=small
```E.md 

### Overriding Parameters

Override any configuration parameter from the command line:

```bash
# Change the dataset size and random seed
uv run lambda-hat data.n_data=5000 seed=123

# Change the model architecture
uv run lambda-hat model.depth=5 model.target_params=20000

# Adjust sampler settings
uv run lambda-hat sampler.hmc.draws=2000 sampler.sgld.step_size=1e-5
```

### Running Sweeps (Multi-Run)

Hydra allows running sweeps over parameters using the `--multirun` (or `-m`) flag.

```bash
# Sweep over different model sizes
uv run lambda-hat -m model.target_params=1000,5000,10000

# Compare base vs fast sampler settings
uv run lambda-hat -m sampler=base,fast
```

Combine sweeps (Cartesian product):

```bash
# 2 sizes x 2 sampler configs = 4 runs
uv run lambda-hat -m model.target_params=1000,5000 sampler=base,fast
```

Multi-run outputs are saved under `multirun/`.

## Two-Stage Workflow (Experimental)

Lambda-Hat now supports a two-stage workflow that separates target building from sampling. This allows you to build a target once and run multiple samplers on it with different configurations:

### Stage 1: Build Target

Build and train the neural network target once:

```bash
# Build a small target with specific configuration
uv run lambda-hat-build-target model=small data=small target.seed=42

# The command outputs a target ID like: tgt_abcd1234
# This target is saved under runs/targets/tgt_abcd1234/
```

### Stage 2: Sample from Target

Run different samplers on the same target:

```bash
# Run HMC on the target
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=hmc

# Run SGLD with custom parameters
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=sgld sampler.sgld.step_size=1e-5

# Parameter sweep on the same target
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=hmc,sgld,mclmc
```

Results are saved under `runs/samples/<target_id>/<sampler>/`.

This workflow ensures:
- **Reproducibility**: Same target ID = same neural network
- **Efficiency**: Train once, sample many times
- **Isolation**: Target config and sampler config are separate

## Asset Promotion

Lambda-Hat includes a promotion utility to select the latest plots from each sampler and copy them to a stable `assets/` directory for README display:

```bash
# Promote latest theta trace plots for all samplers
uv run lambda-hat-promote --samplers sgld,hmc,mclmc --plot-name theta_trace.png

# Promote running LLC plots from specific runs directory
uv run lambda-hat-promote --runs-root runs --outdir assets --plot-name running_llc.png
```

This command:
- Finds the most recent run directory for each specified sampler
- Copies the specified plot from each run's `analysis/` directory
- Saves plots as `assets/{sampler}.png` for stable referencing

## Documentation

- [Configuration Details](./docs/configuration.md)
- [Running on SLURM](./docs/parallelism.md)
- [BlackJAX Notes](./docs/blackjax.md)
