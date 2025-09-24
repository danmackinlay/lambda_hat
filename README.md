# Local Learning Coefficient (LLC) Estimation

This repository provides a streamlined framework for estimating the Local Learning Coefficient (LLC) using various MCMC samplers implemented in JAX (via BlackJAX). It uses Hydra for configuration management and Haiku for neural network definitions.

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

The main entry point is `train.py`. Configuration is managed by Hydra.

### Basic Usage

Run the default configuration (MLP target, all samplers):

```bash
python train.py
```

Outputs (logs, plots, metrics) are automatically saved in a timestamped directory under `outputs/`.

### Using Configuration Presets

The configuration is composable. You can select presets defined in the `conf/` directory.

Run a quick, small experiment using the `fast` sampler settings and `small` model/data:

```bash
python train.py sampler=fast model=small data=small
```

### Overriding Parameters

Override any configuration parameter from the command line:

```bash
# Change the dataset size and random seed
python train.py data.n_data=5000 seed=123

# Change the model architecture
python train.py model.depth=5 model.target_params=20000

# Adjust sampler settings
python train.py sampler.hmc.draws=2000 sampler.sgld.step_size=1e-5
```

### Running Sweeps (Multi-Run)

Hydra allows running sweeps over parameters using the `--multirun` (or `-m`) flag.

```bash
# Sweep over different model sizes
python train.py -m model.target_params=1000,5000,10000

# Compare base vs fast sampler settings
python train.py -m sampler=base,fast
```

Combine sweeps (Cartesian product):

```bash
# 2 sizes x 2 sampler configs = 4 runs
python train.py -m model.target_params=1000,5000 sampler=base,fast
```

Multi-run outputs are saved under `multirun/`.

## Documentation

- [Configuration Details](./docs/configuration.md)
- [Running on SLURM](./docs/parallelism.md)
- [BlackJAX Notes](./docs/blackjax.md)
