# `lambda_hat`: Local Learning Coefficient (LLC) Estimation

In [Singular Learning Theory (SLT)](https://singularlearningtheory.com), the **Local Learning Coefficient (LLC)** quantifies the effective local dimensionality of a model around a trained optimum.
Estimating it can be tricky. That is what we explore here.

![](assets/readme/llc_convergence_combined.png)

This repo provides benchmark estimators of LLC on small but non-trivial neural networks, using standard industrial tooling:

* [BlackJAX](https://github.com/blackjax-devs/blackjax/tree/1.2.5) for sampling
* [ArviZ](https://python.arviz.org/) for diagnostics,
* [Hydra](https://hydra.cc/) for configuration management
* [Haiku](https://github.com/haiku/haiku) for neural network definitions.

We target networks with parameter-space dimension up to about $10^5$ which means we can ground-truth against classic samplers like HMC (which we expect to become non-viable as either dimension or cardinality of the dataset increases).
In this regime we can rely upon classic MCMC to tell us the “true” LLC rather than relying on analytic results for approximate networks such as Deep Linear Networks.

## Installation

Requires Python 3.11

```bash
# Using uv (recommended):
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra cpu   # For CPU/macOS
uv sync --extra cuda12  # For CUDA 12 (Linux)

# Or using pip:
pip install .[cpu]     # For CPU/macOS
pip install .[cuda12]  # For CUDA 12 (Linux)
```

Notes: Previously the requirement in the toml was for jaxlib >= 0.7.1 (or CUDA variant same version) In order to make this project intel mac compatible we've pushed that requirement back to jaxlib >= 0.4.38. Testing now to see if this is ok.

## Running Experiments

`lambda_hat` provides two entry points for running experiments. Configuration is managed by Hydra.

### Basic Usage

Run the default configuration (MLP target, all samplers):

```bash
# Console script (recommended)
uv run lambda-hat
```

Outputs (logs, plots, metrics) are automatically saved in a timestamped directory under `outputs/`.

### Using Configuration Presets

The configuration is composable. You can select presets defined in the `conf/` directory.

Run a quick, small experiment using the `fast` sampler settings and `small` model/data:

```bash
uv run lambda-hat sampler=fast model=small data=small
```

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
uv run lambda-hat --multirun model.target_params=1000,5000,10000

# Compare base vs fast sampler settings
uv run lambda-hat --multirun sampler=base,fast
```

Combine sweeps (Cartesian product):

```bash
# 2 sizes x 2 sampler configs = 4 runs
uv run lambda-hat --multirun model.target_params=1000,5000 sampler=base,fast
```

Multi-run outputs are saved under `multirun/`.

## Documentation

- [Configuration Details](./docs/configuration.md)
- [Running on SLURM](./docs/parallelism.md)
- [BlackJAX Notes](./docs/blackjax.md)


## Background

* [[2308.12108] The Local Learning Coefficient: A Singularity-Aware Complexity Measure](https://arxiv.org/abs/2308.12108)
* [[2507.21449] From Global to Local: A Scalable Benchmark for Local Posterior Sampling](https://arxiv.org/abs/2507.21449)
* [singularlearningtheory.com](https://singularlearningtheory.com/)