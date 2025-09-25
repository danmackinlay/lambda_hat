# Lambda-Hat (λ̂): Two-Stage LLC Experiments

Lambda-Hat provides a streamlined framework for estimating the Local Learning Coefficient (LLC) using a two-stage workflow. It uses JAX/BlackJAX for MCMC sampling, Hydra for configuration management, and Haiku for neural network definitions.

## Concept

The Local Learning Coefficient (LLC) measures the effective number of parameters that a neural network "learns" from data. Lambda-Hat implements a teacher-student framework with a two-stage design:

**Stage A**: Build and train a neural network target once, generating a reproducible target artifact
**Stage B**: Run multiple MCMC samplers on the same target with different configurations

This separation provides:
- **Reproducibility**: Same target ID = identical neural network weights and data
- **Efficiency**: Train expensive targets once, sample many times
- **Isolation**: Target configuration and sampler hyperparameters are decoupled
- **Cost Control**: Expensive target building vs. cheaper sampling jobs can be optimized separately

## Installation

Requires Python 3.11+.

```bash
# Using uv (recommended):
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra cpu     # For CPU/macOS
uv sync --extra cuda12  # For CUDA 12 (Linux)

# Or using pip:
pip install .[cpu]      # For CPU/macOS
pip install .[cuda12]   # For CUDA 12 (Linux)
```

## Stage A: Build Target

Build and train the neural network target once. This creates a content-addressed artifact with a deterministic ID.

```bash
# Build a small target for testing
uv run lambda-hat-build-target model=small data=small target.seed=42

# Build a larger target for production experiments
uv run lambda-hat-build-target model=base data=base target.seed=123

# Override specific parameters
uv run lambda-hat-build-target model.depth=5 model.target_params=10000 data.n_data=5000
```

Each command outputs a target ID like `tgt_abcd1234`. The target artifact contains:
- Neural network parameters (`theta`)
- Training data (`X`, `Y`)
- Model configuration and metadata
- Reference loss (`L0`) for LLC computation

Artifacts are saved under `runs/targets/tgt_<id>/` with content-based deduplication.

## Stage B: Sample

Run MCMC samplers on previously built targets. All samplers (SGLD, HMC, MCLMC) can use the same target.

### Single Sampler Runs

```bash
# Run HMC on a specific target
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=hmc

# Run SGLD with custom step size
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=sgld sampler.sgld.step_size=1e-5

# Run MCLMC with more samples
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=mclmc sampler.mclmc.draws=5000
```

### Parameter Sweeps

Use Hydra's multirun (`-m`) for parameter sweeps:

```bash
# Compare all three samplers on the same target
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=sgld,hmc,mclmc

# HMC step size sweep
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=hmc \
    sampler.hmc.step_size=0.005,0.01,0.02

# Cross-product: 3 samplers × 3 step sizes = 9 runs
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=sgld,hmc,mclmc \
    sampler.hmc.step_size=0.005,0.01,0.02 sampler.sgld.step_size=1e-5,1e-4,1e-3
```

Results are saved under `runs/samples/<target_id>/<sampler>/run_<hash>/` with trace files and analysis.

## Two-Stage Quickstart

Copy-paste commands to get started:

```bash
# Stage A: Build target
uv run lambda-hat-build-target model=small data=small target.seed=42
# → outputs: tgt_abc123 (example)

# Stage B: Run samplers
uv run lambda-hat-sample target_id=tgt_abc123 sampler=sgld
uv run lambda-hat-sample target_id=tgt_abc123 sampler=hmc
uv run lambda-hat-sample target_id=tgt_abc123 sampler=mclmc

# Stage B: Parameter sweep
uv run lambda-hat-sample -m target_id=tgt_abc123 sampler=hmc \
    sampler.hmc.step_size=0.005,0.01 sampler.hmc.num_integration_steps=3,5
```

## Artifacts

The artifact layout provides organized storage:

```
runs/
├── targets/
│   ├── _catalog.jsonl              # Target registry
│   └── tgt_abc123/                 # Target artifact
│       ├── meta.json               # Metadata (model config, L0, etc.)
│       ├── data.npz                # Training data (X, Y)
│       └── params.npz              # Trained parameters (theta)
└── samples/
    └── tgt_abc123/                 # Samples for target
        ├── _index.jsonl            # Sample run registry
        ├── sgld/run_def456/        # SGLD results
        │   ├── trace.nc            # ArviZ trace file
        │   └── analysis.json       # LLC metrics
        ├── hmc/run_ghi789/         # HMC results
        └── mclmc/run_jkl012/       # MCLMC results
```

The `_index.jsonl` manifest records hyperparameters, runtime, and metrics for each sampling run, enabling systematic analysis across parameter sweeps.

## Reproducibility & Precision

### Reproducibility Checklist
- ✅ Use the same `target_id` across sampling runs
- ✅ Check that JAX precision matches between target and sampling (`jax_enable_x64`)
- ✅ Verify package versions recorded in target metadata
- ✅ Ensure parameter shapes validate before sampling

### JAX Precision
The system automatically handles mixed precision:
- **Target building**: Uses `jax_enable_x64=true` for accurate training
- **SGLD sampling**: Uses `float32` for efficiency
- **HMC/MCLMC sampling**: Uses `float64` for accuracy

Precision mismatches between target and sampling stages are automatically detected and will raise an error.

## HPC Usage

Use Hydra's Submitit launcher for SLURM clusters:

```bash
# Build targets on login node or submit as job
uv run lambda-hat-build-target -m model=small,base target.seed=42,123

# Submit sampling jobs to SLURM
uv run lambda-hat-sample -m target_id=tgt_abc123 sampler=sgld,hmc,mclmc \
    hydra/launcher=submitit_slurm hydra.launcher.partition=gpu
```

The two-stage design is particularly valuable for HPC: expensive target building can use different resources than massively parallel sampling sweeps.

## Asset Promotion

Lambda-Hat includes a utility to promote latest plots from sampling runs to a stable `assets/` directory:

```bash
# Promote latest trace plots for documentation
uv run lambda-hat-promote runs_root=runs samplers=sgld,hmc,mclmc plot_name=trace.png

# Custom output directory
uv run lambda-hat-promote runs_root=runs samplers=sgld outdir=figures plot_name=running_llc.png
```

This finds the most recent sampling run for each sampler and copies the specified plot to `assets/{sampler}.png`.

## Why Two Stages?

Lambda-Hat uses a two-stage design for several key benefits:

- **Orthogonality**: Target configuration (model, data, training) is completely separate from sampling configuration (chains, step sizes, warmup)
- **Cost Control**: Expensive target building can use different compute resources than massively parallel sampling sweeps
- **Reproducibility**: Content-addressed target IDs ensure identical experiments across runs and users
- **Efficiency**: Build complex targets once, then run dozens of sampler configurations without retraining

## Documentation

- [Configuration Details](./docs/configuration.md)
- [Running on SLURM](./docs/parallelism.md)
- [BlackJAX Notes](./docs/blackjax.md)