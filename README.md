# Lambda-Hat (λ̂): Two-Stage LLC Experiments

Lambda-Hat provides a streamlined framework for estimating the Local Learning Coefficient (LLC) using a **two-stage workflow**. It uses JAX/BlackJAX for MCMC sampling, Hydra for configuration management, and Haiku for neural network definitions.

## Concept

The Local Learning Coefficient (LLC) measures the effective number of parameters a neural network *actually* learns from data. Lambda-Hat implements a teacher–student framework with a two-stage design:

* **Stage A**: Build and train a neural network target once, generating a reproducible **target artifact**.
* **Stage B**: Run multiple MCMC samplers on the same target with different configurations.

This separation provides:

* **Reproducibility**: Same target ID = identical neural network weights and data
* **Efficiency**: Train expensive targets once, sample many times
* **Isolation**: Target configuration and sampler hyper-parameters are decoupled
* **Cost control**: Resource-intensive target building and cheaper sampling jobs can be optimized independently

---

## Installation

Requires Python 3.11+.

```bash
# Using uv (recommended):
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra cpu      # For CPU/macOS
uv sync --extra cuda12   # For CUDA 12 (Linux)

# Or using pip:
pip install .[cpu]       # For CPU/macOS
pip install .[cuda12]    # For CUDA 12 (Linux)
```

---

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

Each command outputs a target ID like `tgt_abcd1234`. The artifact contains:

* Neural network parameters (`theta`)
* Training data (`X`, `Y`)
* Model configuration and metadata
* Reference loss (`L0`) for LLC computation

Artifacts are saved under `runs/targets/tgt_<id>/`.

---

## Stage B: Sample

Run MCMC samplers on previously built targets. Supported samplers: **SGLD**, **HMC**, **MCLMC**.

### Single Sampler Runs

```bash
# Run HMC on a target
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=hmc

# Run SGLD with a custom step size
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=sgld sampler.sgld.step_size=1e-5

# Run MCLMC with more draws
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=mclmc sampler.mclmc.draws=5000
```

### Parameter Sweeps

Hydra multirun (`-m`) lets you sweep hyper-parameters:

```bash
# Compare all samplers on the same target
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=sgld,hmc,mclmc

# Sweep HMC step sizes
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=hmc \
    sampler.hmc.step_size=0.005,0.01,0.02

# Cross-product: 3 samplers × 3 step sizes = 9 runs
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=sgld,hmc,mclmc \
    sampler.hmc.step_size=0.005,0.01,0.02 sampler.sgld.step_size=1e-5,1e-4,1e-3
```

Results are saved under `runs/samples/<target_id>/<sampler>/run_<hash>/`.

---

## Quickstart

```bash
# Stage A: build target
uv run lambda-hat-build-target model=small data=small target.seed=42
# → prints: tgt_abc123

# Stage B: run samplers
uv run lambda-hat-sample target_id=tgt_abc123 sampler=sgld
uv run lambda-hat-sample target_id=tgt_abc123 sampler=hmc
uv run lambda-hat-sample target_id=tgt_abc123 sampler=mclmc

# Stage B: sweep HMC hyper-parameters
uv run lambda-hat-sample -m target_id=tgt_abc123 sampler=hmc \
    sampler.hmc.step_size=0.005,0.01 sampler.hmc.num_integration_steps=3,5
```

---

## Artifact Layout

```
runs/
├── targets/
│   ├── _catalog.jsonl               # registry of all targets
│   └── tgt_abc123/                  # one target artifact
│       ├── meta.json                # metadata (config, dims, precision, L0)
│       ├── data.npz                 # training data
│       └── params.npz               # trained parameters
└── samples/
    └── tgt_abc123/                  # all samples for this target
        ├── _index.jsonl             # registry of sampler runs
        ├── sgld/run_def456/         # one sampler run
        │   ├── trace.nc             # ArviZ trace (preferred)
        │   └── analysis.json        # metrics
        ├── hmc/run_ghi789/
        └── mclmc/run_jkl012/
```

---

## Reproducibility & Precision

* **Same target ID** = identical data and parameters
* **Precision guard**: mismatch between target build (`jax_enable_x64`) and sampling run → error
* **Metadata**: package versions and code SHA recorded in `meta.json`
* **Parameter shape check**: ensures forward function matches stored parameters

### JAX precision conventions

* **Target building**: typically float64 for stability
* **SGLD**: float32 for efficiency
* **HMC/MCLMC**: float64 for accuracy

---

## HPC Usage

Use Hydra’s Submitit launcher for SLURM clusters:

```bash
# Build multiple targets
uv run lambda-hat-build-target -m model=small,base target.seed=42,123

# Submit sampling sweeps to GPU nodes
uv run lambda-hat-sample -m target_id=tgt_abc123 sampler=sgld,hmc,mclmc \
    hydra/launcher=submitit_slurm hydra.launcher.gpus_per_node=1 \
    hydra.launcher.slurm.additional_parameters.account=OD-228158
```

---

## Asset Promotion

Lambda-Hat includes a utility to copy plots from sampling runs into a stable location for docs:

```bash
# Promote latest trace plots
uv run lambda-hat-promote runs_root=runs samplers=sgld,hmc,mclmc plot_name=trace.png

# Custom output
uv run lambda-hat-promote runs_root=runs samplers=sgld outdir=figures plot_name=running_llc.png
```

---

## Why Two Stages?

* **Orthogonality**: target vs. sampler configs are independent
* **Cost control**: build expensive targets once, sweep samplers cheaply
* **Reproducibility**: content-addressed target IDs guarantee identical experiments
* **Scalability**: works for big/pretrained models and HPC sweeps

## Further documentation

- [Configuration Details](./docs/configuration.md)
- [Running on SLURM](./docs/parallelism.md)
- [BlackJAX Notes](./docs/blackjax.md)