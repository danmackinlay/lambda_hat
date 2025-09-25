# Lambda-Hat (λ̂): LLC Experiments in estimating the Local Learning Coefficient

In [Singular Learning Theory (SLT)](https://singularlearningtheory.com), the **Local Learning Coefficient (LLC)** quantifies the effective local dimensionality of a model around a trained optimum.

This repo provides benchmark estimators of LLC on small but non-trivial neural networks, using standard industrial tooling:

* [BlackJAX](https://github.com/blackjax-devs/blackjax/tree/1.2.5) for sampling
* [ArviZ](https://python.arviz.org/) for diagnostics,
* [Hydra](https://hydra.cc/) for configuration management
* [Haiku](https://github.com/haiku/haiku) for neural network definitions.


## Concept

The Local Learning Coefficient (LLC) measures the effective number of parameters a neural network *actually* learns from data. Lambda-Hat implements a teacher–student framework with a two-stage design:

* **Stage A**: Build and train a neural network target once, generating a reproducible **target artifact**.
* **Stage B**: Run multiple MCMC samplers on the same target with different configurations.

This separation provides:

* **Reproducibility**: Same target ID = identical neural network weights and data
* **Efficiency**: Train expensive targets once, sample many times
* **Isolation**: Target configuration and sampler hyper-parameters are decoupled
* **Cost control**: Resource-intensive target building and cheaper sampling jobs can be optimized independently


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

## Quickstart (N x M Workflow)

The configuration-driven workflow enables  N (Targets) × M (Samplers) sweeps:

```bash
# Define the experiment space (Example: N=4 targets, M=2 samplers)
TARGET_SWEEP="model=small,base target.seed=42,43"
SAMPLER_SWEEP="sampler=hmc,sgld"

# 1. Ensure Targets Exist (Idempotent Build)
# Launches N=4 jobs. If a target already exists, the job finishes instantly.
uv run lambda-hat-build-target -m $TARGET_SWEEP

# 2. Run the Workflow Sweep (N × M = 8 jobs)
# Uses the unified configuration to dynamically calculate the target_id for each run.
uv run lambda-hat-workflow -m $TARGET_SWEEP $SAMPLER_SWEEP
```

This approach eliminates manual target ID harvesting and provides fully automated N × M experiments.

---

## Stage A: Build Target

Build and train the neural network target. This process is **idempotent**: if a target matching the configuration already exists, the build is skipped.

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

### Configuration-Driven Sweeps (N × M)

See the Quickstart above. Use `lambda-hat-workflow` to sweep across target definitions and sampler configurations simultaneously.

### Explicit Target ID (Legacy/Manual)

Use `lambda-hat-sample` if you want to run samplers on a specific, known `target_id`, bypassing the dynamic calculation.

```bash
# Run HMC on a specific target
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=hmc

# Run SGLD with a custom step size
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=sgld sampler.sgld.step_size=1e-5

# Run MCLMC with more draws
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=mclmc sampler.mclmc.draws=5000
```

#### Parameter Sweeps

Hydra multirun (`-m`) lets you sweep hyper-parameters:

```bash
# Compare all samplers on the same target
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=sgld,hmc,mclmc

# Sweep HMC step sizes
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=hmc \
    sampler.hmc.step_size=0.005,0.01,0.02

# Sweep samplers on a single specific target
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=sgld,hmc,mclmc
```

Results are saved under `runs/samples/<target_id>/<sampler>/run_<hash>/`.

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