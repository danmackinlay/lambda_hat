# Lambda-Hat (λ̂): Experiments in estimating the Local Learning Coefficient

In [Singular Learning Theory (SLT)](https://singularlearningtheory.com), the **Local Learning Coefficient (LLC)** quantifies the effective local dimensionality of a model around a trained optimum.

## Concept

The Local Learning Coefficient (LLC) measures the effective number of parameters a neural network *actually* learns from data. Lambda-Hat implements a teacher–student framework with a two-stage design:

* **Stage A**: Build and train a neural network target once, generating a reproducible **target artifact**.
* **Stage B**: Run multiple samplers (MCMC or variational) on the same target with different configurations.

This separation provides:

* **Reproducibility**: Same target ID = identical neural network weights and data
* **Efficiency**: Train expensive targets once, sample many times
* **Isolation**: Target configuration and sampler hyper-parameters are decoupled
* **Cost control**: Resource-intensive target building and cheaper sampling jobs can be optimized independently

In [Singular Learning Theory (SLT)](https://singularlearningtheory.com), the **Local Learning Coefficient (LLC)** quantifies the effective local dimensionality of a model around a trained optimum.
Estimating it can be tricky. That is what we explore here.

![](assets/readme/llc_convergence_combined.png)


* [[2308.12108] The Local Learning Coefficient: A Singularity-Aware Complexity Measure](https://arxiv.org/abs/2308.12108)
* [[2507.21449] From Global to Local: A Scalable Benchmark for Local Posterior Sampling](https://arxiv.org/abs/2507.21449)
* [singularlearningtheory.com](https://singularlearningtheory.com/)


This repo provides benchmark estimators of LLC on small but non-trivial neural networks, using standard industrial tooling:

* [BlackJAX](https://github.com/blackjax-devs/blackjax/tree/1.2.5) for MCMC sampling (HMC, MCLMC, SGLD)
* [ArviZ](https://python.arviz.org/) for diagnostics
* [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management
* [Haiku](https://github.com/haiku/haiku) for neural network definitions

**Supported samplers**: HMC, MCLMC, SGLD, VI (variational inference with MFA or Flow algorithms).
**Note**: Flow VI requires `uv sync --extra flowvi` but is currently broken with Parsl workflows (see docs/flow_prng_issue.md).

We target networks with dimension up to about $10^5$ which means we can ground-truth against classic samplers like HMC (which we expect to become non-viable in higher dimension or dataset size).
In this regime we can relu upon classic MCMC to tell us the “true” LLC rather than relying on analytic results for approximate networks such as Deep Linear Networks.

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

## Entrypoints

Lambda-Hat provides three command-line tools that implement the two-stage workflow. Parsl orchestrates these automatically, but they can also be invoked directly for debugging or custom workflows.

### `lambda-hat-build-target` (Stage A)

Builds a neural network target artifact: trains a model, saves parameters, data, and metadata.

```bash
uv run lambda-hat-build-target \
  --config-yaml config/composed.yaml \
  --target-id tgt_abc123 \
  --target-dir runs/targets/tgt_abc123
```

**Outputs:**
- `meta.json` — config snapshot, dimensions, L0, package versions
- `params.npz` — trained neural network parameters
- `data.npz` — training dataset (X, Y)

**Key features:**
- Content-addressed target IDs ensure reproducibility
- Precision mode (`jax_enable_x64`) recorded in metadata
- Reference loss L0 computed and stored for LLC estimation

### `lambda-hat-sample` (Stage B)

Runs a sampler (MCMC or variational) on a pre-built target artifact.

```bash
uv run lambda-hat-sample \
  --config-yaml config/sampler.yaml \
  --target-id tgt_abc123 \
  --run-dir runs/targets/tgt_abc123/run_hmc_ab12cd34
```

**Outputs:**
- `trace.nc` — ArviZ-compatible NetCDF trace with chains and diagnostics
- `analysis.json` — computed metrics (LLC estimate, ESS, R-hat, etc.)
- `diagnostics/` — trace plots, rank plots, convergence diagnostics

**Key features:**
- Precision guard: fails if sampler x64 setting mismatches target
- Automatic minibatching for SGLD-family samplers
- Parallel chain execution with JAX's vmap

### `lambda-hat-promote`

Utility for copying plots from run directories into stable locations for documentation or galleries.
It  searches under `runs/targets/**/run_{sampler}_*/diagnostics/`.

```bash
# Create an asset gallery with newest run per sampler
uv run lambda-hat-promote gallery \
  --runs-root runs \
  --samplers sgld,hmc,mclmc \
  --outdir runs/promotion \
  --snippet-out runs/promotion/gallery.md

# Copy specific plots
uv run lambda-hat-promote single \
  --runs-root runs \
  --samplers sgld \
  --outdir figures \
  --plot-name running_llc.png
```

These can be orchestrated automatically by adding the `--promote` flag to the Parsl workflow (see Orchestration section below).


---

## Orchestration

We use **Parsl** for the full pipeline. Parsl provides Python-native DAG execution with better support for dynamic parameter sweeps and HPC cluster integration.

### Quickstart

```bash
# Run locally (uses ThreadPoolExecutor)
uv run parsl-llc --local

# Run locally with promotion (generates galleries)
uv run parsl-llc --local --promote
```

### Editing experiments

* Edit `config/experiments.yaml` to add/remove targets and samplers.
* Parsl computes IDs and directories using the same logic; scripts do **not** invent paths.

### Promotion (opt-in)

Promotion generates asset galleries from sampling runs. It's opt-in via the `--promote` flag:

```bash
# Run workflow with promotion
uv run parsl-llc --local --promote

# Specify which plots to promote
uv run parsl-llc --local --promote \
    --promote-plots trace.png,llc_convergence_combined.png
```

### HPC Execution

For SLURM clusters, use the SLURM Parsl config:

```bash
# Run on SLURM cluster (auto-scales 0-50 jobs)
uv run parsl-llc --parsl-config parsl_config_slurm.py

# Customize Parsl config
# Edit parsl_config_slurm.py to adjust partition, walltime, resources
```

### Hyperparameter Optimization

**Optuna workflow** for automated hyperparameter tuning using Bayesian optimization:

```bash
# Optimize hyperparameters locally
uv run parsl-optuna --config config/optuna_demo.yaml --local

# Optimize on SLURM cluster
uv run parsl-optuna --config config/optuna_demo.yaml
```

**How it works:**
1. Computes HMC reference LLC for each problem (high-quality baseline)
2. Optimizes method hyperparameters (SGLD/VI/MCLMC) to minimize `|LLC - LLC_ref|`
3. Uses Optuna's TPE sampler for Bayesian search
4. Results written to `results/optuna_trials.parquet`

**Use cases:**
- Find optimal hyperparameters for your problem class
- Compare methods under fair time budgets
- Automate parameter tuning instead of manual sweeps

See [`docs/optuna_workflow.md`](docs/optuna_workflow.md) for detailed configuration and usage.

---

## Artifact Layout

**Standard workflow** (`workflows/parsl_llc.py`):
```
runs/
└── targets/
    ├── _catalog.jsonl               # registry of all targets
    └── tgt_abc123/                  # one target artifact
        ├── meta.json                # metadata (config, dims, precision, L0)
        ├── data.npz                 # training data
        ├── params.npz               # trained parameters
        ├── _runs.jsonl              # manifest of Stage-B runs
        ├── run_hmc_ab12cd34/        # one sampler run
        │   ├── trace.nc             # ArviZ trace
        │   ├── analysis.json        # metrics
        │   └── diagnostics/
        │       ├── trace.png
        │       └── rank.png
        ├── run_sgld_ef567890/
        └── run_mclmc_gh901234/
```

**Optuna workflow** (`workflows/parsl_optuna.py`):
```
artifacts/
├── problems/
│   └── p_abc123/
│       └── ref.json                 # HMC reference LLC
└── runs/
    └── p_abc123/
        └── vi/
            ├── r_def456/            # one trial
            │   ├── manifest.json    # trial hyperparameters
            │   └── metrics.json     # trial results
            └── r_ghi789/

results/
├── optuna_trials.parquet            # all trials aggregated
└── studies/
    └── optuna_llc/
        └── p_abc123:vi.pkl          # Optuna study (for resume)
```

Artifacts are written to `runs/...` (standard workflow) or `artifacts/...` (Optuna workflow). The sampler name is included in folder names as a human-useful facet; all other hyperparameters live in `analysis.json` or `metrics.json`.

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

## Further documentation

- [Configuration Details](./docs/configuration.md)
- [Running on SLURM](./docs/parallelism.md)
- [Hyperparameter Optimization with Optuna](./docs/optuna_workflow.md)
- [BlackJAX Notes](./docs/blackjax.md)
