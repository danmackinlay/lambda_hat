# Lambda-Hat (λ̂): Experiments in estimating the Local Learning Coefficient

In [Singular Learning Theory (SLT)](https://singularlearningtheory.com), the **Local Learning Coefficient (LLC)** quantifies the effective local dimensionality of a model around a trained optimum.

This repo provides benchmark estimators of LLC on small but non-trivial neural networks, using standard industrial tooling:

* [BlackJAX](https://github.com/blackjax-devs/blackjax/tree/1.2.5) for sampling
* [ArviZ](https://python.arviz.org/) for diagnostics,
* [Snakemake](https://snakemake.github.io/) for workflow orchestration
* [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management
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

In [Singular Learning Theory (SLT)](https://singularlearningtheory.com), the **Local Learning Coefficient (LLC)** quantifies the effective local dimensionality of a model around a trained optimum.

This repo provides benchmark estimators of LLC on small but non-trivial neural networks, using standard industrial tooling:

* [BlackJAX](https://github.com/blackjax-devs/blackjax/tree/1.2.5) for sampling
* [ArviZ](https://python.arviz.org/) for diagnostics,
* [Hydra](https://hydra.cc/) for configuration management
* [Haiku](https://github.com/haiku/haiku) for neural network definitions.

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

## Orchestration

We use **Snakemake** for the full pipeline. OmegaConf parses our YAML configs.

### Quickstart

```bash
# Preview the DAG and outputs
uv run snakemake -n

# Run locally (4 cores)
uv run snakemake -j 4
```

### Editing experiments

* Edit `config/experiments.yaml` to add/remove targets and samplers.
* Snakemake computes IDs and directories; scripts do **not** invent paths.

### Forcing & targeting

```bash
# Force re-run a rule and everything downstream
uv run snakemake --forcerun run_sampler -j 8

# Run a specific output
uv run snakemake runs/targets/tgt_abcdef123456/run_hmc_12ab34cd/analysis.json
```

### Linting

Before committing any changes:

```bash
uv ruff
```

### HPC

Use your Snakemake profile:

```bash
uv run snakemake --profile slurm -j 100
```

(Adjust HPC section to your environment if you keep a profile.)
(Note we are not on the cluster right now so testing for this can be deferred).

---

## Artifact Layout

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
        │   ├── trace.nc             # ArviZ trace (preferred)
        │   ├── analysis.json        # metrics
        │   └── diagnostics/
        │       ├── trace.png
        │       └── rank.png
        ├── run_sgld_ef567890/
        └── run_mclmc_gh901234/
```

Artifacts are written to `runs/...` directly. The sampler name is included in the folder name because it's a low-cardinality, human-useful facet; all other hyperparameters live in `analysis.json`.

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

## Asset Promotion

Lambda-Hat includes a utility to copy plots from sampling runs into a stable location for docs. It now searches under `runs/targets/**/run_{sampler}_*/diagnostics/`.

```bash
# Promote newest run of each sampler into runs/promotion and write a README snippet
uv run lambda-hat-promote gallery --runs-root runs --samplers sgld,hmc,mclmc \
  --plot-name trace.png --outdir runs/promotion --snippet-out runs/promotion/gallery_snippet.md

# Copy newest plots without snippet
uv run lambda-hat-promote single --runs-root runs --samplers sgld --outdir figures \
  --plot-name running_llc.png
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