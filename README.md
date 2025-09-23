# Local Learning Coefficient Sampler Benchmarks

In [Singular Learning Theory (SLT)](https://singularlearningtheory.com), the **Local Learning Coefficient (LLC)** quantifies the effective local dimensionality of a model around a trained optimum.

This repo provides benchmark estimators of LLC on small but non-trivial neural networks, using standard industrial tooling: [BlackJAX](https://github.com/blackjax-devs/blackjax/tree/1.2.5) for sampling and [ArviZ](https://python.arviz.org/) for diagnostics.

---

## Prerequisites

* **Python versions**

  * **SLURM/submitit:** Python **3.12** (cluster standard, at least on our cluster)
  * **Modal:** Python **3.11** (baked into the Modal image)
  * **Local:** Prefer **3.12** for parity with SLURM; 3.11 also works

* **Package manager:** [uv](https://github.com/astral-sh/uv)

---

## Quick Start (CPU example)

1. Create an environment:

   ```bash
   uv venv --python 3.12 && source .venv/bin/activate
   uv sync --extra cpu
   ```

2. Run a small experiment:

   ```bash
   uv run llc run --sampler sgld --preset=quick
   ```

3. Analyze the run:

   ```bash
   uv run llc analyze runs/<run_id> --which all --plots running_llc,rank,ess_evolution,autocorr,energy,theta
   ```

This generates diagnostic plots such as:

![SGLD running LLC](assets/readme/sgld_llc_running.png)
![HMC running LLC](assets/readme/hmc_llc_running.png)
![MCLMC running LLC](assets/readme/mclmc_llc_running.png)

---

## Interpreting Plots

* **Running LLC:** curves should stabilize and agree across chains; divergence = poor mixing.
* **Rank plots:** near-uniform → good; spikes → multimodality/non-convergence.
* **ESS evolution:** should grow and plateau; flat growth → high autocorrelation.
* **Energy (HMC/MCLMC):** distributions should look regular/tight.

More plots are available via `llc analyze` (rank, ESS, autocorrelation, theta traces, energy histograms).

---

## Running on Different Backends

All backends share the same flags (`--backend`, `--gpu-mode`, `--gpu-types`) via a unified executor.

### Local

```bash
uv sync --extra cpu
uv run llc run --backend=local --sampler sgld --preset=quick
```

### SLURM (Python 3.12 + CUDA 12)

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra slurm --extra cuda12
uv run llc run --backend=submitit --gpu-mode=vectorized \
  --slurm-partition=gpu --account=abc123 --sampler sghmc
```

### Modal (Python 3.11 inside image)

```bash
uv sync --extra modal
uv run llc run --backend=modal --gpu-mode=vectorized --sampler sghmc
```

See [docs/backends.md](docs/backends.md) for full setup (Modal + SLURM).

---

## Common Tasks

| Task                    | Command                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| Single SGHMC run        | `uv run llc run --sampler sghmc --preset=quick`                                           |
| Local sweep (8 workers) | `uv run llc sweep --workers=8`                                                            |
| Sweep with study YAML   | `uv run llc sweep --study study.yaml --backend=modal`                                     |
| Sweep samplers (JSON)   | `uv run llc sweep --sampler-grid='[{"name":"sgld","overrides":{"sgld_precond":"adam"}}]'` |
| SLURM sweep             | `uv run llc sweep --backend=submitit --gpu-mode=vectorized`                               |
| Modal sweep             | `uv run llc sweep --backend=modal`                                                        |
| Analyze saved run       | `uv run llc analyze runs/<run_id>`                                                        |
| Plot sweep results      | `uv run llc plot-sweep`                                                                   |
| Refresh README images   | `uv run llc promote-readme-images`                                                        |
| Debug with verbose mode | `uv run llc --verbose run --sampler sgld` (note: `--verbose` goes BEFORE subcommand)      |


## Debugging and Re-running Jobs

Every run and sweep job saves its **exact configuration** to `config.json` in the run directory. In addition, failed jobs now dump a `<run_id>_cfg.json` next to their SLURM logs. This makes it easy to see what parameters were passed and to retry them.

### Inspect config

```bash
cat runs/<run_id>/config.json         # for completed jobs
cat slurm_logs/<jobid>_cfg.json       # for failed SLURM jobs
```

### Repeat a run

You can re-run any job locally or remotely with the **same parameters**:

```bash
# From a run dir
uv run llc repeat --from-run runs/<run_id> --backend=local --gpu-mode=off

# From a dumped cfg JSON (e.g. from a failed sweep job)
uv run llc repeat --cfg-json slurm_logs/<jobid>_cfg.json --backend=submitit --gpu-mode=vectorized --account=abc123
```

This lets you quickly rerun failed SLURM jobs on your laptop to see the error, or resubmit them to the cluster without editing YAMLs.


---

## Concepts

* **Job** = (problem, sampler, seed)
* **Family** = jobs sharing (problem, seed) but differing by sampler
* **Sweep** = many jobs + one CSV summary (`llc_sweep_results.csv`)

Sweeps always run one sampler per job.
Attempts are made to cache identical jobs to avoid wating compute.

### Caching

The run hash includes the code version and **only the config fields relevant to the selected sampler**. Changing an HMC parameter does **not** invalidate cached SGLD runs (and vice-versa).

### Families

`run_family_id` groups runs by problem/data/model/seed and **ignores sampler and code version** so you can compare samplers and code upgrades within the same family.

---

## What's in a Run

Each run is **atomic** and executes **exactly one sampler**. The run directory contains a single NetCDF file named after the sampler (e.g. `sgld.nc`) and a `metrics.json` whose keys are prefixed with that sampler (e.g. `sgld_llc_mean`).

```text
runs/<run_id>/
├── config.json          # full configuration
├── metrics.json         # summary statistics (LLC mean/SE, ESS, WNV, timings)
├── L0.txt               # baseline loss at ERM
├── sgld.nc              # traces for SGLD (exactly one sampler per run)
└── analysis/            # generated by `llc analyze`
    ├── *_running_llc.png
    ├── *_rank.png
    └── ...
```

**Sample `metrics.json`:**

```json
{
  "sgld_llc_mean": 145.7,
  "sgld_llc_se": 8.15,
  "sgld_ess": 28.0,
  "sgld_wnv_time": 0.042,
  "sgld_timing_sampling": 12.83
}
```

---

## Defining Sweeps

### YAML Study Files

```yaml
# study.yaml
base:
  preset: quick
problems:
  - name: small
    overrides: {target_params: 2000}
  - name: large
    overrides: {target_params: 10000}
samplers:
  - name: sgld
    overrides: {sgld_precond: none}
  - name: sgld
    overrides: {sgld_precond: adam}
  - name: hmc
    overrides: {hmc_num_integration_steps: 10}
seeds: [0,1,2]
```

Run:

```bash
uv run llc sweep --backend=modal --study study.yaml
```

### JSON Grid Sweeps

```bash
uv run llc sweep --sampler-grid='[{"name":"sgld","overrides":{"sgld_precond":"adam"}}]'
```

---

## Efficiency Metrics (Advanced)

From saved traces we compute:

* **ESS/sec** — effective samples per second
* **ESS/FDE** — effective samples per full-data-equivalent gradient
* **WNV (time/FDE)** — variance × cost for fair comparison

Results are saved to `metrics.json` per run and `llc_sweep_results.csv` for sweeps.

---

## Installation

```bash
uv sync
uv sync --extra modal      # Modal support
uv sync --extra slurm      # SLURM support
```

---

## Features

* Unified CLI for end-to-end LLC estimation
* Consistent sampler interface: SGLD, SGHMC, HMC, MCLMC
* Three execution backends: Local, SLURM, Modal
* Configurable targets: ReLU/tanh/GeLU MLPs, analytical quadratic
* Full ArviZ diagnostics (ESS, R̂, autocorrelation, rank plots)
* Deterministic caching (reuses runs by config+code hash)

---

## Regenerating README Figures

There are two ways to refresh the images under `assets/readme/`:

### A) One-step (recommended)

Runs a full preset, generates plots, and updates README assets:

```bash
# Choose backend/gpu flags as needed (works with local, submitit, modal)
uv run llc showcase-readme

# Examples with different backends:
uv run llc showcase-readme --gpu-mode=vectorized  # Local GPU
uv run llc showcase-readme --backend=submitit --gpu-mode=vectorized --slurm-partition=gpu
uv run llc showcase-readme --backend=modal --gpu-mode=vectorized --gpu-types=H100

# With verbose logging for debugging:
uv run llc --verbose showcase-readme --backend=submitit --gpu-mode=vectorized --account=abc123
```

### B) Manual approach (advanced)

Useful if you already have a specific run directory:

1. **Analyze** an existing run to generate plots:
   ```bash
   uv run llc analyze runs/<run_id> --which all \
     --plots running_llc,rank,ess_evolution,autocorr,energy,theta
   ```

2. **Promote** curated images into the README assets:
   ```bash
   # With a specific run dir...
   uv run llc promote-readme-images runs/<run_id>

   # ...or let it pick the newest completed run automatically
   uv run llc promote-readme-images
   ```

**Command overview:**
* `showcase-readme`: run (full preset) → analyze → promote (all-in-one)
* `analyze`: only renders figures into `<run_dir>/analysis/`
* `promote-readme-images`: only copies curated images into `assets/readme/`

---

## Documentation

* [Backends (Modal/SLURM setup)](docs/backends.md)
* [Caching behavior](docs/caching.md)
* [Preconditioned SGLD options](docs/sgld-precond.md)
* [Target functions and data generators](docs/targets.md)
* [BlackJAX API notes](docs/blackjax.md)

---

## License

MIT.
