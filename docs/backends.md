# Backend Configuration

This document covers setup and usage for Modal and SLURM backends.

## Modal Serverless

### One-time Setup

```bash
uv run modal token new                   # authenticate
uv run modal volume create llc-runs      # optional; code can create it on first run
```

### Usage

```bash
# Single run
uv run python -m llc run --backend=modal --preset=quick

# Sweep (traditional: one job per config)
uv run python -m llc sweep --backend=modal

# Sweep (split samplers: one job per sampler for better concurrency)
uv run python -m llc sweep --backend=modal --split-samplers
```

### How it Works

We use **object-based Modal function imports** for execution. The local client imports `run_experiment_remote` from `modal_app.py`, which auto-deploys current code.

`modal_app.py` defines resources/timeouts/volumes on the decorator:

```python
app = modal.App("llc-experiments", image=image)

@app.function(
    timeout=3*60*60,  # generous 3 hours
    volumes={"/runs": runs_volume},
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=10.0)
)
def run_experiment_remote(cfg_dict: dict) -> dict:
    ...
```

### Split Samplers (Recommended)

Use `--split-samplers` to run each sampler (SGLD, HMC, MCLMC) as a separate Modal job:

**Benefits:**
- **Shorter jobs** → fewer timeout issues
- **Independent retries** → one sampler failure doesn't block others
- **Max concurrency** → better autoscaler utilization
- **Cleaner failure attribution** → know exactly which sampler failed

**How it works:** Each configuration is expanded into one job per sampler. All jobs in a "family" use the same dataset/ERM (same `cfg.seed`) but run different samplers. Results include a `family_id` column for grouping.

### Artifact Management

**Automatic download**: When you run `llc run/sweep --backend=modal`, runs are automatically downloaded to `./runs/<run_id>/` as each job completes.

**Manual retrieval**: For browsing or recovering old runs from the Modal volume use `llc pull-runs`.

### Troubleshooting

* **Timeouts not taking effect:** Remember timeouts are set in the decorator in `modal_app.py` (we use generous defaults), not per-call flags.
* **Runs missing locally:** Fetch from the volume with `llc pull-runs <run_id>`.

### Cleanup

* Remove old run folders from the volume: `uv run modal volume rm llc-runs /runs/<run-id>`
* Stop the app (rare): `uv run modal app stop llc-experiments`

## SLURM (Submitit)

### Usage

```bash
uv run python -m llc sweep --backend=submitit \
  --partition=gpu --gpus=1 --timeout-min=60
```

Add your cluster-specific submitit parameters as needed.

### Installation

```bash
uv sync --extra slurm
```

## Volume Management

**Modal volume name:** `llc-runs`

The pipeline writes runs under:
- **Local:** `runs/<run_id>`
- **Modal:** `/runs/<run_id>` on the volume

Use `--no-skip` to force recompute; by default identical config+code uses the cached run (see [caching behavior](caching.md)).