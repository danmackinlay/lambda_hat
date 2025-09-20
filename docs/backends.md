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

# Single run with specific GPU types
uv run python -m llc run --backend=modal --gpu-mode=vectorized --gpu-types=H100,A100

# Sweep (one job per sampler - default behavior)
uv run python -m llc sweep --backend=modal --gpu-types=L40S

# Sweep (legacy: multiple samplers per job)
uv run python -m llc sweep --backend=modal --no-split-samplers
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

### Split Samplers (Default)

Sweeps always run one sampler per job (default). This makes jobs shorter, retries independent, and analysis cleaner:

**Benefits:**
- **Shorter jobs** → fewer timeout issues
- **Independent retries** → one sampler failure doesn't block others
- **Max concurrency** → better autoscaler utilization
- **Cleaner failure attribution** → know exactly which sampler failed

**How it works:** Each configuration is expanded into one job per sampler. All jobs in a "family" use the same dataset/ERM (same `cfg.seed`) but run different samplers. Results include a `family_id` column for grouping.

### GPU Selection

Use `--gpu-types` to specify preferred GPU types with fallbacks:

```bash
# Single GPU type
--gpu-types=H100

# Multiple types (priority order)
--gpu-types=H100,A100,L40S
```

**Valid GPU types:** H100, A100, L40S, T4, A10G

If invalid types are specified, the system warns and falls back to L40S.

### Artifact Management

**Automatic download**: When you run `llc run/sweep --backend=modal`, runs are automatically downloaded to `./runs/<run_id>/` as each job completes.

**Manual retrieval**: For browsing or recovering old runs from the Modal volume use `llc pull-runs`.

### Environment Variables

Advanced users can tune Modal behavior via environment variables:

```bash
# GPU selection (set by --gpu-types flag)
LLC_MODAL_GPU_LIST=H100,A100,L40S

# Timeout and retry settings
LLC_MODAL_TIMEOUT_S=10800          # 3 hours default
LLC_MODAL_MAX_RETRIES=3
LLC_MODAL_BACKOFF=2.0
LLC_MODAL_INITIAL_DELAY_S=10

# Funding protection: client-side hang timeout
LLC_MODAL_CLIENT_HANG_TIMEOUT_S=180  # 3 minutes default
```

These are automatically set by CLI flags but can be overridden for advanced use cases.

### Funding Protection

Modal jobs can hang indefinitely if your account runs out of funds, since Modal's function timeouts exclude scheduling time. LLC includes automatic protection:

**Preflight check:** Every run/sweep starts with a quick `ping()` to detect funding issues immediately instead of hanging.

**Scheduling watchdog:** Client-side timeout (default: 3 minutes) detects stalled scheduling and fails fast with clear guidance.

**Error handling:** Funding failures are logged to `llc_sweep_errors.csv` with clear error messages.

### Troubleshooting

* **"Modal preflight failed: likely out of funds":** Top up your Modal balance or enable auto-recharge.
* **"call did not start within 180s":** Scheduling stalled - check Modal balance and adjust `LLC_MODAL_CLIENT_HANG_TIMEOUT_S` if needed.
* **Timeouts not taking effect:** Timeouts are tunable via `LLC_MODAL_TIMEOUT_S` environment variable (default: 3 hours).
* **Runs missing locally:** Fetch from the volume with `llc pull-runs <run_id>`.
* **GPU allocation issues:** Check `--gpu-types` values against the allowlist: H100, A100, L40S, T4, A10G.

### Cleanup

* Remove old run folders from the volume: `uv run modal volume rm llc-runs /runs/<run-id>`
* Stop the app (rare): `uv run modal app stop llc-experiments`

## SLURM (Submitit)

### Usage

```bash
# Single run with GPU
uv run python -m llc run --backend=submitit --gpu-mode=vectorized \
  --slurm-partition=gpu --timeout-min=180

# Sweep (one job per sampler - default behavior)
uv run python -m llc sweep --backend=submitit \
  --gpu-mode=vectorized --slurm-partition=gpu --timeout-min=180
```

### Submitit Configuration

Control SLURM job parameters via CLI flags:

```bash
--slurm-partition=gpu          # SLURM partition
--timeout-min=180              # Job timeout (default: 3 hours)
--cpus=4                       # CPUs per task (default: 4)
--mem-gb=16                    # Memory in GB (default: 16)
--slurm-signal-delay-s=120     # Grace period before kill (default: 120s)
```

**GPU handling:** Use `--gpu-mode=vectorized` or `--gpu-mode=sequential` to automatically request 1 GPU per job. Use `--gpu-mode=off` for CPU-only jobs.

### Error Handling

Submitit jobs now return structured status like Modal:
- Successful jobs: `{"status": "ok", ...}`
- Failed jobs: `{"status": "error", "error_type": "...", "stage": "...", ...}`

Failed jobs are logged to `llc_sweep_errors.csv` with full error details.

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