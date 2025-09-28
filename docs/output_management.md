# Output Layout

This project uses **Snakemake** for orchestration, creating a structured, content-addressed artifact layout. Unlike timestamp-based directories, artifacts are organized by deterministic IDs for reproducibility.

## Directory Structure

The `runs/` directory contains all experimental artifacts:

```
runs/
└── targets/
    ├── _catalog.jsonl               # Registry of all targets
    └── tgt_abc123456789/            # One target artifact
        ├── meta.json                # Metadata (config, dims, precision, L0)
        ├── data.npz                 # Training data (X, Y)
        ├── params.npz               # Trained parameters (θ*)
        ├── _runs.jsonl              # Manifest of Stage-B runs
        ├── run_hmc_xy789abc/        # One sampler run
        │   ├── trace.nc             # ArviZ trace (preferred format)
        │   ├── analysis.json        # LLC estimates and metrics
        │   └── diagnostics/
        │       ├── trace.png        # Trace plots
        │       ├── rank.png         # Rank plots
        │       └── running_llc.png  # Running LLC estimates
        ├── run_sgld_mn456def/       # Another sampler run
        │   ├── trace.nc
        │   ├── analysis.json
        │   └── diagnostics/
        └── run_mclmc_gh901234/      # Yet another sampler run
            └── ...
```

## Content-Addressed IDs

### Target IDs

**Target IDs** are deterministic SHA256 hashes of the Stage A configuration:

- `tgt_abc123456789` = hash(model + data + training + seed)
- **Same config → same ID → identical artifacts**
- Target building is expensive, so targets are cached and reused

### Run IDs

**Run IDs** are SHA1 hashes of the Stage B configuration:

- `xy789abc` = hash(sampler + hyperparameters + runtime_seed)
- **Same sampler config → same ID → reproducible runs**
- Multiple samplers can run on the same target

## Manifest Files

### `_catalog.jsonl`

Registry of all targets in the store:

```jsonl
{"target_id": "tgt_abc123456789", "created_at": 1672531200.0, "model": "base", "data": "small", "seed": 42}
{"target_id": "tgt_def987654321", "created_at": 1672531800.0, "model": "wide", "data": "large", "seed": 43}
```

### `_runs.jsonl`

Per-target manifest of sampling runs:

```jsonl
{"target_id": "tgt_abc123456789", "sampler": "hmc", "hyperparams": {...}, "walltime_sec": 120.5, "artifact_path": "runs/targets/tgt_abc123456789/run_hmc_xy789abc", "created_at": 1672531300.0}
{"target_id": "tgt_abc123456789", "sampler": "sgld", "hyperparams": {...}, "walltime_sec": 450.2, "artifact_path": "runs/targets/tgt_abc123456789/run_sgld_mn456def", "created_at": 1672531400.0}
```

## Key Files

### `meta.json` (Target Metadata)

Contains everything needed to reproduce and validate a target:

```json
{
  "target_id": "tgt_abc123456789",
  "created_at": 1672531200.0,
  "code_sha": "a1b2c3d4e5f6",
  "jax_enable_x64": true,
  "pkg_versions": {"jax": "0.7.1", "blackjax": "1.2.5", ...},
  "seed": 42,
  "model_cfg": {"depth": 3, "widths": [300, 200, 100], "activation": "relu", ...},
  "data_cfg": {"n_data": 20000, "noise_scale": 0.1, ...},
  "training_cfg": {"optimizer": "adam", "steps": 5000, ...},
  "dims": {"n": 20000, "d": 32, "p": 93501},
  "metrics": {"L0": 0.045123}
}
```

### `analysis.json` (Run Results)

Contains LLC estimates and diagnostics for one sampling run:

```json
{
  "llc_mean": 12.34,
  "llc_std": 0.56,
  "ess": 1234.5,
  "rhat": 1.01,
  "total_fges": 10000,
  "walltime_sec": 120.5,
  "sampler_config": {"draws": 1000, "warmup": 200, ...}
}
```

## Aggregating Results

Use Python to aggregate results across multiple runs:

```python
import json
import pandas as pd
from pathlib import Path

def aggregate_results(runs_root="runs"):
    runs_path = Path(runs_root)
    results = []

    for target_dir in (runs_path / "targets").iterdir():
        if not target_dir.is_dir() or not target_dir.name.startswith("tgt_"):
            continue

        # Load target metadata
        meta_file = target_dir / "meta.json"
        if not meta_file.exists():
            continue

        with open(meta_file) as f:
            meta = json.load(f)

        # Find all sampling runs
        for run_dir in target_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            analysis_file = run_dir / "analysis.json"
            if not analysis_file.exists():
                continue

            with open(analysis_file) as f:
                analysis = json.load(f)

            # Combine target and run information
            record = {
                "target_id": meta["target_id"],
                "model": meta["model_cfg"].get("depth", "unknown"),
                "n_data": meta["data_cfg"]["n_data"],
                "target_params": meta["dims"]["p"],
                "L0": meta["metrics"]["L0"],
                "seed": meta["seed"],
                "sampler": run_dir.name.split("_")[1],  # Extract from run_hmc_xyz
                "llc_mean": analysis["llc_mean"],
                "llc_std": analysis["llc_std"],
                "ess": analysis["ess"],
                "rhat": analysis["rhat"],
                "walltime_sec": analysis["walltime_sec"],
            }
            results.append(record)

    return pd.DataFrame(results)

# Usage
df = aggregate_results()
print(df.groupby(["sampler", "n_data"])["llc_mean"].describe())
```

## Reproducibility

### Exact Reproduction

Same `config/experiments.yaml` → same target and run IDs → identical results:

```bash
# These runs are deterministic and cached
uv run snakemake runs/targets/tgt_abc123456789/meta.json
uv run snakemake runs/targets/tgt_abc123456789/run_hmc_xy789abc/analysis.json
```

### Forcing Reruns

Force rebuild of targets or sampling runs:

```bash
# Force rebuild specific target
uv run snakemake --forcerun build_target runs/targets/tgt_abc123456789/meta.json

# Force rerun specific sampler
uv run snakemake --forcerun run_sampler runs/targets/tgt_abc123456789/run_hmc_xy789abc/analysis.json

# Force rerun all samplers (but not targets)
uv run snakemake --forcerun run_sampler -j 4
```

### Partial Runs

Run specific outputs:

```bash
# Build only targets (no sampling)
uv run snakemake "runs/targets/*/meta.json" -j 4

# Run only HMC (no SGLD)
uv run snakemake "runs/targets/*/run_hmc_*/analysis.json" -j 4

# Run specific target-sampler combination
uv run snakemake runs/targets/tgt_abc123456789/run_hmc_xy789abc/analysis.json
```

## Storage Considerations

- **Targets are expensive**: Neural network training, large datasets
- **Runs are cheaper**: MCMC on pre-trained targets
- **Traces can be large**: Consider `--profile` with storage limits on HPC
- **Promotion**: Use `lambda-hat-promote` to copy key plots to stable locations

Example storage usage:
```
runs/targets/tgt_abc123456789/
├── meta.json           # ~10 KB
├── data.npz           # ~10 MB (for n_data=20k)
├── params.npz         # ~1 MB (for 100k params)
└── run_hmc_xy789abc/
    ├── trace.nc       # ~50 MB (1000 draws × 100k params)
    └── analysis.json  # ~5 KB
```