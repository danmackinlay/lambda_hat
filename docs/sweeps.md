# Parameter Sweeps with Snakemake

This document explains how to design and execute parameter sweeps using **Snakemake's configuration-driven approach**. Unlike timestamp-based execution, Lambda-Hat creates deterministic N × M experiment matrices defined in `config/experiments.yaml`.

## Core Concept: N × M Sweeps

Lambda-Hat automatically generates a **Cartesian product** of targets and samplers from your experiment configuration:

- **N targets** (different model/data/seed combinations)
- **M samplers** (different MCMC algorithms and hyperparameters)
- **Total: N × M sampling jobs** (plus N target-building jobs)

### Example Configuration

Define your sweep in `config/experiments.yaml`:

```yaml
store_root: "runs"
jax_enable_x64: true

targets:
  - { model: small, data: small, teacher: _null, seed: 42 }
  - { model: base,  data: base,  teacher: _null, seed: 43 }
  - { model: wide,  data: large, teacher: _null, seed: 44 }

samplers:
  - { name: hmc }
  - { name: sgld, overrides: { step_size: 1e-6 } }
  - { name: mclmc, overrides: { draws: 5000 } }
```

**Result**: 3 targets × 3 samplers = 9 sampling jobs + 3 target-building jobs = 12 total jobs.

### Execute the Sweep

```bash
# Preview the entire DAG
uv run snakemake -n

# Execute locally with 4 parallel jobs
uv run snakemake -j 4

# Execute on HPC cluster
uv run snakemake --profile slurm -j 100
```

## Designing Sweeps

### 1. Model Architecture Sweeps

Create multiple model presets or use overrides:

```yaml
targets:
  # Using different presets
  - { model: small, data: base, seed: 42 }
  - { model: base,  data: base, seed: 42 }
  - { model: wide,  data: base, seed: 42 }

  # Using overrides for one-off variations
  - { model: base, data: base, seed: 42,
      overrides: {
        model: { activation: identity, depth: 6 },
        training: { steps: 15000 }
      }}
```

### 2. Data Scaling Sweeps

Vary dataset size and characteristics:

```yaml
targets:
  - { model: base, data: small, seed: 42 }   # n_data=5000
  - { model: base, data: base,  seed: 42 }   # n_data=20000
  - { model: base, data: large, seed: 42 }   # n_data=100000

  # Custom data configuration
  - { model: base, data: base, seed: 42,
      overrides: {
        data: { n_data: 50000, noise_scale: 0.05 }
      }}
```

### 3. Seed Sweeps for Statistical Robustness

Multiple random seeds for the same configuration:

```yaml
targets:
  - { model: base, data: base, seed: 42 }
  - { model: base, data: base, seed: 43 }
  - { model: base, data: base, seed: 44 }
  - { model: base, data: base, seed: 45 }
  - { model: base, data: base, seed: 46 }
```

### 4. Sampler Hyperparameter Sweeps

Explore different MCMC configurations:

```yaml
samplers:
  # HMC with different step counts
  - { name: hmc }  # Uses defaults from hmc.yaml
  - { name: hmc, overrides: { num_integration_steps: 5 } }
  - { name: hmc, overrides: { num_integration_steps: 20 } }

  # SGLD with different step sizes
  - { name: sgld, overrides: { step_size: 1e-7 } }
  - { name: sgld, overrides: { step_size: 1e-6 } }
  - { name: sgld, overrides: { step_size: 1e-5 } }

  # Multiple chains
  - { name: hmc, overrides: { chains: 8 } }
```

## Advanced Patterns

### 1. Mixed Precision Experiments

Compare float32 vs float64 performance:

```yaml
# High precision (recommended for HMC/MCLMC)
jax_enable_x64: true
targets:
  - { model: base, data: base, seed: 42 }
samplers:
  - { name: hmc }
  - { name: mclmc }
```

Create a separate experiment file for float32:

```yaml
# config/experiments_f32.yaml
jax_enable_x64: false
targets:
  - { model: base, data: base, seed: 42 }
samplers:
  - { name: sgld }  # SGLD works well with float32
```

Run separately:
```bash
uv run snakemake -j 4  # Uses default experiments.yaml
uv run snakemake -j 4 --configfile config/experiments_f32.yaml
```

### 2. Targeted Experiments

Create focused experiments for specific research questions:

```yaml
# config/experiments_wide_vs_deep.yaml
targets:
  # Wide networks
  - { model: base, data: base, seed: 42,
      overrides: { model: { depth: 2, target_params: 50000 } }}
  - { model: base, data: base, seed: 43,
      overrides: { model: { depth: 2, target_params: 50000 } }}

  # Deep networks
  - { model: base, data: base, seed: 42,
      overrides: { model: { depth: 8, target_params: 50000 } }}
  - { model: base, data: base, seed: 43,
      overrides: { model: { depth: 8, target_params: 50000 } }}

samplers:
  - { name: hmc }
  - { name: sgld }
```

### 3. Ablation Studies

Systematic feature removal:

```yaml
# config/experiments_ablation.yaml
targets:
  # Baseline
  - { model: base, data: base, seed: 42 }

  # No bias terms
  - { model: base, data: base, seed: 42,
      overrides: { model: { bias: false } }}

  # Linear networks (identity activation)
  - { model: base, data: base, seed: 42,
      overrides: { model: { activation: identity } }}
```

## Partial Execution and Debugging

### Run Specific Subsets

```bash
# Build targets only (no sampling)
uv run snakemake "runs/targets/*/meta.json" -j 4

# Run only HMC samplers
uv run snakemake "runs/targets/*/run_hmc_*/analysis.json" -j 4

# Run specific target across all samplers
uv run snakemake "runs/targets/tgt_abc123456789/run_*/analysis.json" -j 4

# Single target-sampler combination
uv run snakemake runs/targets/tgt_abc123456789/run_hmc_xy789abc/analysis.json
```

### Debugging Failed Jobs

```bash
# See what jobs would run
uv run snakemake -n --detailed-summary

# Check for failed/incomplete jobs
uv run snakemake --summary

# Rerun only failed jobs
uv run snakemake --rerun-incomplete -j 4

# Force rerun specific rules
uv run snakemake --forcerun build_target -j 4
uv run snakemake --forcerun run_sampler -j 4
```

### Incremental Execution

Add new configurations without rebuilding existing ones:

```yaml
# Add to existing experiments.yaml
targets:
  - { model: small, data: small, seed: 42 }  # Already built
  - { model: base,  data: base,  seed: 43 }  # Already built
  - { model: huge,  data: large, seed: 44 }  # NEW - will be built

samplers:
  - { name: hmc }     # Already run on existing targets
  - { name: sgld }    # Already run on existing targets
  - { name: mclmc }   # NEW - will run on all targets
```

Snakemake will automatically:
1. Skip building existing targets
2. Skip running existing sampler combinations
3. Build the new target
4. Run the new sampler on all targets (including existing ones)

## Results Aggregation

After sweep completion, analyze results programmatically:

```python
import pandas as pd
from pathlib import Path
import json

def analyze_sweep_results(runs_root="runs"):
    results = []

    for target_dir in (Path(runs_root) / "targets").iterdir():
        if not target_dir.name.startswith("tgt_"):
            continue

        # Load target metadata
        with open(target_dir / "meta.json") as f:
            meta = json.load(f)

        # Collect all sampling runs
        for run_dir in target_dir.iterdir():
            if not run_dir.name.startswith("run_"):
                continue

            with open(run_dir / "analysis.json") as f:
                analysis = json.load(f)

            results.append({
                "target_id": meta["target_id"],
                "model_depth": meta["model_cfg"]["depth"],
                "n_data": meta["data_cfg"]["n_data"],
                "target_params": meta["dims"]["p"],
                "seed": meta["seed"],
                "sampler": run_dir.name.split("_")[1],
                "llc_mean": analysis["llc_mean"],
                "llc_std": analysis["llc_std"],
                "ess": analysis["ess"],
                "walltime": analysis["walltime_sec"]
            })

    df = pd.DataFrame(results)

    # Example analysis: LLC vs network size by sampler
    summary = df.groupby(["sampler", "target_params"]).agg({
        "llc_mean": ["mean", "std"],
        "ess": "mean",
        "walltime": "mean"
    }).round(3)

    return df, summary

# Usage
df, summary = analyze_sweep_results()
print(summary)
```

## Performance Considerations

### 1. Target Reuse
- Targets are expensive (neural network training)
- Same target config → same target ID → automatic reuse
- Build targets once, sweep samplers many times

### 2. Computational Resources
```bash
# Memory-intensive targets, lightweight sampling
uv run snakemake "runs/targets/*/meta.json" -j 2 --resources mem_mb=16000
uv run snakemake "runs/targets/*/run_*/analysis.json" -j 8 --resources mem_mb=4000

# HPC with different resource requirements per rule
uv run snakemake --profile slurm -j 100 \
  --set-resources build_target:time=120 \
  --set-resources run_sampler:time=360
```

### 3. Storage Management
- Large traces: Use promotion to extract key plots
- Cleanup old experiments: Remove unused target directories
- Archive completed sweeps: Tar up `runs/targets/` subdirectories

```bash
# Promote key results before cleanup
uv run lambda-hat-promote gallery --runs-root runs --samplers hmc,sgld,mclmc \
  --outdir results/sweep_2024_01 --snippet-out results/sweep_2024_01/README.md

# Archive specific targets
tar -czf archive_targets_2024_01.tar.gz runs/targets/tgt_*
```