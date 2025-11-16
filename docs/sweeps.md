# Parameter Sweeps

This document explains how to design and execute parameter sweeps with Parsl. Lambda-Hat creates deterministic N × M experiment matrices defined in `config/experiments.yaml`.

## Core Concept: N × M Sweeps

Lambda-Hat automatically generates a **Cartesian product** of targets and samplers from your experiment configuration:

- **N targets** (different model/data/seed combinations)
- **M samplers** (different MCMC/VI algorithms and hyperparameters)
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
  - { name: vi, overrides: { M: 8, r: 2 } }
```

**Result**: 3 targets × 4 samplers = 12 sampling jobs + 3 target-building jobs = **15 total jobs**.

### Execute the Sweep

```bash
# Local execution (testing)
uv run lambda-hat workflow llc --local

# SLURM cluster (production)
uv run lambda-hat workflow llc --parsl-card config/parsl/slurm/gpu-a100.yaml

# With promotion (galleries)
uv run lambda-hat workflow llc --local --promote
```

Parsl automatically:
- Builds all N targets in parallel
- Runs all M samplers per target (waiting for target to complete)
- Aggregates results into `results/llc_runs.parquet`

---

## Designing Sweeps

### 1. Model Architecture Sweeps

Compare different network architectures:

```yaml
targets:
  # Using different presets
  - { model: small, data: base, seed: 42 }
  - { model: base,  data: base, seed: 42 }
  - { model: wide,  data: base, seed: 42 }

  # Using overrides for one-off variations
  - { model: base, data: base, seed: 42,
      overrides: { model: { depth: 8, activation: gelu } } }
```

**Use case**: Understand how architecture affects LLC estimates.

### 2. Data Distribution Sweeps

Test robustness across different data distributions:

```yaml
targets:
  - { model: base, data: gauss_iso, seed: 42 }
  - { model: base, data: gauss_aniso, seed: 42 }
  - { model: base, data: mixture, seed: 42 }
  - { model: base, data: heavy_tail, seed: 42,
      overrides: { data: { noise_model: student_t } } }
```

**Use case**: Validate sampler performance on diverse data.

### 3. Sampler Hyperparameter Sweeps

Compare different sampler configurations:

```yaml
samplers:
  # HMC with different integration steps
  - { name: hmc, overrides: { num_integration_steps: 5 } }
  - { name: hmc, overrides: { num_integration_steps: 10 } }
  - { name: hmc, overrides: { num_integration_steps: 20 } }

  # SGLD with different step sizes
  - { name: sgld, overrides: { step_size: 1e-5 } }
  - { name: sgld, overrides: { step_size: 1e-6 } }
  - { name: sgld, overrides: { step_size: 1e-7 } }

  # VI with different ranks
  - { name: vi, overrides: { M: 4, r: 1 } }
  - { name: vi, overrides: { M: 8, r: 2 } }
  - { name: vi, overrides: { M: 16, r: 4 } }
```

**Use case**: Hyperparameter tuning and sensitivity analysis.

### 4. Seed Sweeps (Replications)

Multiple random seeds for statistical significance:

```yaml
targets:
  - { model: base, data: base, seed: 42 }
  - { model: base, data: base, seed: 43 }
  - { model: base, data: base, seed: 44 }
  - { model: base, data: base, seed: 45 }
  - { model: base, data: base, seed: 46 }
```

**Use case**: Error bars and statistical testing.

**Note**: Each target with a different seed creates a *different neural network* with independent weights, providing true replication.

### 5. Combined Sweeps

Complex factorial designs:

```yaml
targets:
  # 2 models × 2 datasets × 3 seeds = 12 targets
  - { model: small, data: small, seed: 42 }
  - { model: small, data: small, seed: 43 }
  - { model: small, data: small, seed: 44 }
  - { model: small, data: large, seed: 42 }
  - { model: small, data: large, seed: 43 }
  - { model: small, data: large, seed: 44 }
  - { model: base, data: small, seed: 42 }
  - { model: base, data: small, seed: 43 }
  - { model: base, data: small, seed: 44 }
  - { model: base, data: large, seed: 42 }
  - { model: base, data: large, seed: 43 }
  - { model: base, data: large, seed: 44 }

samplers:
  # 3 samplers
  - { name: hmc }
  - { name: sgld }
  - { name: vi }
```

**Result**: 12 targets × 3 samplers = **36 sampling jobs** + 12 target builds = 48 total jobs.

---

## Analyzing Sweep Results

All results are aggregated into a single parquet file:

```python
import pandas as pd

# Load results
df = pd.read_parquet('results/llc_runs.parquet')

# Group by sampler
print(df.groupby('sampler')[['llc_mean', 'llc_std', 'ess_bulk']].mean())

# Filter by target configuration
hmc_runs = df[df['sampler'] == 'hmc']

# Plot LLC estimates
import matplotlib.pyplot as plt
df.boxplot(column='llc_mean', by='sampler')
plt.show()
```

Each row contains:
- `target_id`: Target identifier
- `sampler`: Sampler name
- `run_id`: Run identifier
- `llc_mean`, `llc_std`: LLC estimates
- `ess_bulk`, `ess_tail`, `r_hat`: MCMC diagnostics
- `walltime_sec`: Execution time
- Configuration metadata from `config_yaml`

---

## Best Practices

### Start Small

Begin with a minimal sweep to test the pipeline:

```yaml
targets:
  - { model: small, data: small, seed: 42 }

samplers:
  - { name: vi, overrides: { steps: 100 } }  # Fast VI for testing
```

Run locally: `uv run lambda-hat workflow llc --local`

Once validated, scale up to full sweep on cluster.

### Incremental Scaling

**Phase 1**: Test with 1 target × 1 sampler locally
**Phase 2**: Scale to 2-3 targets × 2-3 samplers locally
**Phase 3**: Move to cluster with 10+ targets × 4+ samplers
**Phase 4**: Full production sweep (50+ targets × 5+ samplers)

### Resource Planning

Estimate total compute time:

```
Total time ≈ (N_targets × target_build_time) + (N_targets × M_samplers × sample_time)
```

**Example**:
- 20 targets, each takes 30 min to build
- 5 samplers per target, each takes 2 hours
- With perfect parallelism: `max(20 × 30min, 20 × 5 × 2hr) = max(10hr, 200hr) = 200hr`
- With 50 parallel SLURM jobs: `200hr / 50 = 4 hours` wall clock

Adjust `max_blocks` in `parsl_config_slurm.py` based on your cluster quota.

### Naming Conventions

Use descriptive preset names for clarity:

```bash
# Good preset names
lambda_hat/conf/model/cnn_shallow.yaml
lambda_hat/conf/model/mlp_deep_relu.yaml
lambda_hat/conf/data/mnist_clean.yaml
lambda_hat/conf/data/cifar10_noisy.yaml

# Less clear
lambda_hat/conf/model/exp1.yaml
lambda_hat/conf/data/test2.yaml
```

This makes `config/experiments.yaml` self-documenting.

---

## Common Sweep Patterns

### Baseline Comparison

Compare new sampler against established baselines:

```yaml
samplers:
  - { name: hmc }        # Gold standard
  - { name: mclmc }      # Fast alternative
  - { name: sgld }       # Stochastic
  - { name: vi }         # Variational (new!)
```

### Ablation Study

Test individual components:

```yaml
targets:
  - { model: base, data: base, seed: 42,
      overrides: { model: { bias: true } } }
  - { model: base, data: base, seed: 42,
      overrides: { model: { bias: false } } }

  - { model: base, data: base, seed: 42,
      overrides: { model: { activation: relu } } }
  - { model: base, data: base, seed: 42,
      overrides: { model: { activation: tanh } } }
```

### Scaling Study

Understand performance at different scales:

```yaml
targets:
  - { model: tiny,  data: tiny,  seed: 42 }  # d=10, n=50
  - { model: small, data: small, seed: 42 }  # d=50, n=100
  - { model: base,  data: base,  seed: 42 }  # d=500, n=1000
  - { model: large, data: large, seed: 42 }  # d=5000, n=10000
```

---

## Troubleshooting Sweeps

**Problem**: Sweep is too large (hundreds of jobs)

**Solution**:
1. Reduce scope by commenting out some targets/samplers
2. Use `max_blocks` to limit parallelism: edit `parsl_config_slurm.py`
3. Run in batches: split `config/experiments.yaml` into multiple files

**Problem**: Some jobs fail while others succeed

**Solution**:
1. Check logs: `logs/run_sampler/*.err`
2. Identify failed configs from missing `analysis.json`
3. Create new `config/retry.yaml` with only failed configs
4. Rerun: `uv run lambda-hat workflow llc --config config/retry.yaml --local`

**Problem**: Results are missing metrics

**Solution**:
- Parsl creates `results/llc_runs.parquet` with all successful runs
- Failed runs are automatically excluded
- Check which runs succeeded: `ls runs/targets/*/run_*/analysis.json`

---

## See Also

- [Configuration Guide](./configuration.md) - Detailed config syntax
- [Parallel Execution](./parallelism.md) - Execution strategies
- [Output Management](./output_management.md) - Understanding artifacts
