# Optuna Hyperparameter Optimization Workflow

**Bayesian optimization for SGLD, VI, and MCLMC hyperparameters using Parsl + Optuna**

## Overview

The Optuna workflow automatically tunes hyperparameters for approximate LLC estimation methods (SGLD, VI, MCLMC) by optimizing against high-quality HMC references. It uses:

- **Optuna's ask-and-tell API** for Bayesian optimization (TPE sampler)
- **Parsl** for parallel trial execution across clusters
- **Huber/absolute error objective**: minimize `|LLC_method - LLC_HMC|`
- **Content-addressed artifacts**: idempotent runs with automatic deduplication

**Key benefits**:
- **Data-driven tuning**: Find optimal hyperparameters empirically, not manually
- **Multi-problem optimization**: Test robustness across different architectures/datasets
- **Fair comparison**: Fixed time budget per trial (100 minutes default)
- **Scalable**: Runs locally or on SLURM/PBS clusters with the same command

---

## Quick Start

### 1. Create an Optuna config

`config/my_optuna_run.yaml`:
```yaml
problems:
  - model: small
    data: base
    teacher: _null
    seed: 42

methods:
  - vi
  - sgld

optuna:
  max_trials: 100          # Trials per (problem, method)
  batch_size: 16           # Concurrent trials
  hmc_budget_sec: 7200     # HMC reference: 2 hours
  method_budget_sec: 600   # Method trial: 10 minutes
```

**Note on budgets**: Default HMC budget (10 hours) is generous for production-quality references. For local testing or quick iteration, use shorter budgets via CLI:
```bash
uv run parsl-optuna --config config/my_optuna_run.yaml --local \
    --hmc-budget 1800 --method-budget 300  # 30min HMC, 5min trials
```

### 2. Run locally (testing)

```bash
uv run parsl-optuna --config config/my_optuna_run.yaml --local
```

### 3. Run on SLURM cluster

```bash
uv run parsl-optuna --config config/my_optuna_run.yaml \
    --parsl-config parsl_config_slurm.py
```

### 4. Analyze results

Results are written to `results/optuna_trials.parquet`:

```python
import pandas as pd

df = pd.read_parquet("results/optuna_trials.parquet")

# Best trials per method
for method in df["method"].unique():
    best = df[df["method"] == method].nsmallest(1, "objective")
    print(f"{method}: {best[['hyperparams', 'objective', 'error']].to_dict('records')}")

# Convergence plots
import matplotlib.pyplot as plt
for method in df["method"].unique():
    subset = df[df["method"] == method].sort_values("trial_id")
    plt.plot(subset["objective"].cummin(), label=method)
plt.legend()
plt.xlabel("Trial")
plt.ylabel("Best Objective (cumulative min)")
plt.show()
```

---

## Workflow Stages

### Stage 1: HMC References

For each problem in `config.problems`:

1. Compute content-addressed problem ID (`p_<hash>`)
2. Check if `artifacts/problems/<pid>/ref.json` exists
3. If not, submit HMC job with generous settings:
   - 4 chains × 10,000 samples
   - High acceptance rate (0.8)
   - Target budget: 10 hours (configurable via `--hmc-budget`)
4. Extract reference LLC estimate (`llc_ref`) and SE

**All HMC references run in parallel** (limited by Parsl executor capacity).

### Stage 2: Bayesian Optimization

For each `(problem, method)` pair:

1. Create Optuna study with TPE sampler
2. **Ask-and-tell loop**:
   - **Ask**: Optuna proposes hyperparameters via Bayesian optimization
   - **Submit**: Create trial manifest, launch Parsl app with time budget
   - **Tell**: When trial completes, compute `|LLC_hat - LLC_ref|` and report to Optuna
   - **Refill**: Keep `batch_size` trials in flight at all times
3. Run until `max_trials` completed or convergence
4. Save best hyperparameters and study state

**Trials within a (problem, method) run concurrently** up to `batch_size`. Multiple (problem, method) pairs run in sequence (parallelization across problems requires manual orchestration).

### Stage 3: Aggregation

All trial metrics are collected into a single DataFrame and written to `results/optuna_trials.parquet`. Each row contains:

- `pid`: Problem ID
- `method`: Method name
- `trial_id`: Trial ID (content-addressed)
- `hyperparams`: Dict of hyperparameters
- `llc_hat`: Estimated LLC
- `llc_ref`: Reference LLC (from HMC)
- `objective`: `|llc_hat - llc_ref|`
- `error`: Same as objective (for convenience)
- `runtime_sec`: Wall-time for trial
- `diagnostics`: ESS, R-hat, work metrics, etc.

---

## Configuration Reference

### Problem Specification

Each problem is a dict with keys matching `lambda_hat/conf/`:

```yaml
problems:
  - model: small          # → lambda_hat/conf/model/small.yaml
    data: base            # → lambda_hat/conf/data/base.yaml
    teacher: _null        # → lambda_hat/conf/teacher/_null.yaml
    seed: 42              # Training seed (affects target network weights)
    overrides:            # Optional config overrides
      model:
        depth: 3
        target_params: 5000
```

**Reuses existing configs** from `lambda_hat/conf/` - no duplication.

### Methods

List of method names to optimize:

```yaml
methods:
  - sgld
  - vi
  - mclmc
```

**Hyperparameter search spaces** (defined in `lambda_hat/entrypoints/parsl_optuna.py::suggest_method_params`):

- **SGLD**: `eta0` (log-uniform), `gamma`, `batch`, `precond_type`, `steps`
- **VI (MFA)**: `lr` (log-uniform), `M`, `r`, `whitening_mode`, `steps`, `batch_size`
- **MCLMC**: `step_size` (log-uniform), `target_accept`, `L`, `steps`

To customize search spaces, edit `suggest_method_params()` in `lambda_hat/entrypoints/parsl_optuna.py`.

### Optuna Settings

```yaml
optuna:
  max_trials: 200            # Stop after N trials per (problem, method)
  batch_size: 32             # Concurrent trials (limited by cluster capacity)
  hmc_budget_sec: 36000      # HMC reference: 10 hours (default)
  method_budget_sec: 6000    # Method trial: 100 minutes (default)
```

**CLI overrides** available via `--max-trials`, `--batch-size`, `--hmc-budget`, `--method-budget`.

---

## Directory Structure

```
artifacts/
  problems/
    p_<hash>/
      ref.json                     # HMC reference: {llc_ref, se_ref, diagnostics}
  runs/
    p_<hash>/
      <method>/
        r_<trial_hash>/
          manifest.json              # Trial spec (hyperparams, seed, budget)
          metrics.json               # Trial results (llc_hat, objective, ...)

results/
  optuna_trials.parquet              # Aggregated results (all trials)
  studies/
    optuna_llc/
      <pid>:<method>.pkl             # Pickled Optuna study (for resume)
```

**Content-addressed IDs**:
- Problem ID: `p_<sha256(problem_spec)[:12]>`
- Trial ID: `r_<sha256(manifest)[:12]>`

This ensures **idempotent runs**: re-running with the same config reuses existing artifacts.

---

## Customization

### Custom Objective Function

To use Huber loss instead of absolute error, edit `lambda_hat/entrypoints/parsl_optuna.py`:

```python
# In objective_from_metrics():
return huber_loss(diff, huber_delta=0.1)  # Set delta to ~2× HMC SE
```

### Custom Search Spaces

Edit `suggest_method_params()` to add/remove hyperparameters or change ranges:

```python
def suggest_method_params(trial, method_name):
    if method_name == "vi":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),  # Narrowed range
            "M": trial.suggest_categorical("M", [8, 16, 32]),       # More components
            "entropy_bonus": trial.suggest_float("entropy_bonus", 0.0, 0.5),  # New param
            # ...
        }
```

Update `_build_sampler_cfg()` in `lambda_hat/runners/run_method.py` to pass new params to the sampler.

### Multi-Objective Optimization

To optimize both LLC error **and** runtime:

1. Change Optuna study to multi-objective:
   ```python
   study = optuna.create_study(directions=["minimize", "minimize"])  # Error, Runtime
   study.tell(t, [obj, runtime_sec])
   ```

2. Optuna will return a Pareto front of non-dominated trials.

---

## Comparison with Standard Workflow

| Feature | `parsl_llc.py` | `parsl_optuna.py` |
|---------|----------------|-------------------|
| **Purpose** | Run pre-defined experiments | Optimize hyperparameters |
| **Config** | `config/experiments.yaml` | `config/optuna_demo.yaml` |
| **Samplers** | Fixed hyperparams | Bayesian search over hyperparams |
| **Objective** | None (just run) | Minimize `\|LLC - LLC_ref\|` |
| **HMC Reference** | Optional (user-defined) | Always computed (Stage 1) |
| **Output** | `results/llc_runs.parquet` | `results/optuna_trials.parquet` |
| **Use Case** | Production runs, sweeps | Hyperparameter tuning, method development |

**Recommended workflow**:
1. Use `parsl_optuna.py` to find optimal hyperparameters for your problem class
2. Use `parsl_llc.py` with tuned hyperparameters for production experiments

---

## Testing

The Optuna workflow includes comprehensive integration tests to verify end-to-end functionality.

### Running Tests

```bash
# Run all Optuna integration tests
uv run pytest tests/test_optuna_workflow.py -v

# Run specific test
uv run pytest tests/test_optuna_workflow.py::test_optuna_workflow_integration -v

# Run with output (see print statements)
uv run pytest tests/test_optuna_workflow.py -v -s
```

### Test Configuration

Tests use `tests/test_optuna_config.yaml` with minimal settings for fast execution:
- 1 small problem (model: small, data: small)
- 1 method (VI only)
- 3 trials maximum
- 3-minute HMC budget
- 1-minute trial budget

**Expected test duration**: ~5-7 minutes total

### What Tests Verify

1. **HMC reference computation**: Validates reference LLC is computed and cached correctly
2. **Trial execution**: Verifies trials run and produce valid results
3. **Results structure**: Checks parquet file has correct columns and data
4. **Artifacts**: Validates manifest.json, metrics.json, study pickle exist
5. **Caching**: Ensures HMC references are reused across runs

**Note**: Tests cleanup `artifacts/`, `results/`, and other temporary directories automatically.

---

## Troubleshooting

### "HMC reference failed"

- **Cause**: Target too large, budget too small, or numerical issues
- **Fix**: Increase `--hmc-budget` or simplify problem (smaller model/dataset)
- **Debug**: Check `artifacts/problems/<pid>/ref.json` for diagnostics

### "All trials have infinite objective"

- **Cause**: Method failing to run (config error, OOM, etc.)
- **Fix**: Check `artifacts/runs/<pid>/<method>/<trial_id>/metrics.json` for error messages
- **Debug**: Run a single trial manually:
  ```bash
  uv run python -c "
  from lambda_hat.runners.run_method import run_method_trial
  run_method_trial(
      problem_spec={'model': 'small', 'data': 'base', 'teacher': '_null', 'seed': 42},
      method_cfg={'name': 'vi', 'lr': 0.001, 'M': 8, 'r': 2},
      ref_llc=5.0,
      out_metrics_json='test_trial.json',
      budget_sec=600
  )
  "
  ```

### "Parsl workers idle, no trials running"

- **Cause**: Batch size exceeds executor capacity, or workers not starting
- **Fix**: Reduce `--batch-size` or check Parsl executor config
- **Debug**: Check `runinfo/` directory for Parsl logs

### "Resume from crashed run"

Optuna studies are periodically saved to `results/studies/optuna_llc/<pid>:<method>.pkl`. To resume:

1. Re-run the same command - completed trials will be skipped (content-addressed)
2. Optuna will load the study pickle and continue from the last checkpoint

---

## Performance Tips

### Efficient HMC References

- **Cache**: HMC references are cached at `artifacts/problems/<pid>/ref.json` - reuse across runs
- **Parallel**: All problems' HMC references run in parallel (limited by cluster capacity)
- **Budget**: Balance quality vs speed - 1-2 hours often sufficient for small problems

### Trial Throughput

- **Batch size**: Set to match cluster node count for full utilization
- **Method budget**: 100 minutes (default) balances quality vs iteration speed
- **Worker init**: Ensure `parsl_config_*.py` has fast environment setup (cached conda env, etc.)

### Search Efficiency

- **Start small**: Use `max_trials=20-50` for initial exploration, then increase
- **Warm start**: If you have good guesses, add them as initial trials via Optuna's `enqueue_trial()`
- **Pruning**: Optuna supports early stopping via MedianPruner - requires method to report intermediate values

---

## See Also

- **Design**: `plans/optuna.md` - Full architectural design document
- **Standard workflow**: `docs/vi_mfa.md` - MFA VI configuration and usage
- **Parsl config**: `parsl_config_slurm.py`, `parsl_config_local.py` - Executor configs
- **Optuna docs**: https://optuna.readthedocs.io - Advanced features (pruning, multi-objective, etc.)
