# Experiment Matrices

This document explains how to design experiments using the matrix approach: jobs = (problem × sampler × seed).


## Core Concepts

**ProblemVariant**: Defines a model/data configuration (e.g., different sizes)
**SamplerVariant**: Defines a sampler with specific hyperparameters
**Seed**: Random seed for reproducibility

**Family**: Jobs that share (problem, seed) but differ by sampler - used for cross-sampler comparisons.

## YAML Study Files

Define experiments declaratively in a study file:

```yaml
# study.yaml
base:
  preset: quick
  target: mlp
  depth: 3

problems:
  - name: small
    overrides: {target_params: 2000}
  - name: medium
    overrides: {target_params: 5000}
  - name: large
    overrides: {target_params: 10000}

samplers:
  - name: sgld
    overrides: {sgld_precond: none, sgld_step_size: 1e-6}
  - name: sgld
    overrides: {sgld_precond: adam, sgld_beta1: 0.9, sgld_beta2: 0.999, sgld_step_size: 1e-6}
  - name: sghmc
    overrides: {sghmc_temperature: 1.0, sghmc_step_size: 1e-6}
  - name: hmc
    overrides: {hmc_num_integration_steps: 10}
  - name: mclmc
    overrides: {mclmc_integrator: isokinetic_mclachlan}

seeds: [0, 1, 2, 3, 4]
backend: modal
gpu_mode: vectorized
```

Run with:

```bash
uv run llc sweep --study study.yaml
```

This creates 3 problems × 5 samplers × 5 seeds = 75 jobs.

## JSON Grid Sweeps

For quick experiments, use JSON grids directly:

### Sampler Variants Only

```bash
uv run llc sweep --sampler-grid='[
  {"name":"sgld","overrides":{"sgld_precond":"none"}},
  {"name":"sgld","overrides":{"sgld_precond":"adam","sgld_beta2":0.999}},
  {"name":"hmc","overrides":{"hmc_num_integration_steps":20}}
]'
```

### Problem Variants Only

```bash
uv run llc sweep --problem-grid='[
  {"name":"dim_5k","overrides":{"target_params":5000}},
  {"name":"dim_10k","overrides":{"target_params":10000}}
]'
```

### Both Together

```bash
uv run llc sweep \
  --problem-grid='[{"name":"large","overrides":{"target_params":10000}}]' \
  --sampler-grid='[{"name":"sgld","overrides":{"sgld_precond":"adam"}}]'
```

## Common Patterns

### Fix Sampler, Sweep Size

Study how performance scales with model size for one sampler:

```yaml
base:
  preset: quick
problems:
  - name: tiny
    overrides: {target_params: 500}
  - name: small
    overrides: {target_params: 2000}
  - name: medium
    overrides: {target_params: 5000}
  - name: large
    overrides: {target_params: 10000}
samplers:
  - name: sgld
    overrides: {sgld_precond: adam}
seeds: [0, 1, 2]
```

### Fix Size, Sweep Sampler Variants

Compare different sampler configurations on a fixed problem:

```yaml
base:
  preset: quick
  target_params: 5000
problems:
  - name: fixed
    overrides: {}
samplers:
  - name: sgld
    overrides: {sgld_precond: none}
  - name: sgld
    overrides: {sgld_precond: rmsprop, sgld_beta2: 0.99}
  - name: sgld
    overrides: {sgld_precond: adam, sgld_beta1: 0.9, sgld_beta2: 0.999}
  - name: sghmc
    overrides: {sghmc_temperature: 0.5}
  - name: sghmc
    overrides: {sghmc_temperature: 1.0}
  - name: hmc
    overrides: {}
seeds: [0, 1, 2, 3, 4]
```

## Results and Analysis

**CSV Output**: Each job produces one row in `llc_sweep_results.csv` with columns:
- `problem`: Problem variant name
- `sampler`: Sampler name
- `family_id`: Groups jobs with same (problem, seed)
- Performance metrics: `llc_mean`, `ess`, `wnv_time`, etc.

**Plotting**: Use `llc plot-sweep` to visualize results:

```bash
# ESS/sec vs problem size, colored by sampler
uv run llc plot-sweep --size-col target_params --samplers sgld,sghmc,hmc

# Filter to specific problem variants
uv run llc plot-sweep --filters "problem=large"
```

**Family Grouping**: The `family_id` allows cross-sampler comparisons on identical (problem, seed) combinations, enabling fair statistical comparisons.

## Legacy Compatibility

If no study/grid options are provided, sweeps fall back to the legacy `sweep_space()` behavior for backward compatibility. However, the matrix approach is recommended for new experiments.