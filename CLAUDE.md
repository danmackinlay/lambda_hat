
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

**Development Environment:**
```bash
# Setup (Python 3.11+ required)
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra cpu          # For CPU/macOS
uv sync --extra cuda12       # For CUDA 12 (Linux)

# Running experiments
uv run python train.py                                    # Default configuration
uv run python train.py sampler=fast model=small data=small  # Quick test run
uv run python train.py sampler.sgld.steps=30 sampler.sgld.warmup=10  # Custom parameters

# Testing
uv run pytest tests/                                      # Run all tests
uv run pytest tests/test_mclmc_validation.py             # Single test file

# Multi-run experiments (Hydra sweeps)
uv run python train.py -m model.target_params=1000,5000  # Parameter sweep
uv run python train.py -m sampler=base,fast              # Configuration sweep
```

**Environment Variables:**
- `HYDRA_FULL_ERROR=1` - Enable full error traces for debugging
- `LAMBDA_HAT_CODE_VERSION=test123` - Version identifier for test runs

## Architecture Overview

**Core Pipeline:**
The system implements a teacher-student framework for estimating the Local Learning Coefficient (LLC) using multiple MCMC samplers:

1. **Target Building** (`lambda_hat/targets.py`): Creates neural network targets using Haiku
2. **Data Generation** (`lambda_hat/data.py`): Generates synthetic datasets with configurable distributions
3. **Training** (`lambda_hat/training.py`): ERM optimization to find L₀ reference loss
4. **Sampling** (`lambda_hat/sampling.py`): MCMC sampling using SGLD, HMC, MCLMC via BlackJAX
5. **Analysis** (`lambda_hat/analysis.py`): LLC computation using formula: λ̂ = n × β × (E[Lₙ(w)] - L₀)
6. **Artifacts** (`lambda_hat/artifacts.py`): Save results, plots, and metrics

**Configuration System:**
- Hydra-based hierarchical configuration in `conf/`
- Composable presets: `model/{base,small}`, `data/{base,small}`, `sampler/{base,fast}`
- Structured configs defined in `lambda_hat/config.py`
- Access nested configs as `cfg.data.n_data`, `cfg.model.target_params`, etc.

**Key Data Flow:**
```
Config → Target/Data → ERM Training (L₀) → MCMC Sampling → LLC Analysis → Artifacts
```

## Critical Implementation Details

**JAX Tree API (Project Requirement):**
- Project assumes `jax>=0.4.28` - use new API exclusively
- **Rules (no exceptions):**
  - ✅ Use **`jax.tree.map`** exclusively for mapping over pytrees
  - ❌ **Never** emit `jax.tree_map` (removed) or `jax.tree_util.tree_map` (legacy)
  - When batching across chains vs parameters, set `vmap` axes explicitly (e.g., `in_axes=(0, None)` for `(keys, params)`)
  - Prefer `jax.tree.*` namespace (`jax.tree.leaves`, `jax.tree.flatten`) where available

**JAX/BlackJAX Integration:**
- Uses JAX 64-bit precision for HMC/MCLMC: `jax.config.update("jax_enable_x64", True)`
- Haiku models require RNG parameter: `model.apply(params, None, X)` (even for deterministic models)

**MCMC Samplers:**
- **SGLD**: Custom implementation with optional preconditioning (Adam, RMSprop)
- **HMC/MCLMC**: BlackJAX wrappers with proper API usage
- Each sampler handles different precision: SGLD uses float32, HMC/MCLMC use float64

**Configuration Patterns:**
- Nested access: `cfg.data.n_data` (not `cfg.n_data`)
- Warmup validation: Analysis handles warmup >= draws gracefully with warnings
- Parameter flattening/unflattening for structured Haiku parameters

**Output Management:**
- Single runs save to `outputs/YYYY-MM-DD/HH-MM-SS/`
- Multi-runs save to `multirun/YYYY-MM-DD/HH-MM-SS/`
- Artifacts include: config.yaml, target_info.json, run_info.json, metrics CSV, plots (if enabled), summary.txt
- Uses `HydraConfig.get().run.dir` to ensure artifacts go to proper Hydra output directory (not repo root)

## Troubleshooting Common Issues

**Configuration Errors:**
- Always use nested access patterns: `cfg.data.n_data`, not `cfg.n_data`
- Check Hydra warnings about ConfigStore registrations

**JAX/BlackJAX API Issues:**
- Follow JAX tree API rules above - never use deprecated `jax.tree_map`
- Ensure Haiku `model.apply()` includes RNG parameter (use `None` if not needed)
- MCLMC init signatures vary by BlackJAX version - check API documentation

**MCMC Sampling:**
- HMC/MCLMC may fail with complex parameter structures - check flattening/unflattening
- Adjust warmup settings if warmup >= total draws
- Use smaller step sizes for high-dimensional problems

**Testing:**
- Tests validate specific API contracts and parameter transformations
- MCLMC tests check BlackJAX integration compatibility
- Data generation tests verify reproducible PRNG behavior
