
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

**Development Environment:**
```bash
# Setup (Python 3.11+ required)
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra cpu          # For CPU/macOS
uv sync --extra cuda12       # For CUDA 12 (Linux)

# Running experiments (MODERN COMMANDS)
uv run lambda-hat                                         # Default configuration
uv run lambda-hat sampler=fast model=small data=small    # Quick test run
uv run lambda-hat sampler.sgld.steps=30 sampler.sgld.warmup=10  # Custom parameters

# Testing and development
uv run pytest tests/                                      # Run all tests
uv run pytest tests/test_mclmc_validation.py             # Single test file
uv run ruff check lambda_hat/                            # Lint code
uv run ruff format lambda_hat/                           # Format code

# Multi-run experiments (Hydra sweeps)
uv run lambda-hat --multirun model.target_params=1000,5000  # Parameter sweep
uv run lambda-hat --multirun sampler=base,fast              # Configuration sweep
```

**Environment Variables:**
- `HYDRA_FULL_ERROR=1` - Enable full error traces for debugging
- `JAX_PLATFORMS=cpu` - Force CPU backend (useful for debugging)
- `LAMBDA_HAT_CODE_VERSION=test123` - Version identifier for test runs

## Architecture Overview

**Entry Points:**
- **`lambda-hat`** (main CLI): Defined in `lambda_hat/cli_main.py`, calls `lambda_hat/entry.py`
- **`lambda-hat-promote`**: Defined in `lambda_hat/cli_promote.py` for result promotion
- Configuration resolved via Hydra from `conf/config.yaml` and composable presets

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
  - ❌ **Never** emit `jax.tree_map` (removed in JAX v0.6.0)
  - ✅ Use `jax.tree_util.tree_map` for compatibility with current BlackJAX 1.2.5
  - When batching across chains vs parameters, set `vmap` axes explicitly (e.g., `in_axes=(0, None)` for `(keys, params)`)
  - Prefer `jax.tree.*` namespace (`jax.tree.leaves`, `jax.tree.flatten`) where available

**JAX/BlackJAX Integration:**
- Uses JAX 64-bit precision for HMC/MCLMC: `jax.config.update("jax_enable_x64", True)`
- Haiku models require RNG parameter: `model.apply(params, None, X)` (even for deterministic models)

**MCMC Samplers (Memory-Efficient Architecture):**
- **SGLD**: Custom implementation with optional preconditioning (Adam, RMSprop)
- **HMC/MCLMC**: BlackJAX wrappers with proper API usage
- Each sampler handles different precision: SGLD uses float32, HMC/MCLMC use float64
- **Critical**: All samplers use efficient scalar Ln recording to avoid 300GB memory allocations
- Use `loss_full_fn` parameter to record scalar loss values during sampling, not full parameter trajectories

**Configuration Patterns:**
- Nested access: `cfg.data.n_data` (not `cfg.n_data`)
- Warmup validation: Analysis handles warmup >= draws gracefully with warnings
- Parameter flattening/unflattening for structured Haiku parameters

**Memory-Efficient Sampling Architecture:**
- `simple_inference_loop` only records positions when `theta_every > 0` (default: 0 = no positions)
- All samplers record scalar `Ln` values efficiently via `aux_fn` parameter
- Analysis functions require traces with `"Ln"` key - position-based analysis removed to prevent OOM
- Key functions: `make_loss_full()` creates scalar recording functions for `loss_full_fn` parameter

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
- Follow JAX tree API rules above - use `jax.tree.map` for new code, `jax.tree_util.tree_map` for BlackJAX compatibility
- Ensure Haiku `model.apply()` includes RNG parameter (use `None` if not needed)
- **MCLMC**: BlackJAX 1.2.5 requires RNG keys for init, always provide identity inverse mass matrix
- **HMC**: May encounter `num_steps` parameter error in `window_adaptation` - this is a BlackJAX internal issue

**Memory Issues:**
- If encountering OOM during analysis, ensure samplers use `loss_full_fn` parameter for scalar Ln recording
- Never store full parameter trajectories for large models - use `theta_every=0` (default)
- Analysis functions now require `"Ln"` key in traces - position-based analysis removed

**MCMC Sampling:**
- **SGLD**: Working perfectly with memory-efficient architecture
- **MCLMC**: Working perfectly, requires proper parameter flattening/unflattening
- **HMC**: Has BlackJAX internal issues but architecture is sound
- Adjust warmup settings if warmup >= total draws

**Testing:**
- Tests validate specific API contracts and parameter transformations
- MCLMC tests check BlackJAX integration compatibility
- Data generation tests verify reproducible PRNG behavior
