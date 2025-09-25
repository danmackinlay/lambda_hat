
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

**Development Environment:**
```bash
# Setup (Python 3.11+ required)
uv sync --extra cpu          # For CPU/macOS
uv sync --extra cuda12       # For CUDA 12 (Linux)

# Direct Python execution (always use uv run)
uv run python -m lambda_hat.entrypoints.sample --help
uv run python script.py     # Instead of: python script.py

# Entry points
uv run lambda-hat-build-target -m model=small,base target.seed=42,43  # Build targets
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=hmc           # Direct sampling
uv run lambda-hat-workflow -m model=small,base sampler=hmc,sgld       # N×M workflows

# Testing
uv run pytest tests/                           # All tests
uv run pytest tests/test_mclmc_validation.py  # Single test
```

**Environment Variables:**
- `HYDRA_FULL_ERROR=1` - Full error traces for debugging
- `LAMBDA_HAT_CODE_VERSION=test123` - Version identifier for runs

## Architecture Overview

**Two-Stage LLC Estimation Pipeline:**

**Stage A (Target Building):**
- `lambda_hat/targets.py`: Neural network target creation (Haiku)
- `lambda_hat/data.py`: Synthetic dataset generation
- `lambda_hat/training.py`: ERM optimization to find L₀ reference loss
- Content-addressed artifact storage with deterministic fingerprinting

**Stage B (MCMC Sampling):**
- `lambda_hat/sampling.py`: SGLD, HMC, MCLMC sampling (BlackJAX)
- `lambda_hat/analysis.py`: LLC computation: λ̂ = n × β × (E[Lₙ(w)] - L₀)
- Enhanced visualization with FGE tracking and efficiency analysis

**Configuration:**
- Hydra-based hierarchical configs in `lambda_hat/conf/`
- Nested access: `cfg.data.n_data`, `cfg.model.target_params`
- Dynamic target ID resolution: `${fingerprint:${model},${data},...}`

## Implementation Notes

**No defensive coding or back compat:**
- prefer fail-fast errors.
- we pin package versions rather than introspecting APIs or versions

**JAX Tree**
- Use `jax.tree.map` exclusively (never `jax.tree_map` or `jax.tree_util.tree_map`)
- Set `vmap` axes explicitly: `in_axes=(0, None)` for `(keys, params)`

**MCMC Integration:**
- JAX 64-bit precision for HMC/MCLMC: `jax.config.update("jax_enable_x64", True)`
- Haiku models need RNG parameter: `model.apply(params, None, X)`
- SGLD uses float32, HMC/MCLMC use float64

**Analysis Pipeline:**
- Use `analyze_traces()` with pre-computed Ln values (memory efficient)
- Samplers track Full-Data Gradient Evaluations (FGEs) and precise timing
- Enhanced visualization includes convergence plots, WNV analysis, and ArviZ diagnostics

**Configuration Patterns:**
- Nested access: `cfg.data.n_data` (not `cfg.n_data`)
- Warmup handling: Analysis gracefully handles warmup >= draws
- Output to `HydraConfig.get().run.dir` for proper artifact placement

## Common Issues

**Configuration:**
- Use nested access: `cfg.data.n_data`, not `cfg.n_data`
- Target ID resolution: Check `${fingerprint:...}` resolver in workflow configs
- Sampler configs must exist in `lambda_hat/conf/sample/sampler/`

**BlackJAX API:**
- MCLMC requires RNG keys for init
- version 1.2.5 is pinned; note that the online documentation is for the `main` branch with many breaking changes
- Warmup issues: Adjust if warmup >= total draws
- BlackJAX API: HMC `warmup.run()` takes `num_steps`, MCLMC `init()` takes no `key`

**JAX Integration:**
- Use `jax.tree.map` (never deprecated `jax.tree_map`)
- Haiku requires RNG parameter: `model.apply(params, None, X)`

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
