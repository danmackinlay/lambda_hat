
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

# lint before commits
uv run ruff format
uv run ruff check --fix
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

**JAX**
- Use `jax.tree.map` (never deprecated `jax.tree_map`); eventually migrate from compat layer `jax.tree_util.tree_map`
- Haiku requires RNG parameter: `model.apply(params, None, X)`
- Set `vmap` axes explicitly: `in_axes=(0, None)` for `(keys, params)`

**MCMC:**
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

**BlackJAX API (Version 1.2.5):**
- **HMC warmup:**
  ```python
  wa = blackjax.window_adaptation(blackjax.hmc, logdensity_fn,
                                  target_acceptance_rate=...,
                                  num_integration_steps=...)
  warm = wa.run(rng_key, position, num_steps=adaptation_steps)
  (final_state, (step_size, inv_mass)), _ = warm
  ```
- **HMC sampling:**
  ```python
  hmc = blackjax.hmc(logdensity_fn, step_size=..., num_integration_steps=...,
                     inverse_mass_matrix=inv_mass)
  state = hmc.init(position)
  state, info = hmc.step(rng_key, state)
  ```
- **MCLMC:**
  ```python
  state0 = blackjax.mclmc.init(position, logdensity_fn)  # no rng key
  kernel = blackjax.mclmc(logdensity_fn, L=L, step_size=eps, sqrt_diag_cov=sqrt_cov)
  state, info = kernel.step(rng_key, state)
  # tuner:
  (L, eps, sqrt_cov), _ = blackjax.mclmc_find_L_and_step_size(
      mclmc_kernel=kernel_factory, state=state0, rng_key=key, num_steps=...)
  ```

**JAX API (Version 0.7.1+):**
- **Trees:** Use `jax.tree.map`, never deprecated `jax.tree_map`
- **Random:** `jax.random.split(key, n)`, `jax.random.choice(key, n, shape=(k,))`
- **Utils:** `jax.flatten_util.ravel_pytree`, `jax.lax.scan` with `@jax.jit`
