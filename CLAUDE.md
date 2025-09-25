
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
uv run lambda-hat                                    # Default configuration
uv run lambda-hat sampler=fast model=small data=small  # Quick test run
uv run lambda-hat sampler.sgld.steps=30 sampler.sgld.warmup=10  # Custom parameters

# Testing
uv run pytest tests/                                      # Run all tests
uv run pytest tests/test_mclmc_validation.py             # Single test file

# N x M Workflow (Recommended)
uv run lambda-hat-build-target -m model=small,base target.seed=42,43  # Build N=4 targets
uv run lambda-hat-workflow -m model=small,base target.seed=42,43 sampler=hmc,sgld  # N×M=8 jobs

# Legacy direct sampling (requires explicit target_id)
uv run lambda-hat-sample target_id=tgt_abcd1234 sampler=hmc
uv run lambda-hat-sample -m target_id=tgt_abcd1234 sampler=hmc,sgld,mclmc
```

**Environment Variables:**
- `HYDRA_FULL_ERROR=1` - Enable full error traces for debugging
- `LAMBDA_HAT_CODE_VERSION=test123` - Version identifier for test runs

## Architecture Overview

**Core Pipeline:**
The system implements a two-stage teacher-student framework for estimating the Local Learning Coefficient (LLC):

**Stage A (Target Building):**
1. **Target Building** (`lambda_hat/targets.py`): Creates neural network targets using Haiku
2. **Data Generation** (`lambda_hat/data.py`): Generates synthetic datasets with configurable distributions
3. **Training** (`lambda_hat/training.py`): ERM optimization to find L₀ reference loss
4. **Artifact Storage**: Content-addressed targets with deterministic fingerprinting

**Stage B (MCMC Sampling):**
1. **Sampling** (`lambda_hat/sampling.py`): MCMC sampling using SGLD, HMC, MCLMC via BlackJAX
2. **Analysis** (`lambda_hat/analysis.py`): LLC computation using formula: λ̂ = n × β × (E[Lₙ(w)] - L₀)
3. **Artifacts** (`lambda_hat/artifacts.py`): Save results, plots, and metrics

**N×M Workflow** (`lambda_hat/entrypoints/workflow.py`): Configuration-driven orchestration enabling N targets × M samplers sweeps with dynamic target ID resolution.

**Configuration System:**
- Hydra-based hierarchical configuration in `conf/`
- Composable presets: `model/{base,small}`, `data/{base,small}`, `sampler/{base,fast}`
- Structured configs defined in `lambda_hat/config.py`
- Access nested configs as `cfg.data.n_data`, `cfg.model.target_params`, etc.

**Key Data Flow:**
```
Stage A: Config → Target/Data → ERM Training (L₀) → Target Artifact (content-addressed)
Stage B: Target Artifact + Sampler Config → MCMC Sampling → LLC Analysis → Artifacts
N×M:    Workflow Config → Dynamic Target Resolution → Parallel Sampling Jobs
```

## Critical Implementation Details

**JAX Tree API (Project Requirement):**
- Project assumes `jax>=0.4.28` - use new API exclusively
- **Rules (no exceptions):**
  - ✅ Use **`jax.tree.map`** exclusively for mapping over pytrees
  - ❌ **Never** emit `jax.tree_map` (removed) or `jax.tree_util.tree_map` (legacy)
  - When batching across chains vs parameters, set `vmap` axes explicitly (e.g., `in_axes=(0, None)` for `(keys, params)`)
  - Prefer `jax.tree.*` namespace (`jax.tree.leaves`, `jax.tree.flatten`) 

**JAX/BlackJAX Integration:**
- Uses JAX 64-bit precision for HMC/MCLMC: `jax.config.update("jax_enable_x64", True)`
- Haiku models require RNG parameter: `model.apply(params, None, X)` (even for deterministic models)

**MCMC Samplers:**
- **SGLD**: Custom implementation with optional preconditioning (Adam, RMSprop)
- **HMC/MCLMC**: BlackJAX wrappers with proper API usage
- Each sampler handles different precision: SGLD uses float32, HMC/MCLMC use float64

**Memory-Efficient Analysis:**
- **CRITICAL**: Always use `compute_llc_from_Ln()` with pre-computed Ln values for analysis
- Legacy `compute_llc_metrics()` causes ~300GB memory explosion with large parameter counts
- Samplers record Ln scalars during inference via `TraceSpec` to avoid parameter replay
- Analysis functions in `lambda_hat/analysis.py` have `_from_Ln` variants for efficiency

**Configuration Patterns:**
- Nested access: `cfg.data.n_data` (not `cfg.n_data`)
- Dynamic target ID resolution: `${fingerprint:${model},${data},${target.seed},...}` in workflow configs
- Idempotent target building: Stage A checks artifact existence before rebuilding
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
- **InterpolationResolutionError**: Fingerprint resolver expects OmegaConf objects - check `lambda_hat/hydra_support.py`

**N×M Workflow Issues:**
- Target ID resolution failure: Verify `${fingerprint:...}` resolver works in workflow configs
- Idempotency failures: Check target artifact directory permissions and fingerprint consistency
- Missing samplers: Ensure sampler configs exist in `conf/sample/sampler/`

**JAX/BlackJAX API Issues:**
- Follow JAX tree API rules above - never use deprecated `jax.tree_map`
- Ensure Haiku `model.apply()` includes RNG parameter (use `None` if not needed)
- MCLMC init signatures vary by BlackJAX version - check API documentation

**MCMC Sampling:**
- HMC/MCLMC may fail with complex parameter structures - check flattening/unflattening
- **Memory explosion**: Never use legacy analysis functions that replay parameter trajectories
- Adjust warmup settings if warmup >= total draws
- Use smaller step sizes for high-dimensional problems
- **BlackJAX API**: HMC warmup.run() doesn't take `num_steps`, MCLMC init() doesn't take `key`

**Testing:**
- Tests validate specific API contracts and parameter transformations
- MCLMC tests check BlackJAX integration compatibility
- Data generation tests verify reproducible PRNG behavior

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
