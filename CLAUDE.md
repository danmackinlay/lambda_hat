
# CLAUDE.md

Critical guidance for working with this LLC estimation codebase.

## Essential Commands

```bash
# Quick test
uv run lambda-hat sampler=fast model=small data=small

# Development
uv run pytest tests/
uv run ruff check lambda_hat/
```

## Critical Implementation Details

**JAX Tree API (BREAKING):**
- ✅ Use `jax.tree_util.tree_map` for BlackJAX compatibility
- ❌ Never use `jax.tree_map` (removed in JAX v0.6.0)

**Haiku Models:**
- Always include RNG: `model.apply(params, None, X)` (even for deterministic models)

**Memory-Efficient Sampling:**
- All samplers use `loss_full_fn` parameter for scalar Ln recording (not parameter trajectories)
- Analysis requires traces with `"Ln"` key - no position-based analysis
- SGLD: float32, HMC/MCLMC: float64

**Configuration:**
- Nested access only: `cfg.data.n_data` (not `cfg.n_data`)

## Common Gotchas

**SGLD batch size > dataset size:**
- Use `replace=True` in `jax.random.choice` when `batch_size > n_data`

**BlackJAX API:**
- MCLMC requires RNG keys for init
- HMC `window_adaptation` may have internal `num_steps` errors
- version 1.2.5 is pinned; note that the online documentation is for the `main` branch with many breaking changes
