
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

## Implementation Notes

**No defensive coding or back compat:**
- prefer fail-fast errors.
- we pin package versions rather than introspecting APIs or versions

**JAX Tree API (Required):**
- Use `jax.tree.map` exclusively (never `jax.tree_map` or `jax.tree_util.tree_map`) which are from old jax versions and are unsuppored
- Set `vmap` axes explicitly: `in_axes=(0, None)` for `(keys, params)`

**Haiku Models:**
- Always include RNG: `model.apply(params, rng, X)` (even for deterministic models where `rng=None`)

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
- version 1.2.5 is pinned; note that the online documentation is for the `main` branch with many breaking changes
- Warmup issues: Adjust if warmup >= total draws
- BlackJAX API: HMC `warmup.run()` takes `num_steps`, MCLMC `init()` takes no `key`

