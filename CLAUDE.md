
# CLAUDE.md

Critical guidance for working with this LLC estimation codebase.

## Essential Commands

```bash
# Quick test
uv run lambda-hat sampler=fast model=small data=small

# Development
uv run pytest tests/
uv run ruff check --fix
uv run ruff format
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

## API cheat-sheet  (use these, do not “upgrade” them)

* **BlackJAX 1.2.5**

  * **HMC warmup**:

    ```python
    wa = blackjax.window_adaptation(blackjax.hmc, logdensity_fn,
                                    target_acceptance_rate=...,
                                    num_integration_steps=...)
    warm = wa.run(rng_key, position, num_steps=adaptation_steps)
    (final_state, (step_size, inv_mass)), _ = warm
    ```
  * **HMC sampling**:

    ```python
    hmc = blackjax.hmc(logdensity_fn, step_size=..., num_integration_steps=..., inverse_mass_matrix=inv_mass)
    state = hmc.init(position)
    state, info = hmc.step(rng_key, state)
    ```
  * **MCLMC**:

    ```python
    state0 = blackjax.mclmc.init(position, logdensity_fn)  # no rng key
    kernel = blackjax.mclmc(logdensity_fn, L=L, step_size=eps, sqrt_diag_cov=sqrt_cov, integrator=blackjax.mcmc.integrators.isokinetic_mclachlan)
    state, info = kernel.step(rng_key, state)
    # tuner:
    (L, eps, sqrt_cov), _ = blackjax.mclmc_find_L_and_step_size(mclmc_kernel=kernel_factory, state=state0, rng_key=key, num_steps=..., ...)
    ```

* **JAX 0.7.1+**

  * **Trees**: `jax.tree.map`, `jax.tree_util.tree_leaves`, `jax.flatten_util.ravel_pytree`.
  * **Random**: `jax.random.split(key, n)`, `jax.random.choice(key, n, shape=(k,), replace=False)`.
  * **Loops**: `jax.lax.scan(fn, carry, xs)` with `@jax.jit`.

These match how the repo already uses the libraries.
Do **not** switch to main-branch BlackJAX symbols or older `jax.tree_map`/`tree_util.tree_map`.
