# Compatibility

**Pinned versions and compatibility notes for Lambda-Hat.**

---

## Version Pins

Lambda-Hat pins critical dependencies to ensure reproducibility and reduce maintenance burden:

* **JAX** `~=0.7.2` (CPU or CUDA12 extras)
* **BlackJAX** `==1.2.5`
* **FlowJAX** (optional; required for flow-based VI, enable with `--extra flowvi`)

These pins are enforced in `pyproject.toml`. **Do not** bump versions without updating code paths and re-running tests.

---

## BlackJAX 1.2.5 Notes

### API Compatibility

When consulting BlackJAX documentation, ensure you are viewing the [1.2.5 tag](https://github.com/blackjax-devs/blackjax/tree/1.2.5). The online documentation defaults to the `main` branch, which may differ significantly.

**HMC warmup** uses this pattern:

```python
wa = blackjax.window_adaptation(
    blackjax.hmc, logdensity_fn,
    target_acceptance_rate=...,
    num_integration_steps=...
)
warm = wa.run(rng_key, position, num_steps=adaptation_steps)
(final_state, (step_size, inv_mass)), _ = warm
```

**MCLMC init** requires an RNG key:

```python
state0 = blackjax.mclmc.init(position, logdensity_fn, rng_key=key)
kernel = blackjax.mclmc(logdensity_fn, L=L, step_size=eps, sqrt_diag_cov=sqrt_cov, integrator=blackjax.mcmc.integrators.isokinetic_mclachlan)
state, info = kernel.step(rng_key, state)
```

**MCLMC adaptation** uses:

```python
(L, eps, sqrt_cov), _ = blackjax.mclmc_find_L_and_step_size(
    mclmc_kernel=kernel_factory,
    state=state0,
    rng_key=key,
    num_steps=...
)
```

### RNG Key Guidelines

* `.init(...)` **may** need `rng_key` (MCLMC does in 1.2.5)
* `.step(key, state)` **always** needs an RNG key
* When using `jax.vmap` over chains, **vmap the RNG split** too (one key per chain)

---

## JAX 0.7.2 API Usage

Use the modern JAX API:

* **Trees**: `jax.tree.map`, `jax.tree_util.tree_leaves`, `jax.flatten_util.ravel_pytree`
* **Random**: `jax.random.split(key, n)`, `jax.random.choice(key, n, shape=(k,), replace=False)`
* **Loops**: `jax.lax.scan(fn, carry, xs)` with `@jax.jit`

**Do not** use deprecated functions like `jax.tree_map` or patterns from older JAX versions.

---

## Precision Conventions

* **SGLD**: `float32` (for efficiency)
* **HMC/MCLMC**: `float64` (for accuracy)
* **VI**: `float32` (configurable via `sampler.vi.dtype`)

Precision is set per-sampler via the `dtype` config field. Mismatches between target build precision and sampling precision will trigger errors (precision guard).

---

## Enforcement

* CI checks verify that pinned versions in `pyproject.toml` match this documentation
* Stale path checks prevent references to moved/renamed modules
* Run `uv run python docs/_checks.py` to validate manually

See `docs/CONTRIBUTING.md` for the drift prevention checklist.
