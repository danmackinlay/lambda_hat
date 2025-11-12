Minimal rules for this repo. Deviations break CI.

## Versions (hard pins)
- **Python ≥ 3.11**
- **JAX ≥ 0.7.1** (use `jax.tree.map`, `jax.lax.scan`, etc.)
- **BlackJAX == 1.2.5** (don’t import newer APIs)
- Use **uv** for everything; use **Snakemake** for workflows.

## Install

```bash
uv sync --extra cpu        # CPU/macOS
uv sync --extra cuda12     # CUDA 12 (Linux)
```

## Run (always via uv)

```bash
# Console scripts from pyproject.toml
uv run lambda-hat-build-target --config-yaml config.yaml --target-id tgt_abc123
uv run lambda-hat-sample       --config-yaml config.yaml --target-id tgt_abc123
uv run lambda-hat-promote      --help
```

## Workflow (Snakemake is canonical)

**Two-stage pipeline:** Stage A builds targets (neural networks + datasets), Stage B runs samplers (MCMC or variational). Snakemake orchestrates N targets × M samplers in parallel.

```bash
uv run snakemake -n                 # dry run
uv run snakemake -j 4               # local
uv run snakemake --profile slurm -j 100   # cluster
```

## Testing & Lint

```bash
uv run pytest -q
uv run ruff format
uv run ruff check --fix
```

## Precision & APIs

* SGLD: `float32`; HMC/MCLMC: `float64`; VI: `float32` (configurable).
* Precision is per-sampler via `sampler.<name>.dtype` config field.
* Haiku call shape: `model.apply(params, None, X)`.
* Use `jax.tree.map` (not deprecated `tree_map`), `jax.flatten_util.ravel_pytree`, `jax.random.split`.

## Config

* Edit `config/experiments.yaml`; nested keys only (`cfg.data.n_data`, not `cfg.n_data`).
* Artifacts live under `runs/targets/tgt_<hash>/`.

## Don't

* Don't run entry points directly; always use `uv run` and prefer Snakemake for workflows.
* Don't bump BlackJAX/JAX without updating code paths.
* Don't add back-compat shims; fail fast.
* Don't use legacy Hydra CLI (removed); Snakemake + OmegaConf is the only supported workflow.

## Samplers

**Implemented**: HMC, MCLMC, SGLD (with Adam/RMSProp preconditioning)

**Planned**: Variational Inference (VI) - infrastructure complete, algorithm implementation pending


## API cheat-sheet  (use these, do not "upgrade" them)

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
