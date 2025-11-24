Minimal rules for this repo. Deviations break CI.

## Versions
- **Python ≥ 3.11**
- **JAX ≥ 0.7.2** (use `jax.tree.map`, `jax.lax.scan`, etc.)
- **BlackJAX == 1.2.5** (don't import newer APIs)
- Use **uv** for dependencies and execution; use **Parsl** for workflows.

## Priorities

There are no downstream users. There are no other developers.
The highest good is to reduce maintenance burden for me.
You can break things to achieve that good.
No back compat shims. Don't deprecate when you can destroy.

## Install

```bash
uv sync --extra cpu        # CPU/macOS
uv sync --extra cuda12     # CUDA 12 (Linux)
uv sync --extra flowvi     # Add FlowJAX for flow-based VI (optional)
```

## Run (unified CLI)

```bash
# All commands via unified 'lambda-hat' CLI
uv run lambda-hat build --config-yaml config/experiments.yaml --target-id tgt_abc123
uv run lambda-hat sample --config-yaml config/experiments.yaml --target-id tgt_abc123
uv run lambda-hat promote gallery --runs-root artifacts/experiments/dev/runs --samplers sgld,hmc
uv run lambda-hat artifacts gc      # garbage collect old artifacts
uv run lambda-hat artifacts ls      # list experiments and runs
uv run lambda-hat --help            # see all commands
```

## Workflow

**Three-stage pipeline:** Build targets (neural networks + datasets), run samplers (MCMC or variational), optionally promote results (gallery + aggregation). Parsl orchestrates N targets × M samplers in parallel using Python-native `@python_app` functions with content-addressed IDs for reproducibility.

```bash
# Local execution (testing)
uv run lambda-hat workflow llc --backend local

# SLURM cluster execution
uv run lambda-hat workflow llc --parsl-card config/parsl/slurm/gpu-a100.yaml

# With promotion (opt-in via --promote flag)
uv run lambda-hat workflow llc --backend local  --promote

# Optuna hyperparameter optimization
uv run lambda-hat workflow optuna --config config/optuna_demo.yaml --backend local
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
* Equinox call shape: `model(X)` (model IS the params).
* Use `jax.tree.map` (not deprecated `tree_map`), `jax.flatten_util.ravel_pytree`, `jax.random.split`.

## Config

* Edit `config/experiments.yaml` to define target/sampler configurations.
* Use `overrides` dict for custom values (e.g., `overrides: { training: { steps: 10000 } }`).
* See `docs/experiments.md` for experiments guide and `docs/config.md` for complete schema.

## Artifacts

* Default artifact home: `artifacts/` (configurable via `LAMBDA_HAT_HOME`).
* Store: `artifacts/store/` (content-addressed immutable objects).
* Experiments: `artifacts/experiments/{experiment}/runs/`.
* See `docs/output_management.md` for complete layout details.

## Don't

* Don't run entry points directly; always use `uv run` and prefer Parsl for workflows.
* Don't bump BlackJAX/JAX without updating code paths.
* Don't add back-compat shims; fail fast.
* Backwards compatibility is anathema

## Samplers

**Implemented**:
- HMC, MCLMC: Full-batch MCMC with adaptation
- SGLD: Stochastic gradient Langevin dynamics with Adam/RMSProp preconditioning
- VI: Variational inference with pluggable algorithms (unified VIAlgorithm protocol)
  - MFA (default): Mixture of factor analyzers with STL + Rao-Blackwellized gradients
  - Flow: Normalizing flows via manifold-plus-noise construction (requires `--extra flowvi`)
    - RealNVP coupling flow (default), MAF, or NSF architectures
    - Low-rank latent space with orthogonal noise
    - Vmap-compatible (PRNG key issues resolved Nov 2025)
    - NOTE: HVP control variate deferred to future work
  - Work metrics: `n_full_loss` (MC samples for lambda), `n_minibatch_grads` (optimization steps), `sampler_flavour`
  - FGE tracking: `cumulative_fge = batch_size / n_data` per step (minibatch accounting)


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

* **JAX 0.7.2**

  * **Trees**: `jax.tree.map`, `jax.tree_util.tree_leaves`, `jax.flatten_util.ravel_pytree`.
  * **Random**: `jax.random.split(key, n)`, `jax.random.choice(key, n, shape=(k,), replace=False)`.
  * **Loops**: `jax.lax.scan(fn, carry, xs)` with `@jax.jit`.

These match how the repo already uses the libraries.
Do **not** switch to main-branch BlackJAX symbols or older `jax.tree_map`/`tree_util.tree_map`.
