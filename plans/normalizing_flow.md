Below is a concrete refactor plan that lets you plug in **new “VI‑flavoured” algorithms** (e.g., a reparameterization/flow‑based estimator) alongside your current *mixture‑of‑factor‑analysers* (MFA) VI, without disrupting the rest of the pipeline.

I anchor each recommendation to the places in your repo that will change, and I call out library choices (FlowJAX/NuX) and the Haiku↔Equinox question.

---

## 0) What you have today (relevant anchors)

* **Sampler entrypoint and dispatch.** `run_sampler(...)` routes to `run_vi(...)` when `sampler.name == "vi"`. That call wires up the minibatch/full‑data loss functions and passes your VI config and β,γ to the VI runner.
* **VI implementation.** All of the MFA logic (params, optimizer, per‑step ELBO, diagnostics) lives in `lambda_hat/variational.py`, with the public API `fit_vi_and_estimate_lambda(...)`. It returns `lambda_hat`, step traces, and extras.
* **VI configuration.** Hydra/OmegaConf YAML at `lambda_hat/conf/sample/sampler/vi.yaml` (e.g., `M`, `r`, `steps`, `gamma`, `eval_samples`, whitening, LR scheduling, etc.). This file already has a Stage‑3 “Advanced” block (e.g., `r_per_component`, `lr_schedule`) that we’ll re‑use.
* **TensorBoard diagnostics.** There are tests and a writer path that log VI metrics like `elbo`, `logq`, `pi_entropy`, `grad_norm`, etc., via `tensorboardX`. We should keep emitting a superset of these from any new VI family.
* **Toolchain pins.** JAX `~0.7.1`, Haiku, Optax. No Equinox in core deps yet.

---

## 1) The core refactor: make **VI a pluggable “family”**

Introduce a small, explicit interface for VI families. Keep it JAX‑first; don’t force Equinox globally.

> **New module:** `lambda_hat/vi/api.py`

```python
# lambda_hat/vi/api.py
from typing import Protocol, Any, Dict, Tuple, NamedTuple, Callable
import jax, jax.numpy as jnp

class VITrace(NamedTuple):
    scalars: Dict[str, jnp.ndarray]   # e.g., elbo[steps], logq[steps], ...
    extras: Dict[str, Any]            # family-specific (e.g., pi, D_sqrt)

class VIFamily(Protocol):
    def init(self, key, wstar_flat: jnp.ndarray, whitener, cfg) -> Any: ...
    def step(self, key, state, minibatch) -> Tuple[Any, Dict[str, jnp.ndarray]]: ...
    def final_mc(
        self, key, state, eval_samples: int
    ) -> Tuple[float, Dict[str, jnp.ndarray], Dict[str, Any]]:
        """Return (lambda_hat, traces_like, extras)"""
```

* `init/step/final_mc` mirrors your current `build_elbo_step`/scan and final evaluation inside `fit_vi_and_estimate_lambda`.
* Keep **diagnostic names stable** (`elbo`, `elbo_like`, `logq`, `grad_norm`, …) so the existing TB writing code keeps working, regardless of family.

> **New dispatcher:** `lambda_hat/vi/factory.py`

```python
def make_vi_family(cfg) -> "VIFamily":
    fam = getattr(cfg, "family", "mfa")
    if fam == "mfa":
        from .mfa import MFAFamily
        return MFAFamily(cfg)
    elif fam == "flow":
        from .flow import FlowFamily
        return FlowFamily(cfg)
    else:
        raise ValueError(f"Unknown VI family: {fam}")
```

> **Thin runner:** `lambda_hat/vi/run.py`

Wraps the shared book‑keeping (optim loop, averaging across chains, TB logging compatibility) and calls into the selected `VIFamily`.

Finally, change the call site:

* In `lambda_hat/sampling.py`, **replace** the direct call into `variational.fit_vi_and_estimate_lambda` with the unified runner:

```python
# from lambda_hat import vi (module)
from lambda_hat.vi.run import run_vi   # new

# existing behavior here:
run_result = run_vi(
    key,
    loss_mini,
    loss_full,
    params0_typed,
    data=(X_typed, Y_typed),
    config=cfg.sampler.vi,
    num_chains=cfg.sampler.chains,
    beta=beta,
    gamma=gamma,
)
```

This preserves the current sampler interface (`run_vi`), but **under the hood** it creates a family and routes through the common loop. The existing MFA code can move to `lambda_hat/vi/mfa.py` with minimal edits (largely slicing your current `variational.py` into `api.py` + `mfa.py`).

---

## 2) Config changes (backward‑compatible)

Extend `lambda_hat/conf/sample/sampler/vi.yaml` with a **family switch** and per‑family sections:

```yaml
# @package _global_.sampler
name: vi
vi:
  family: mfa             # "mfa" (default) | "flow"
  # --- common across families ---
  steps: 5000
  batch_size: 256
  lr: 0.01
  eval_every: 50
  eval_samples: 64
  gamma: 0.001
  dtype: float32
  whitening_mode: none      # "none"|"rmsprop"|"adam"
  clip_global_norm: 5.0
  tensorboard: false

  # --- MFA-specific (existing) ---
  M: 8
  r: 2
  r_per_component: null
  lr_schedule: null
  lr_warmup_frac: 0.05
  entropy_bonus: 0.0
  alpha_dirichlet_prior: null
  prune_threshold: 1e-3

  # --- FLOW-specific (new) ---
  backend: flowjax          # "flowjax" | "nux"
  d_latent: 8
  sigma_perp: 1e-3
  flow:
    template: maf           # "maf"|"realnvp"|"spline_coupling"|...
    depth: 4
    hidden: [128, 128]
    bins: 8
    init_scale: 1e-2
```

Hydra overrides remain as today (e.g., `{ name: vi, overrides: { M: 8, r: 2, gamma: 1e-3 } }`), and now you can also do `{ name: vi, overrides: { family: flow, d_latent: 8, backend: flowjax } }`.

---

## 3) Flow‑based VI family (reparameterization) with **FlowJAX** (preferred)

### Why FlowJAX first?

* FlowJAX ships on PyPI (`pip install flowjax`) and is **built on Equinox**, with distributions/bijections as PyTrees that interoperate cleanly with JAX autodiff and transformations. It includes examples for **variational inference** and an ELBO loss helper. ([Daniel Ward][1])
* NuX is useful, but its own README says **the PyPI package is outdated** and recommends cloning the repo; it also advertises an “improved version” GeneraX. That makes it a less drop‑in dependency for your pipeline. ([GitHub][2])

### Adapter design

> **New:** `lambda_hat/vi/flow.py`

* Build a **local‑posterior flow** in a *low‑dimensional latent space* (`d_latent`) and lift into parameter space with the **manifold‑plus‑noise** map you drafted earlier (orthonormal basis (U), complement (E_\perp), diagonal preconditioner (D), small orthogonal noise (\sigma_\perp)). The sample & log‑density are computed with a **block‑triangular** Jacobian; only the latent flow’s log‑det is nontrivial.
* Implement this as a small Equinox module that **wraps** a FlowJAX flow and exposes **`sample_and_log_prob(key, shape=None)` returning (w, log_q(w))** so we can plug straight into FlowJAX’s `ElboLoss`. FlowJAX distributions are Equinox Modules and already standardize on `sample_and_log_prob`. ([Daniel Ward][3])

Skeleton:

```python
# lambda_hat/vi/flow.py
import equinox as eqx
import jax, jax.numpy as jnp
from flowjax.flows import MAF, RealNVP, NSF_AR  # examples

class InjectiveLift(eqx.Module):
    flow: eqx.Module        # a flowjax flow in R^{d_latent}
    U: jax.Array            # (d, d_latent) orthonormal basis
    E_perp: jax.Array       # (d, d - d_latent) orthonormal complement
    D_sqrt: jax.Array       # (d,) preconditioner^{1/2}
    sigma_perp: float
    wstar: jax.Array        # (d,)

    def sample_and_log_prob(self, key, shape=()) -> tuple[jax.Array, jax.Array]:
        k1, k2 = jax.random.split(key)
        # latent sample + logq_z from the flow
        z, logq_z = self.flow.sample_and_log_prob(k1, shape=shape)

        # orthogonal gaussian noise
        eps = jax.random.normal(k2, shape + (self.E_perp.shape[1],))
        w = self.wstar + self.D_sqrt * (self.U @ z + self.E_perp @ (self.sigma_perp * eps))

        # explicit log|det| terms from lift:
        log_det = -0.5 * jnp.sum(jnp.log(jnp.square(self.D_sqrt))) \
                  - (self.E_perp.shape[1]) * jnp.log(self.sigma_perp)

        logq_w = logq_z + log_det  # block-triangular Jacobian
        return w, logq_w
```

* **Latent flow choice.** For small `d_latent`, use neural spline flows / autoregressive (MAF/NSF) or coupling (RealNVP). These are first‑class in FlowJAX. You can initialize them near identity (`init_scale`) to keep early steps stable. ([Daniel Ward][1])
* **Objective.** Compose your unnormalized local posterior
  (\log p(w);=; -n\beta L_n(w) - \tfrac{\gamma}{2}|w-w^*|^2)
  (you already use this form for MCMC), and optimize **ELBO** using FlowJAX’s `ElboLoss(target_logprob, num_samples=K)` or your own equivalent that draws a minibatch for `L_n`. You already have closures for full/minibatch losses wired up in the sampler.

Training loop inside `FlowFamily.step(...)`:

```python
# Pseudocode: pathwise ELBO with FlowJAX loss helper
def step(self, key, state, minibatch):
    # Compose minibatch-scaled log p(w) using your existing loss fns:
    def logpost(w):
        Xb, Yb = minibatch
        Ln_b = self.loss_batch_fn(self.unravel(w), Xb, Yb)
        Ln_est = (self.n_data / self.batch_size) * Ln_b  # unbiased minibatch scaling
        return - self.n_data * self.beta * Ln_est - 0.5 * self.gamma * jnp.sum((w - self.wstar)**2)

    # ELBO = E_q[log p(w) - log q(w)]. Pathwise via reparam in sample_and_log_prob
    def elbo(flow_params, key):
        dist = eqx.tree_at(lambda f: f, self.dist, flow_params)  # replace params
        w, logq = dist.sample_and_log_prob(key)
        return logpost(w) - logq

    # Maximize ELBO with Optax (same optimizer infra as MFA)
    loss = -jnp.mean(jax.vmap(elbo, in_axes=(None, 0))(state.flow_params, keys_for_K))
    grads = jax.grad(lambda p: loss)(state.flow_params)
    updates, opt_state = self.optimizer.update(grads, state.opt_state, state.flow_params)
    new_params = optax.apply_updates(state.flow_params, updates)
    new_state = state._replace(flow_params=new_params, opt_state=opt_state, step=state.step+1)
    metrics = {"elbo": -loss, "grad_norm": optax.global_norm(grads)}
    return new_state, metrics
```

* **Diagnostics.** Emit the common VI metrics (`elbo`, `logq`, `grad_norm`, `cumulative_fge`) plus flow‑specific ones, e.g., `logdet_latent` summary, `sigma_perp`, and a small covariance proxy along `U`. These land in the same TB channel you already write.

### Minimal Haiku↔Equinox coexistence

You **don’t** need to migrate your target networks to Equinox. Keep Haiku for the model; the VI flow stack can be Equinox‑based because both are just PyTrees in JAX.

* You already flatten/unflatten Haiku params; reuse that to evaluate `L_n(w)` from a flat `w`.
* Only the flow object and its params live in Equinox; your loss closures and runners remain the same.

### Version pin (important)

* The latest Equinox release explicitly targets **JAX 0.7.2** and warns that **JAX 0.7.0/0.7.1 have known bugs**. Since your repo pins `jax~=0.7.1`, I recommend bumping to `0.7.2` when enabling FlowJAX. ([GitHub][4])
* FlowJAX itself is Equinox‑based with standard `sample_and_log_prob` APIs. ([Daniel Ward][3])

---

## 4) Optional: NuX adapter

If you do want a NuX‑backed flow:

> **New:** `lambda_hat/vi/flow_nux.py`

* Mirror the same `InjectiveLift` concept around a NuX flow.
* **Caveat:** NuX’s own README states that **the pip package is outdated** and suggests cloning; that’s at odds with your reproducible CLI/sweeps. If you include it, gate it behind an **extra** and keep FlowJAX as the default backend. ([GitHub][2])

---

## 5) Packaging and dependency options

Update `pyproject.toml`:

```toml
[project.optional-dependencies]
flowvi = [
  "equinox>=0.13.1",   # pairs with JAX 0.7.2
  "flowjax>=0",        # use current release
]
nuxvi = [
  "git+https://github.com/Information-Fusion-Lab-Umass/NuX.git",
]

# Bump JAX if enabling flowvi
cpu = [ "jax~=0.7.2" ]
cuda12 = [ "jax[cuda12_local]~=0.7.2" ]
```

(Keep Haiku as‑is for your models; no conflict with Equinox.)

---

## 6) Inference loop wiring (unchanged call sites)

`run_vi(...)` (new) should align to the existing shape used by the CLI/runner, so `lambda_hat/entrypoints/sample.py` and `sampling_runner` stay unchanged. You continue to create the **full/minibatch** loss closures once, and the family uses them internally for ELBO.

**Final MC** (`FlowFamily.final_mc`) should mimic the current MFA path:

* Draw `eval_samples` of `w` from the learned q(w).
* Compute (E_q[L_n(w)]) (on full data) possibly with your **HVP control‑variate** and trace correction to reduce variance (that utility already exists). Return `lambda_hat = nβ (E_q[L_n] - L_n(w^*))`.

---

## 7) Diagnostics & TensorBoard (keep one place)

Keep the **one TB writer path** and log a superset of today’s VI scalars so dashboards continue to work:

* Common: `elbo`, `elbo_like` (when separated), `logq`, `grad_norm`, `cumulative_fge`.
* MFA: `pi_min/max/entropy`, `D_sqrt_*`, `A_col_norm_max`.
* Flow: `sigma_perp`, `latent_logdet_mean`, `latent_scale_*`.

You already have the logging loop and tests; just ensure the new traces compose into the same dict keys.

---

## 8) Tests (fast “smoke” + numerical checks)

* **Flow smoke test** (`tests/test_vi_flow_smoke.py`): tiny quadratic or tiny MLP, `d_latent=2`, `steps≈50`, assert finite `lambda_hat`, finite `elbo`, and that TB event files exist (mirrors `test_vi_tensorboard_smoke`).
* **Interface compatibility test**: ensure both families return the same trace keys needed by the writer (subset).
* **Minibatch correctness**: test that using minibatch scaling gives the same mean `E_q[L_n]` (within noise) as full‑batch on a small dataset.

---

## 9) Documentation updates

* **`docs/vi.md`**: add “Families: MFA / Flow” and a short *When to use which* table.
* **New `docs/vi_flow.md`**: include the manifold‑plus‑noise diagram, config snippet, and a note that FlowJAX/Equinox is optional and gated behind `[flowvi]`.
* Update the **examples YAML** to show `family: flow` override for your quick preset. The same CLI remains (`lambda-hat-sample` etc.), which is good for adoption.

---

## 10) Haiku → Equinox migration (should we?)

**Short answer:** not now.

* FlowJAX works fine **alongside** Haiku. Both are PyTrees in JAX; your loss closures already accept flat `w` and unflatten via `unravel_fn`.
* A **full migration** to Equinox would be justified only if (a) you want to share model submodules between the flow and the target NN, or (b) you want to remove Haiku entirely. Otherwise you add maintenance risk for little gain.
* If/when you do migrate, plan it as a **separate, mechanical pass** on `lambda_hat/models.py`—but keep this VI refactor independent.

> **Important version note:** Equinox v0.13.1 highlights compatibility with **JAX 0.7.2** and warns against `0.7.0/0.7.1` due to upstream bugs; if you bring Equinox into the tree, bump JAX to `0.7.2`. ([GitHub][4])

---

## 11) What about NuX?

NuX can be supported via a second adapter (same interface). Just be aware:

* NuX’s README says the **pip package is outdated; clone is recommended**; that’s awkward for your **sweeps** and **remote backends** (Modal/SLURM) because it complicates reproducibility. Keep it as an **optional extra** only. ([GitHub][2])

---

## 12) Concrete TODO checklist

**A. Interfaces & wiring**

* [ ] Add `lambda_hat/vi/{api.py,factory.py,run.py}` (new).
* [ ] Move current VI to `lambda_hat/vi/mfa.py` implementing `VIFamily` via thin wrapper of today’s `fit_vi_and_estimate_lambda` internals. Keep all current metrics.
* [ ] Replace direct MFA call in `sampling.py` with `from lambda_hat.vi.run import run_vi`.

**B. Flow family**

* [ ] Add `lambda_hat/vi/flow.py` with `FlowFamily` + `InjectiveLift` (FlowJAX backend).
* [ ] Implement `init()`:

  * estimate (U) cheaply (or start with random/identity for a smoke test), set `D_sqrt` from Adam/RMSProp whitener you already have, set `sigma_perp` from config.
* [ ] Implement `step(...)` using FlowJAX‑style ELBO (pathwise). Start with `K=1..4` samples; add antithetic/QMC later.
* [ ] Implement `final_mc(...)` (re‑use your **HVP control variate** if enabled in config).

**C. Config & packaging**

* [ ] Extend `lambda_hat/conf/sample/sampler/vi.yaml` with `family`, and a `flow` block.
* [ ] Add optional deps `[flowvi] = {equinox, flowjax}` and bump JAX to 0.7.2 when `[flowvi]` is used.

**D. Diagnostics**

* [ ] Keep TB logging in one place; for Flow family add `sigma_perp`, `latent_logdet_mean`, and a small covariance summary along span(U). Tests similar to `test_vi_tensorboard.py` should pass.

**E. Tests**

* [ ] `tests/test_vi_flow_smoke.py`: finite `lambda_hat`, TB file written (mirror existing tests).
* [ ] Interface compatibility: both families produce the keys your writer expects.

---

## 13) Library references (for reviewers)

* **FlowJAX overview**: Equinox‑based flows & distributions with VI example and `sample_and_log_prob` convention. ([Daniel Ward][1])
* **FlowJAX API**: Distributions are Equinox Modules (PyTrees). ([Daniel Ward][3])
* **Equinox release**: v0.13.1 pairs with **JAX 0.7.2**; avoid `0.7.0/0.7.1`. ([GitHub][4])
* **NuX README**: pip package is outdated; clone recommended; mentions “improved version” (GeneraX). ([GitHub][2])

---

### Net effect

* You keep **one** sampler name (`vi`), switchable **families** (`mfa`, `flow`) behind the same public runner and the same diagnostics surface.
* You can onboard FlowJAX **without** migrating Haiku or touching downstream analysis/TensorBoard.
* If Flow‑VI proves itself, you have a clean path to experiment with alternative backends (NuX) by adding one file and one `[extra]`.

If you’d like, I can sketch the exact `FlowFamily` signatures against your `loss_batch_fn`/`loss_full_fn` closures so the minibatch scaling matches your current MFA implementation and the **work/FGE metrics** continue to be tracked uniformly.

[1]: https://danielward27.github.io/flowjax/ "FlowJAX — FlowJAX"
[2]: https://github.com/Information-Fusion-Lab-Umass/NuX "GitHub - Information-Fusion-Lab-Umass/NuX: Normalizing Flows using JAX"
[3]: https://danielward27.github.io/flowjax/api/distributions.html "Distributions — FlowJAX"
[4]: https://github.com/patrick-kidger/equinox/releases "Releases · patrick-kidger/equinox"
