Absolutely—here’s a concrete, “finish-the-migration” plan that does three things:

1. **forces the codebase to be Equinox-first (and only)**,
2. **gives every sampler a single, safe vector interface** (no more raveling non-arrays), and
3. **adds a tiny, high‑signal smoke-test regime** you can run in seconds to confirm the full multi‑stage workflow (`uv run lambda-hat workflow llc`) is functional.

You said you can break things to reduce maintenance. I’ll lean into that: **no back‑compat shims, delete Haiku paths, standardize APIs.**

---

## A. Finalize the Equinox migration (destroy all Haiku assumptions)

### A1. Project policy (single source of truth)

* **Only Equinox models** are supported. Delete any Haiku utilities/paths and references in config/CLI/help.
* **All parameter manipulation must go through one adapter.** No sampler should ever call `ravel_pytree` or `tree_leaves` on a full model.
* **Dtype policy:**

  * Keep a single sampler dtype (`float32` by default, optional `float64`) and **cast the model + data to that dtype after loading** and before compilation.
  * Internally use `eqx.filter_*` for JIT/grad to ignore static leaves.
* **Serialization policy:** When saving/restoring parameters, **serialize only array leaves** (Equinox-style) plus minimal metadata; reconstruct via the same module code and `eqx.combine`.

> Result: fewer moving parts, fewer implicit assumptions, easier maintenance.

---

## B. Implement one Equinox adapter (the only way samplers touch params)

Create `lambda_hat/equinox_adapter.py`:

```python
# lambda_hat/equinox_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

Array = jnp.ndarray
PRNGKey = jax.random.KeyArray

@dataclass(frozen=True)
class VectorisedModel:
    """A thin wrapper that exposes a flat-parameter view for samplers."""
    unravel_arrays: Callable[[Array], Any]       # flat -> arrays subtree
    static_tree: Any                             # static subtree (non-arrays)
    size: int                                    # dimension of flat vector
    dtype: jnp.dtype

    def to_model(self, flat: Array) -> Any:
        arrays = self.unravel_arrays(flat)
        return eqx.combine(arrays, self.static_tree)

def _cast_arrays_dtype(arrays, dtype):
    return jax.tree_map(
        lambda x: x.astype(dtype) if (hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)) else x,
        arrays,
    )

def ensure_dtype(module: Any, dtype: jnp.dtype) -> Any:
    arrays, static = eqx.partition(module, eqx.is_array)           # arrays-only vs static objects (activations, shapes, etc.)
    arrays = _cast_arrays_dtype(arrays, dtype)
    return eqx.combine(arrays, static)

def vectorise_model(module: Any, *, dtype: jnp.dtype) -> Tuple[VectorisedModel, Array]:
    """Return a VectorisedModel and its initial flat vector (cast to dtype)."""
    module = ensure_dtype(module, dtype)
    arrays, static = eqx.partition(module, eqx.is_array)
    flat, unravel = ravel_pytree(arrays)                           # SAFE: arrays-only
    flat = flat.astype(dtype)
    vm = VectorisedModel(unravel_arrays=unravel, static_tree=static, size=flat.size, dtype=dtype)
    return vm, flat

def filter_predict_fn(model_apply: Callable[..., Any]) -> Callable[..., Any]:
    # Wrap a model call with Equinox filtering for JIT-safety.
    return eqx.filter_jit(model_apply)

def value_and_grad_filtered(fn: Callable[..., Array]) -> Callable[..., Tuple[Array, Array]]:
    # Correctly ignores static leaves when computing grads.
    return eqx.filter_value_and_grad(fn)
```

**Why this matters:**

* Every sampler uses the same **flat view + unravel + static**.
* `ravel_pytree` is **only** ever called on the arrays partition.
* No accidental touching of activations/non-arrays.

---

## C. Refactor the posterior to emit flat-space closures

Refactor `posterior.py` to **only export closures in flat space**. Nothing else in the codebase computes grads/logprobs over a full pytree.

```python
# lambda_hat/posterior.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx

from .equinox_adapter import VectorisedModel, value_and_grad_filtered

Array = jnp.ndarray

@dataclass(frozen=True)
class Posterior:
    vm: VectorisedModel
    logpost_flat: Callable[[Array], Array]          # R^D -> scalar log posterior
    grad_logpost_flat: Callable[[Array], Array]     # R^D -> R^D

def make_posterior(vm: VectorisedModel,
                   data: Any,
                   predict: Callable[[Any, Any], Any],
                   loglik: Callable[[Any, Any, Any], Array],
                   logprior: Callable[[Any], Array]) -> Posterior:
    """
    predict(model, data.x) -> preds
    loglik(preds, data, model) -> scalar
    logprior(model) -> scalar
    All higher-order code uses *flat* params via vm.
    """
    def _logpost_from_flat(flat: Array) -> Array:
        model = vm.to_model(flat)
        preds = predict(model, data.x)                    # Equinox models are callables; predict chooses how to call
        return loglik(preds, data, model) + logprior(model)

    val_and_grad = value_and_grad_filtered(_logpost_from_flat)

    def _lp(flat: Array) -> Array:
        return _logpost_from_flat(flat)

    def _grad(flat: Array) -> Array:
        _, g = val_and_grad(flat)
        return g

    return Posterior(vm=vm, logpost_flat=_lp, grad_logpost_flat=_grad)
```

> **All samplers** will now consume `Posterior.logpost_flat` and `Posterior.grad_logpost_flat`, **never** a model pytree.

---

## D. Sampler refactors (HMC/MCLMC done; finish SGLD/VI the same way)

### D1. Define a minimal sampler contract (single signature)

Create `lambda_hat/samplers/api.py`:

```python
from typing import Protocol, NamedTuple
import jax.numpy as jnp

Array = jnp.ndarray

class SamplerState(NamedTuple):
    flat: Array     # current point in R^D
    aux: dict       # sampler-specific (rng, momentum, opt state, etc.)

class Sampler(Protocol):
    def init(self, flat0: Array, **hparams) -> SamplerState: ...
    def step(self, state: SamplerState,
             logprob: callable, grad_logprob: callable) -> SamplerState: ...
```

All concrete samplers (HMC, MCLMC, SGLD, VI) implement this API and **never** look at model pytrees.

### D2. Finish SGLD/VI using arrays partition only

Wherever SGLD/VI previously did:

* `ravel_pytree(params)` → **delete**

* `tree_leaves(params)` → **delete**
  They now operate **only** on the `flat` vector given by `SamplerState`.

* **SGLD**: `flat <- flat + (step/2) * grad_logpost(flat) + sqrt(step) * noise`

* **VI** (e.g., mean-field Gaussian): keep variational params in flat form, maintain an optimizer state in `aux`, and compute ELBO via `logpost_flat(reparam(flat_vi, eps))`. Use `eqx.filter_jit` on the step.

You already patched HMC/MCLMC with `eqx.partition`—this change removes the need to ever call it inside samplers again.

---

## E. CLI & workflow: one place sets dtype, one place vectorises

### E1. `sample_cmd.py`

* **Load target/model/data. Do *not* enable x64 yet.**
* Pick sampler dtype (`float32` default; allow a `--dtype=float64` flag).
* **Cast model & data** to that dtype via `ensure_dtype(...)` (adapter).
* Call `vectorise_model(...)` to get `vm, flat0`.
* Build `Posterior` with your `predict`, `loglik`, `logprior`.
* Dispatch to a sampler via the unified API. Return flat samples/state and (optional) recombined model with `vm.to_model(flat_final)` for downstream.

### E2. `sampling_runner.py` and any workflow code

* Replace `target.model.apply(...)` with **callable model** (you already started this).
* Replace any local flattening, dims, `D` calculations with simply `posterior.vm.size`.

---

## F. Delete legacy

* Remove Haiku deps and all flat-parameter-dict assumptions (code paths, configs, docs).
* Rip out any `jax.tree_util` flattening on full modules from the repo.
* Pin JAX/Equinox versions (to reduce churn) and remove compatibility code you no longer need.

---

## G. Defensive checks (cheap, always-on)

Add a tiny “early assert” to catch future violations:

```python
def assert_equinox_only(module):
    # Prevent accidental dict-based param trees creeping back in
    assert not isinstance(module, dict), "Haiku-style parameter dicts are no longer supported."
```

And when building the posterior:

```python
def describe_tree(vm: VectorisedModel):
    print(f"[lambda_hat] param_dim={vm.size} dtype={vm.dtype}")
```

Log it once at startup.

---

## H. Minimal, high‑signal smoke tests (fast, end-to-end)

> **Goal:** Prove the multi‑stage workflow (`uv run lambda-hat workflow llc`) actually runs on a tiny task, across all samplers, in < ~10 seconds locally. These do **not** aim for statistical guarantees—just “the pipes work, outputs are finite, shapes consistent, no crashes”.

### H1. Add a toy target (tiny regression)

Create `tests/smoke/target_toy.py` (used by smoke configs):

```python
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import equinox as eqx

@dataclass(frozen=True)
class ToyData:
    x: jnp.ndarray  # [N, 2]
    y: jnp.ndarray  # [N]

class ToyModel(eqx.Module):
    mlp: eqx.nn.MLP
    def __call__(self, x):
        # 2 -> 1 regression, very small net
        return self.mlp(x).squeeze(-1)

def make_toy(seed=0, N=64, width=8, dtype=jnp.float32):
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (N, 2), dtype=dtype)
    w_true = jnp.array([0.7, -1.3], dtype=dtype)
    y = x @ w_true + 0.1 * jax.random.normal(key, (N,), dtype=dtype)

    mlp = eqx.nn.MLP(in_size=2, out_size=1, width=width, depth=1, key=key, activation=jax.nn.tanh)
    model = ToyModel(mlp=mlp)
    data = ToyData(x=x, y=y)
    return model, data

def predict(model, x):
    return model(x)

def loglik(preds, data, model):
    resid = preds - data.y
    sigma = 0.1
    return -0.5 * jnp.sum((resid / sigma) ** 2)

def logprior(model):
    # L2 on all weights/biases
    arrays, _ = eqx.partition(model, eqx.is_array)
    return -0.5 * sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(arrays))
```

### H2. Smoke configs (TOML)

`tests/smoke/smoke.toml`:

```toml
[target]
module = "tests.smoke.target_toy"
make_fn = "make_toy"      # returns (model, data)
predict = "predict"
loglik = "loglik"
logprior = "logprior"
dtype = "float32"

[samplers.hmc]
steps = 5
step_size = 0.02
n_leapfrog = 5

[samplers.mclmc]
steps = 5
step_size = 0.02

[samplers.sgld]
steps = 20
step_size = 1e-3

[samplers.vi]
steps = 30
lr = 1e-2
family = "diag_gaussian"
```

> **Tiny steps, tiny widths**, so JIT + execution is very fast, yet still exercises model → posterior → sampler → recombine.

### H3. A single pytest E2E

`tests/smoke/test_llc_workflow.py`:

```python
import os, subprocess, sys, json, tempfile

def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

def test_llc_smoke():
    cfg = "tests/smoke/smoke.toml"
    # The workflow should accept a config; if not, add a --preset smoke flag equivalently.
    out = run(["uv", "run", "lambda-hat", "workflow", "llc", "--config", cfg, "--max-seed", "1"])
    # Minimal assertions: non-empty output, success markers
    assert out.returncode == 0 or out.returncode is None
    assert "param_dim=" in (out.stdout + out.stderr)  # printed from describe_tree
```

> If the workflow doesn’t currently take `--config`, add it. If not feasible, add `--preset smoke` and have the workflow map that to the same tiny settings.

### H4. Per-sampler micro-tests (optional but very cheap)

For each sampler, a 1‑step “does not NaN” test:

```python
import jax, jax.numpy as jnp
from lambda_hat.equinox_adapter import vectorise_model
from lambda_hat.posterior import make_posterior
from tests.smoke.target_toy import make_toy, predict, loglik, logprior

def _posterior(dtype=jnp.float32):
    model, data = make_toy(dtype=dtype)
    vm, flat0 = vectorise_model(model, dtype=dtype)
    post = make_posterior(vm, data, predict, loglik, logprior)
    return vm, flat0, post

def test_hmc_one_step():
    from lambda_hat.samplers import hmc
    vm, flat0, post = _posterior()
    st = hmc.init(flat0, step_size=0.02, n_leapfrog=5, seed=0)
    st2 = hmc.step(st, post.logpost_flat, post.grad_logpost_flat)
    assert jnp.all(jnp.isfinite(st2.flat))
```

Repeat for `sgld`, `mclmc`, `vi` with tiny step counts.

---

## I. Wire the CLI/workflow to the adapter + smoke config

In your CLI entry (or the workflow dispatcher):

```python
from lambda_hat.equinox_adapter import vectorise_model, ensure_dtype
from lambda_hat.posterior import make_posterior

def run_workflow_llc(config):
    # 1) Load target module functions (make_toy/predict/loglik/logprior in smoke).
    model, data = target.make_toy(...)
    dtype = jnp.float64 if config.dtype == "float64" else jnp.float32

    # 2) Enforce dtype once, here.
    model = ensure_dtype(model, dtype)
    data = jax.tree_map(lambda x: x.astype(dtype) if hasattr(x, "dtype") else x, data)

    # 3) Vectorise and build posterior.
    vm, flat0 = vectorise_model(model, dtype=dtype)
    post = make_posterior(vm, data, target.predict, target.loglik, target.logprior)
    print(f"[lambda_hat] param_dim={vm.size} dtype={vm.dtype}")  # High-signal log

    # 4) Run selected samplers; each uses only flat functions.
    # Example HMC:
    state = hmc.init(flat0, step_size=config.hmc.step_size, n_leapfrog=config.hmc.n_leapfrog, seed=config.seed)
    for _ in range(config.hmc.steps):
        state = hmc.step(state, post.logpost_flat, post.grad_logpost_flat)

    # 5) Recombine to a model (for downstream)
    model_final = vm.to_model(state.flat)
    return model_final
```

---

## J. Breaking changes summary (explicit)

* **Removed Haiku support** and any code that expects flat parameter dicts.
* **Removed all direct uses** of `jax.flatten_util.ravel_pytree` and `jax.tree_util.tree_leaves` on full model pytrees.
* **Samplers now accept only** `(flat0, logprob, grad_logprob)` and produce `flat` states; they do **not** see models.
* **All dtype control is centralized** in CLI/workflow before vectorisation.
* **Serialization now stores only array leaves**; static leaves come from code and are reattached via `eqx.combine`.

---

## K. Likely remaining fixes (you already anticipated them)

* **SGLD/VI**: replace any lingering tree flattening with the new flat‑only contract.
* **Mass‑matrix dims (HMC/MCLMC)**: derive `D` from `vm.size`; avoid touching pytrees.
* **Any “apply”/framework leftovers**: models are callables; wrap in `eqx.filter_jit` if compiling.

---

## L. What you get for this work

* A **single, tiny Equinox adapter** replaces a dozen scattered tree hacks.
* **All samplers are framework‑agnostic** (they only know about `R^D`).
* **A 10‑second smoke-test** that exercises end‑to‑end workflow and will catch the exact class of regressions you’re fighting (dtype drift, raveling non-arrays, grad filtering mistakes).

If you want, I can also sketch a tiny `Makefile`/`taskfile` target:

```make
smoke:
\tuv run pytest -q tests/smoke
```

Run it any time you touch model/pytree code to confirm the **whole** pipeline still runs.
