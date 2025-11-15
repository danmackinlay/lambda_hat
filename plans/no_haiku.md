**Short answer (optimize for lowest maintenance):**
Unify on **Equinox** and delete **Haiku** from the repo. You already pulled in Equinox via FlowJAX; the only Haiku usage is a small MLP wrapper plus a Haiku‑specific NPZ serializer and a couple of call sites. Porting that to Equinox is straightforward and touches a handful of files. You can then drop all the “dynamic dependency” machinery around FlowJAX imports and extras.

Below I’ll (1) justify the recommendation from your tree, (2) give a concrete migration plan (copy‑pasteable steps), and (3) note a zero‑effort stopgap if you truly can’t touch model code today.

---

## Why Equinox-only is the least maintenance for *this* repo

* **Haiku is used narrowly.** The Haiku footprint is a single MLP module and a `hk.transform()` factory; both are thin and easy to reproduce in Equinox. See `lambda_hat/models.py` (“Modernized model definitions using Haiku”) and its `MLP` & `build_mlp_forward_fn` which just create a stack of `hk.Linear` + optional norms/activations, then call `hk.transform`.
  The Haiku‑style call is also assumed inside `lambda_hat/losses.py` (calls `model_apply(params, None, X)`), which is trivial to adapt to Equinox.

* **Your sampling/VI glue already expects flattened PyTrees and pure JAX.** Training uses Optax on arbitrary PyTrees (no Haiku specifics), so swapping in Equinox params works as‑is.

* **FlowJAX already commits you to Equinox.** You added FlowJAX and deliberately lazy‑imported the flow VI to avoid making it a hard dep: the registry comment literally says “flow is imported lazily in get() to avoid requiring flowjax.” If we standardize on Equinox, we can delete this complexity and either hard‑depend on FlowJAX or keep a single clear import error.

* **Haiku-specific serialization is the only real blocker—and it’s small.** `lambda_hat/target_artifacts.py` flattens Haiku’s `{'module': {'param': array}}` structure into a key/value NPZ and reconstructs it the same way; replacing this with Equinox’s partition/serialize pattern is contained to that file.

* **Call sites are localized.** The Haiku factory is referenced in exactly the places where you build the target and where you rehydrate the model for sampling (`lambda_hat/targets.py` and `lambda_hat/entrypoints/sample.py`). Both pass `model.apply` into loss creation; they’re the only places you need to update to Equinox.

* **The “llc” package isn’t tied to Haiku at all.** It uses a pure‑JAX param dict MLP; you don’t have to touch it for this change (optionally dedupe later).

---

## Recommended plan (do this once, then you’re done)

### 0) Decide the packaging stance

* **Make Equinox mandatory** in the environment (base install). Remove Haiku entirely. (FlowJAX can remain optional if you like, but you can also make it mandatory to eliminate extras entirely.)
* With Equinox mandatory, delete the lazy import path in the VI registry and just import the flow implementation directly (or keep one `try/except` with a friendly error if you prefer it optional). The current lazy import exists only to avoid a hard FlowJAX dep.

### 1) Add an Equinox MLP and factory

Create `lambda_hat/nn_eqx.py` (or rename `models.py`) with a small Equinox module. It mirrors your current MLP: widths, activation, bias, optional layernorm; residuals are easy to keep if you want. (You already compute widths elsewhere.) The gist:

```python
# lambda_hat/nn_eqx.py
from typing import List, Callable, Optional
import jax, jax.numpy as jnp, equinox as eqx

def _act(name: str) -> Callable:
    return {"relu": jax.nn.relu, "tanh": jnp.tanh, "gelu": jax.nn.gelu, "identity": lambda x: x}[name]

class EqxMLP(eqx.Module):
    layers: list
    out: eqx.nn.Linear
    act: Callable
    layernorm: bool = False
    skip: bool = False
    residual_period: int = 2

    def __init__(self, in_dim: int, widths: List[int], out_dim: int,
                 activation="relu", bias=True, layernorm=False, skip=False,
                 residual_period=2, *, key):
        keys = jax.random.split(key, len(widths) + 1)
        self.act = _act(activation)
        self.layernorm = layernorm
        self.skip = skip
        self.residual_period = residual_period

        d = in_dim
        layers = []
        for i, w in enumerate(widths):
            layers.append(eqx.nn.Linear(d, w, use_bias=bias, key=keys[i]))
            if layernorm:
                layers.append(eqx.nn.LayerNorm(w, axis=-1))
            d = w
        self.layers = layers
        self.out = eqx.nn.Linear(d, out_dim, use_bias=bias, key=keys[-1])

    def __call__(self, x):
        h = x
        for i, lyr in enumerate(self.layers):
            h_new = self.act(lyr(h)) if not isinstance(lyr, eqx.nn.LayerNorm) else lyr(h)
            # simple residuals if you want parity with llc MLP:
            # if self.skip and (i % self.residual_period == self.residual_period - 1):
            #     h = (h if h.shape[-1] == h_new.shape[-1] else jnp.pad(...)) + h_new
            # else:
            h = h_new
        return self.out(h)

def build_mlp(in_dim: int, widths: List[int], out_dim: int, *,
              activation="relu", bias=True, layernorm=False, skip=False,
              residual_period=2, key) -> EqxMLP:
    return EqxMLP(in_dim, widths, out_dim, activation, bias, layernorm, skip, residual_period, key=key)

def count_params(model) -> int:
    params, _ = eqx.partition(model, eqx.is_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
```

This replaces your Haiku `MLP` + `build_mlp_forward_fn` + `count_params` trio.

### 2) Replace Haiku factory at call sites

* In **`lambda_hat/targets.py`** and **`lambda_hat/entrypoints/sample.py`**, replace:

```python
from lambda_hat.models import build_mlp_forward_fn, count_params
...
model = build_mlp_forward_fn(...); params_init = model.init(subkey, X[:1])
...
loss_full, loss_minibatch = make_loss_fns(model.apply, X, Y, ...)
```

with:

```python
from lambda_hat.nn_eqx import build_mlp, count_params
...
model = build_mlp(..., key=subkey)                       # Equinox module
...
def predict(params_or_model, Xb): return params_or_model(Xb)   # Equinox call
loss_full, loss_minibatch = make_loss_fns(predict, X, Y, ...)
```

These two files are the main Haiku call sites.

> Note: your training loop already updates generic PyTrees via Optax, so using the Equinox module as “params” works unchanged.

### 3) Update loss helpers to Equinox call signature

Right now `lambda_hat/losses.py` assumes the Haiku `apply(params, None, X)` signature. Make it generic: “a callable that takes (params_or_model, X)”. That’s a one‑line change at the call sites inside `make_loss_fns`:

```python
# lambda_hat/losses.py
def make_loss_fns(predict, X, Y, loss_type="mse", noise_scale=0.1, student_df=4.0):
    if loss_type == "mse":
        def full(params):     return jnp.mean((predict(params, X)  - Y)  ** 2)
        def minibatch(p, Xb, Yb): return jnp.mean((predict(p, Xb) - Yb) ** 2)
    ...
```

Previously it called `model_apply(params, None, X)`; now it calls `predict(params, X)`.

### 4) Swap the artifact format to Equinox

* Replace the Haiku‑specific NPZ flattener in **`lambda_hat/target_artifacts.py`** with Equinox partition/serialization:

  * At save time:

    ```python
    import equinox as eqx, jax, numpy as np

    params, static = eqx.partition(model, eqx.is_array)
    # model params -> bytes
    with open(tdir / "model.eqx", "wb") as f:
        eqx.tree_serialise_leaves(f, params)
    # persist statics via JSON meta (your meta.json already exists), or pickle a small "static.eqx":
    with open(tdir / "static.eqx", "wb") as f:
        eqx.tree_serialise_leaves(f, static)
    ```
  * At load time:

    ```python
    # reconstruct model architecture from meta (same as today)
    model = build_mlp(..., key=jax.random.PRNGKey(0))       # shapes only
    params, static = eqx.partition(model, eqx.is_array)
    with open(tdir / "model.eqx", "rb") as f:
        params = eqx.tree_deserialise_leaves(f, params)
    with open(tdir / "static.eqx", "rb") as f:
        static = eqx.tree_deserialise_leaves(f, static)
    model = eqx.combine(params, static)
    ```

  Then return `(X, Y, model, meta, tdir)` instead of `(X, Y, params, meta, tdir)` and thread `model` through the pipeline. This change is isolated to this file plus the two call sites above. The code you’re replacing is the Haiku‑flatten/unflatten logic here.

  > If you prefer to keep NPZ: you can also `jax.tree_util.tree_flatten` the Equinox parameters and save leaves as `p000`, `p001`, ... with a `tree_def.pkl`. But Equinox’s `tree_serialise_leaves` is simpler to maintain.

### 5) Clean up the VI registry and FlowJAX knobs

* In **`lambda_hat/vi/registry.py`**, delete the lazy import and just import both VI algorithms normally. Keep a single `try/except ImportError` around the *flow* import that raises a clear error (“Install FlowJAX to use algo=flow”). That’s all you need; the current lazy loader exists only to accommodate optional FlowJAX.
* Keep your global PRNG setting to threefry in `lambda_hat/__init__.py`—that’s still a good idea for consistent behavior with FlowJAX and Parsl.

### 6) Remove Haiku and fix imports

* Delete `lambda_hat/models.py` (or keep only helpers like `infer_widths`) and switch its imports to `nn_eqx`.
* Remove any `haiku as hk` references (e.g., the version probe in the build entrypoint, if any).
* Update docs where you list Haiku as the NN framework to Equinox. (Your README mentions Haiku explicitly; just swap the sentence.)

### 7) Update the sampling entrypoint to use the loaded Equinox model

* In **`lambda_hat/entrypoints/sample.py`**, after loading the artifact, use the `model` object and `predict = lambda m, X: m(X)` for `make_loss_fns`. This replaces the Haiku path that rebuilds a Haiku model and calls `model.apply`. See the current Haiku usage here.

### 8) (Optional) Deduplicate the MLP in `llc/`

Your `llc` subpackage defines a separate, pure‑JAX MLP with optional residuals/layernorm. You can leave it as‑is (it’s not adding a dependency). If you want *one* source of truth, change `llc.models` to call your Equinox MLP + `ravel_pytree` for training. That’s a nice‑to‑have, not required.

---

## What to delete after the migration

* **Haiku dependency & code paths**

  * `lambda_hat/models.py` (the Haiku class and `hk.transform` factory).
  * Haiku‑specific flatten/unflatten in `target_artifacts.py`.
  * Any doc lines that say Haiku is the NN framework.
* **Dynamic import plumbing**

  * The lazy FlowJAX import in `vi/registry.py` (keep one clean error instead).

---

## Stopgap (if you want the *absolute* least work today)

If you truly don’t want to touch model code right now:

1. **Make Equinox mandatory** in your environment and **keep Haiku**. This removes the optional/extras friction and all the try/except/lazy import complexity.
2. In `vi/registry.py`, delete the lazy import and just import FlowJAX; if FlowJAX isn’t installed, error out with a clear message (“install with `--extra flowvi`”) and exit.

This leaves two NN frameworks (inelegant), but it removes the *dynamic* dependency handling, which seems to be your current pain point. It’s still more maintenance long‑term than the Equinox‑only path.

---

## Gotchas to watch for

* **Initializers & norms:** Haiku’s `VarianceScaling` defaults differ slightly from Equinox’s `Linear`; if your training is sensitive, set explicit init schemes (or let Optax wash it out). You already record `L0` in metadata and expect to recompute it; the training loop is robust.
* **Param counting:** Swap to `eqx.partition(..., eqx.is_array)` then sum leaves; it replaces your Haiku count.
* **Artifacts:** Because you said you don’t care about backward compatibility, simply bump the artifact version in `meta.json` and fail fast when a legacy NPZ is loaded (“rebuild the target”); your loader already throws on unexpected formats.

---

## Checklist (one pass)

* [ ] Add `lambda_hat/nn_eqx.py` with `EqxMLP`, `build_mlp`, `count_params`.
* [ ] In `lambda_hat/targets.py`, build Equinox model with a key; replace `model.apply` with `predict(model, X)` in losses.
* [ ] In `lambda_hat/entrypoints/sample.py`, load Equinox model from artifact; use `predict` in losses.
* [ ] In `lambda_hat/losses.py`, change calls from `model_apply(params, None, X)` → `predict(params, X)`.
* [ ] In `lambda_hat/target_artifacts.py`, switch to Equinox `tree_serialise_leaves` for params/statics; update loader to return an Equinox model.
* [ ] In `lambda_hat/vi/registry.py`, remove lazy imports (or keep a single clean `ImportError` for FlowJAX).
* [ ] Remove Haiku from your environment; update docs to say “Equinox.”
* [ ] (Optional) Point `llc` at the same Equinox MLP to dedupe.

---

### Bottom line

Given your constraints (no back‑compat, one developer, maintenance is king), **migrating the small Haiku slice to Equinox** is the cleanest way to kill an entire class of ongoing friction: two NN stacks, two ways of serializing, and dynamic dependency tricks. The touch points are few and well‑isolated in your repo. After this change, your stack is “JAX + Optax + BlackJAX + Equinox (+ FlowJAX if you want VI)”, with no special‑case import gymnastics.
