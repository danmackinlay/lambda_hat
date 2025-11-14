Great direction. Here’s a concrete, minimally–invasive refactor plan that lets you keep the current VI (“mixture of factor analysers”, MFA) working while making it trivial to plug in a new VI “flavour” (e.g., a normalizing-flow VI) later.

---

## 0) Where the current hooks are (so we refactor the right places)

* **Runner entry:** the VI path is selected alongside SGLD/HMC/MCLMC, and the runner calls `run_vi(..., config=cfg.sampler.vi, ...)`. This is the natural dispatch point for multiple VI flavours.
* **Current VI implementation & API:** the main entry is `fit_vi_and_estimate_lambda(...)` (returns `lambda_hat, traces, extras`) in the variational module. Tests import `from lambda_hat import variational as vi`, and docs also say “VI implementation is in `lambda_hat/variational.py`”, so we need a backward‑compat shim.
* **Current VI config surface:** `VIConfig` contains `M, r, steps, batch_size, lr, eval_every, gamma, eval_samples, dtype, use_whitening`, and (in your latest iteration) extended knobs like `whitening_mode`, `clip_global_norm`, `alpha_temperature`, LR schedules, etc. We’ll add an `algo`/`flavour` selector without breaking existing fields. The shipped YAML preset for VI lives at `lambda_hat/conf/sample/sampler/vi.yaml`.
* **Target bundle & losses:** VI needs the ERM center and an **unravel** to convert between flat vectors and PyTrees. Your current target bundle passed around for Stage‑B (sampling) has losses and data, but not `unravel_fn` nor `w⋆` explicitly; VI’s helper expects both. We’ll add them to the bundle so all VI flavours can share them.

---

## 1) Module layout: introduce a pluggable VI package

Create a new package:

```
lambda_hat/vi/
  base.py        # common interfaces / result types
  registry.py    # string → algorithm factory
  mfa.py         # the existing “mixture of factor analysers” impl (moved here)
  flow.py        # (stub) normalizing-flow VI to be implemented later
  __init__.py    # re-exports; optional deprecation notices
```

**Keep a backwards‑compat shim**:

* Replace `lambda_hat/variational.py` with a thin shim that re‑exports `lambda_hat.vi.mfa` and emits a `DeprecationWarning`. This keeps `tests/test_vi_mlp.py` working while you migrate imports at your pace.

**`base.py` (interface)**

```python
# lambda_hat/vi/base.py
from __future__ import annotations
from typing import Protocol, Any, Dict, NamedTuple

class VIRunResult(NamedTuple):
    traces: Dict[str, Any]   # e.g., {"elbo": ..., "radius2": ..., ...}
    timings: Dict[str, float]
    work: Dict[str, float]   # n_full_loss, n_minibatch_grads, sampler_flavour="iid"

class VIAlgorithm(Protocol):
    name: str
    def run(
        self,
        rng_key,
        loss_batch_fn,      # (w_pytree, Xb, Yb) -> scalar
        loss_full_fn,       # (w_pytree) -> scalar
        wstar_flat,         # (d,)
        unravel_fn,         # flat -> pytree
        data,               # (X, Y)
        n_data: int,
        beta: float,
        gamma: float,
        vi_cfg,             # the VIConfig (common + algo-specific)
    ) -> VIRunResult: ...
```

**`registry.py`**

```python
# lambda_hat/vi/registry.py
from typing import Dict, Callable
from .base import VIAlgorithm
from .mfa import make_mfa_algo
# from .flow import make_flow_algo   # later

_REGISTRY: Dict[str, Callable[[], VIAlgorithm]] = {
    "mfa": make_mfa_algo,
    # "flow": make_flow_algo,        # later
}

def get(name: str) -> VIAlgorithm:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown VI algorithm '{name}'. Available: {sorted(_REGISTRY)}")
```

**`mfa.py`** (move the current code here, minimally adapted)

* Lift your existing `fit_vi_and_estimate_lambda(...)` and its helpers into `mfa.py`. Wrap it into a small adapter class that conforms to `VIAlgorithm.run(...)` and converts its `(lambda_hat, traces, extras)` into the unified `VIRunResult`.
* Keep the enriched parameters you added recently (e.g., `clip_global_norm`, `alpha_temperature`, per‑component ranks, LR schedule) — they already exist in your signature and tests.

---

## 2) Config schema: add a flavour selector with backward compatibility

**Dataclass.** Extend `VIConfig` with a new field (defaulting to the current algorithm) so existing configs keep working:

```python
# lambda_hat/config.py
@dataclass
class VIConfig:
    algo: str = "mfa"      # NEW: flavour selector ("mfa", "flow", ...)
    # (existing common fields:)
    M: int = 8
    r: int = 2
    steps: int = 5_000
    batch_size: int = 256
    lr: float = 0.01
    eval_every: int = 50
    gamma: float = 0.001
    eval_samples: int = 64
    dtype: str = "float32"
    # ... your extended knobs already present (whitening_mode, clip, schedules, etc.)
```

Those base fields (`M, r, ...`) remain meaningful for MFA and are simply ignored by other flavours that don’t use them. You already carry extended knobs like `whitening_mode`, `clip_global_norm`, `alpha_temperature`, and optional schedules; keep them as **flavour‑agnostic** where possible.

**YAML preset.** Make the flavour explicit, defaulting to MFA:

```yaml
# lambda_hat/conf/sample/sampler/vi.yaml
# @package _global_.sampler
name: vi
vi:
  algo: mfa            # NEW: choose the VI flavour
  M: 8
  r: 2
  steps: 5000
  batch_size: 256
  lr: 0.01
  eval_every: 50
  gamma: 0.001
  eval_samples: 64
  dtype: float32
  whitening_mode: none
```

This sits alongside the existing file; we’re just adding `algo:` and retaining your current knobs.

> **Optional** (future): if you want strongly typed, nested per‑flavour configs later, add optional sub‑blocks (e.g., `vi.flow.{layers,hidden,arch}`) and teach the algorithm to read its own section; keep the flat fields for backward compatibility during the migration.

---

## 3) Runner dispatch: a single `run_vi` that delegates by flavour

Refactor the **single** call‑site you already have to dispatch through a registry:

```python
# lambda_hat/sampling.py (or wherever run_vi currently lives)
from lambda_hat.vi.registry import get as get_vi_algo

def run_vi(key, loss_mini, loss_full, params0_flat, data, config, num_chains, beta, gamma):
    # 1) Retrieve extras VI needs:
    #    - unravel_fn and w⋆ (flat) must be available here.
    #    If they are not yet present in your target bundle, add them (see §4).
    unravel_fn = ...  # from target bundle
    wstar_flat = params0_flat

    # 2) Dispatch by flavour
    algo_name = getattr(config, "algo", "mfa")
    algo = get_vi_algo(algo_name)
    result = algo.run(
        rng_key=key,
        loss_batch_fn=loss_mini,    # expects pytree params
        loss_full_fn=loss_full,     # expects pytree params
        wstar_flat=wstar_flat,
        unravel_fn=unravel_fn,
        data=data,
        n_data=config.n_data if hasattr(config, "n_data") else ???,
        beta=beta,
        gamma=gamma,
        vi_cfg=config,
    )

    return result  # VIRunResult(traces, timings, work)
```

Because you already pass `config=cfg.sampler.vi` into `run_vi(...)`, this is the only place that needs to “know” which flavour is selected.

---

## 4) Make the target bundle VI‑ready (one small addition)

All VI flavours (including MFA and a future flow) need:

* `w⋆` (flattened) to center the local posterior,
* `unravel_fn` (flat → PyTree), because your VI losses operate on **PyTrees**. Your `fit_vi_and_estimate_lambda` and tests assume this.

**Add two fields** to the Stage‑B bundle (or keep them in a sibling structure you pass along):

```python
# lambda_hat/targets.py
@dataclass
class TargetBundle:
    ...
    wstar_flat: jnp.ndarray         # NEW: ERM solution (flattened)
    unravel_fn: Callable[[jnp.ndarray], Any]  # NEW: flat -> pytree
```

Populate them when you build the target (when you already train or compute the ERM and flatten parameters). This avoids “re‑flattening” gymnastics inside VI.

---

## 5) MFA adapter: turn the existing code into a VIAlgorithm

Inside `lambda_hat/vi/mfa.py`:

```python
from .base import VIAlgorithm, VIRunResult
# import your existing helpers and fit_vi_and_estimate_lambda(...)

class _MFAAlgo:
    name = "mfa"
    def run(self, rng_key, loss_batch_fn, loss_full_fn, wstar_flat, unravel_fn,
            data, n_data, beta, gamma, vi_cfg):
        # Call the existing routine (already supports advanced knobs)
        lam, traces, extras = fit_vi_and_estimate_lambda(
            rng_key=rng_key,
            loss_batch_fn=loss_batch_fn,
            loss_full_fn=loss_full_fn,
            wstar_flat=wstar_flat,
            unravel_fn=unravel_fn,
            data=data,
            n_data=n_data,
            beta=beta,
            gamma=gamma,
            M=vi_cfg.M,
            r=vi_cfg.r,
            steps=vi_cfg.steps,
            batch_size=vi_cfg.batch_size,
            lr=vi_cfg.lr,
            eval_samples=vi_cfg.eval_samples,
            whitener=make_whitener_from_cfg(vi_cfg),  # your current whitening
            clip_global_norm=getattr(vi_cfg, "clip_global_norm", None),
            alpha_temperature=getattr(vi_cfg, "alpha_temperature", 1.0),
            # etc: schedule, masks, priors, entropy bonus...
        )
        # Normalize outputs for the runner
        timings = {"adaptation": 0.0, "sampling": 0.0, "total": 0.0}  # (fill if you measure)
        work = {
            "sampler_flavour": "iid",  # VI produces IID draws at the end
            # Count full-data evaluations and minibatch grads consistently with MCMC metrics
            "n_full_loss": int(traces.get("n_full_loss", 0)),
            "n_minibatch_grads": int(traces.get("cumulative_fge", 0.0) * n_data),
        }
        return VIRunResult(traces=traces, timings=timings, work=work)

def make_mfa_algo() -> VIAlgorithm:
    return _MFAAlgo()
```

The pieces above are just rearranging what you already have in `fit_vi_and_estimate_lambda(...)` (including the extended knobs like clipping and LR schedules you recently added).

> **Note:** your MFA entry function already logs and returns rich traces: `{"elbo", "logq", "radius2", "cumulative_fge", ...}`. Keep those names; the flow flavour can emit the same keys so your analysis stays uniform.

---

## 6) Flow stub: define the hooks now, fill in later

Add a `lambda_hat/vi/flow.py` **stub** that satisfies the same interface and raises a helpful error until implemented:

```python
from .base import VIAlgorithm, VIRunResult

class _FlowAlgo:
    name = "flow"
    def run(...):
        raise NotImplementedError(
            "Normalizing-flow VI is not implemented yet. "
            "Plumbed through config as vi.algo=flow."
        )

def make_flow_algo() -> VIAlgorithm:
    return _FlowAlgo()
```

This lets you toggle `vi.algo: flow` in configs and confirm the registry/dispatch path without any algorithmic code yet.

---

## 7) TensorBoard & diagnostics: one small interface for all flavours

Since you’ve started logging VI traces out of the runner, keep the **logging concern** out of algorithms:

* Add an optional `on_trace(step_idx: int, scalars: Dict[str, float])` callback to `VIAlgorithm.run(...)` or to the `vi_cfg` (e.g., `vi_cfg.tensorboard=True` already exists in the expanded config). The MFA implementation can call it inside the optimization scan body, and future flow code can do the same.
* Your test and docs already enumerate the typical scalars: ELBO, `radius2`, `logq`, plus any `work_fge` counters you record. Keep these names stable so dashboards don’t change between flavours.

This avoids tying TensorBoard to a specific flavour and keeps JIT‑ability—callbacks can be outside JIT and fed by host callbacks only if enabled.

---

## 8) Minimal code edits summary

* **New files:** `lambda_hat/vi/{base.py,registry.py,mfa.py,flow.py,__init__.py}`.
* **Shim:** replace `lambda_hat/variational.py` with re‑exports from `lambda_hat.vi.mfa` (backward compat for tests and docs).
* **Config:** add `algo: "mfa"` to `VIConfig` (dataclass) and to `lambda_hat/conf/sample/sampler/vi.yaml`.
* **Runner:** change `run_vi(...)` to fetch `algo = registry.get(cfg.sampler.vi.algo)` and call `algo.run(...)`.
* **Target bundle:** add `wstar_flat` and `unravel_fn` so all VI flavours can share them.

---

## 9) Documentation & examples

Update `docs/vi.md` to reflect the new “flavour” selector and keep MFA as default:

```yaml
samplers:
  - name: vi
    overrides:
      algo: mfa          # or: flow
      # MFA knobs (kept for compatibility)
      M: 8
      r: 2
      steps: 5000
      batch_size: 256
      lr: 0.01
      eval_every: 50
      eval_samples: 64
```

Also add a small section “Adding a new VI flavour” explaining the `VIAlgorithm` interface and the registry pattern above (copy the 10–12 line example). Your current docs explicitly point to `lambda_hat/variational.py`; replace that with `lambda_hat/vi/mfa.py` and mention the shim to ease migration.

---

## 10) Tests: keep the old, add two tiny ones

* Keep your existing MFA tests as‑is (they’ll pass through the shim).
* Add two small tests:

  1. **Dispatch:** `cfg.sampler.vi.algo="mfa"` produces identical traces to the direct call (sanity).
  2. **Flow not implemented:** `cfg.sampler.vi.algo="flow"` raises `NotImplementedError` from `flow.py` (proves plumbing).

---

## 11) Why this shape works for normalizing flows later

* **Loss surface & data handling** are already abstracted: `loss_batch_fn(w_pytree, Xb, Yb)` and `loss_full_fn(w_pytree)`; both a flow and MFA consume the same interface.
* **Centering at `w⋆`** and tempering `(β, γ)` still come from the same `PosteriorConfig`/runner logic; you won’t touch the Stage‑A build or Stage‑B orchestration to introduce a flow.
* **Diagnostics & work accounting**: by standardizing `traces`, `timings`, and `work`, your downstream analysis doesn’t have to special‑case algorithms. MCMC efficiency metrics already consume `work` and timings in a uniform way; VI can report `sampler_flavour="iid"` and consistent gradient counts for apples‑to‑apples comparisons.

---

## 12) Migration checklist (copy/paste)

1. **Create package** `lambda_hat/vi/...` and move MFA there.
2. **Add** `algo: "mfa"` to `VIConfig` and YAML preset.
3. **Add** `wstar_flat` and `unravel_fn` to the Stage‑B target bundle.
4. **Refactor** `run_vi(...)` to dispatch through `vi.registry.get(cfg.sampler.vi.algo)`.
5. **Shim** `lambda_hat/variational.py` → re‑export from `lambda_hat.vi.mfa`.
6. **Docs**: update paths and add the “flavours” section.
7. **Tests**: keep existing MFA tests; add dispatch + “flow not implemented yet” tests.

That’s it—after these edits, adding a normalizing‑flow VI is just “implement `flow.py` and register it”. The rest of the system (configs, CLI/runner, diagnostics) will already know how to use it.
