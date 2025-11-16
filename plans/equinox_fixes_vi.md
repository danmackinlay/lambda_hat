Below is a **single, destructive migration plan** to finish the VI refactor so everything is **flat‑space only**, consistent with the other samplers, and cheaper to maintain. Follow the steps in order.

---

## A. Freeze the new public API (flat‑only)

1. **Adopt one canonical VI algorithm interface** (no model‑space anywhere):

   ```python
   # lambda_hat/vi/types.py
   from typing import Protocol, Tuple, Any
   import jax.numpy as jnp
   Array = jnp.ndarray
   Batch = tuple[Array, Array]

   class FlatObjective(Protocol):
       def loss(self, w_flat: Array, batch: Batch) -> Array: ...
       def grad(self, w_flat: Array, batch: Batch) -> Array: ...
       # Optional convenience (algos may call this to avoid double AD):
       def value_and_grad(self, w_flat: Array, batch: Batch) -> tuple[Array, Array]: ...

   # Every VI algo gets:
   def run(
       *, rng_key,  # typed PRNG key
       objective: FlatObjective,  # flat value/grad
       wstar_flat: Array,
       data: Batch,
       n_data: int,
       beta: float,
       gamma: float,
       vi_cfg: Any,
       whitener: Any,    # as today; identity or diag
   ) -> dict:  # returns {lambda_hat, traces, extras, timings, work}
   ```

   **Delete** from *all* VI algorithms: `loss_batch_fn`, `loss_full_fn`, `unravel_fn`.

2. **Change the sampler entrypoint signature** to demand flat value+grad explicitly:

   ```python
   def run_vi(
       key,
       posterior,
       data,
       config,
       num_chains,
       loss_minibatch_flat,      # NEW: scalar value in flat space
       grad_loss_minibatch,      # existing: flat gradient
       loss_full_flat,           # NEW: scalar full‑data loss in flat space (for Ln_wstar, Eq[L_n] checks)
       n_data=None, beta=None, gamma=None,
   ) -> SamplerRunResult:
       ...
   ```

   * **Remove** `loss_full_fn` (model‑space) and **remove** every use of `unravel_fn` in the sampler.

---

## B. Build the flat objective once, centrally

3. In `lambda_hat/samplers/vi.py`, build a tiny adapter the algorithms will consume:

   ```python
   import jax

   class _FlatObjective:
       def __init__(self, loss_b, grad_b):
           self._loss_b = loss_b
           self._grad_b = grad_b
           self._vag = jax.value_and_grad(loss_b)

       def loss(self, w, batch): return self._loss_b(w, batch)
       def grad(self, w, batch): return self._grad_b(w, batch)
       def value_and_grad(self, w, batch): return self._vag(w, batch)
   ```

4. **Construct `loss_minibatch_flat` and `loss_full_flat` outside VI** (once) using your `Posterior`:

   * Add helpers on `Posterior` (or a small `objective_builder.py`) that produce **flat** closures; do the **only** `flat -> model` unflatten there, not in VI.
   * Example (sketch):

     ```python
     def make_flat_losses(posterior):
         def loss_full_flat(w_flat, data):
             model = posterior.vm.to_model(w_flat)    # the only unflatten
             return posterior.loss_full(model, data)  # existing model-space full loss

         def loss_minibatch_flat(w_flat, batch):
             model = posterior.vm.to_model(w_flat)
             return posterior.loss_minibatch(model, batch)
         return loss_minibatch_flat, loss_full_flat
     ```
   * If you already have flat primitives elsewhere, just wire them through; the point is **centralize** the conversion, not spread it across algorithms.

---

## C. Rewrite each VI algorithm to flat

5. **Mechanical edits in every VI algo** (`lambda_hat/vi/mfa.py`, `lambda_hat/vi/flow.py`, …):

   * Function signature → the new flat interface in §A.1.
   * Replace **all**:

     * `loss_batch_fn(model, Xb, Yb)` → `objective.loss(w_flat, (Xb, Yb))`
     * `jax.grad(loss_batch_fn)(model, ...)` → `objective.grad(w_flat, (Xb, Yb))`
     * Any `value_and_grad` usage → `objective.value_and_grad(...)`
   * **Delete** all unravelling and model‑typed params. Every internal sample of `w` must be **flat**.
   * Keep KL/log‑q terms exactly as today (they’re in variational parameter space, unaffected).

6. **ELBO/energy computation**:

   * Wherever ELBO needs the data term, call `objective.loss` on **minibatches** and scale as you already do.
   * If you previously relied on autodiff of the model‑space loss, switch to:

     * **Prefer** `objective.value_and_grad` (fastest, single pass).
     * **Else** call `objective.loss` + `objective.grad` separately (slower but equivalent).

---

## D. Simplify the sampler now that algorithms are flat

7. In `run_vi`:

   * **Remove** `loss_batch_fn_wrapped` / `loss_full_fn_wrapped` / `unravel_fn`.
   * Build `_FlatObjective(loss_minibatch_flat, grad_loss_minibatch)` and pass it to `algo.run`.
   * Compute `Ln_wstar` once at the sampler level via `loss_full_flat(posterior.flat0, (X, Y))` and put it in `work`. Algorithms no longer need full‑loss callables.

8. Keep your whitening pre‑pass exactly as is (it already consumes the **flat** gradient), and continue to pass `whitener` to algorithms.

---

## E. Normalize returns & traces (one contract)

9. **Enforce a common return shape** for all algorithms (already mostly true):

   * Must return dict with **only** JAX arrays / Python scalars.
   * Required keys:

     * `lambda_hat: Array[(,), dtype=float]`
     * `traces: dict[str, Array[(num_steps,), ...]]` at least containing:

       * `"elbo"`, `"grad_norm"`, `"cumulative_fge"`
     * `extras`: may include `"Eq_Ln"`, `"cv_info"`, etc.
     * `timings`: `{adaptation: float, sampling: float}`
     * `work`: `{"sampler_flavour": "...", ...}`
   * In the sampler, keep your existing normalization that vmaps chains and expands CV metrics; it should work unchanged.

10. **Stop carrying VI‑only model‑space artifacts**:

* `traces["Ln"]` can remain `nan` to fail fast (as you already do) or **drop** it entirely from VI traces if your analysis code tolerates missing keys. Given “you can break things”, consider **removing** it.

---

## F. Rip out the old surface

11. **Delete** (don’t deprecate) all model‑space paths in VI:

* From algorithms: parameters `loss_batch_fn`, `loss_full_fn`, `unravel_fn`.
* From `lambda_hat/vi/__init__.py` (registry): update factories to the new signature only.
* From `lambda_hat/samplers/vi.py`: remove wrapper functions and any `NotImplementedError` related to model‑space.

12. **Repo‑wide replace** to catch stragglers:

* `rg -n 'loss_batch_fn|loss_full_fn|unravel_fn' lambda_hat/vi`
* `rg -n 'to_model\\(' lambda_hat/vi`
* Fix/kill every match inside the VI module.

---

## G. Tests and guards

13. **Unit tests (flat only)**:

* Property test: `objective.loss` + `objective.grad` agree with `jax.grad(objective.loss)` within tolerance on random `w_flat`/minibatches.
* Contract test: Every VI algo returns only arrays/scalars; forbid pytree leaves of unknown types (you already have `_all_leaves_are_arrays`, keep it).
* Smoke test: MFA + Flow each run for a few steps with identity whitener and batch size>1.

14. **Regression checks** (cheap, no golden data):

* `lambda_hat` end‑to‑end: confirm `lambda_hat_mean`, `lambda_hat_std` are finite; `elbo` decreases (or increases, depending on sign) for first ~50 steps.

---

## H. Documentation & config

15. **Docs**: Update “VI expects model‑space loss” to “VI is flat‑only”. Remove old examples. Add a short snippet showing how to create `loss_minibatch_flat` / `grad_loss_minibatch` from `Posterior`.

16. **Config**: If any VI config mentions model‑space, delete those fields. Keep `whitening_mode`, `whitening_decay`, `steps`, `batch_size`, `lr`, etc.

---

## I. (Optional) Micro‑polish to reduce maintenance even further

17. **Hide value/grad plumbing behind a helper**:

```python
def make_flat_objective(loss_minibatch_flat, grad_loss_minibatch):
    # returns the _FlatObjective instance; algorithms never see raw callables
    return _FlatObjective(loss_minibatch_flat, grad_loss_minibatch)
```

Now every sampler creates exactly one object and passes it around.

18. **Drop ELBO value tracing** inside algorithms if it meaningfully complicates code; keep only `grad_norm` and `lambda_hat`. Given your priorities, fewer traces = less maintenance.

---

## J. Success criteria (quick self‑check)

* No symbol named `loss_full_fn`, `loss_batch_fn`, or `unravel_fn` exists anywhere under `lambda_hat/vi/`.
* `run_vi` compiles without wrappers and never touches model objects.
* All VI algorithms run with the same flat objective; whitening uses only flat grads.
* Tests show only arrays/scalars in returns; no PjitFunction leaks; ELBO/grad_norm traces have expected shapes.

---
