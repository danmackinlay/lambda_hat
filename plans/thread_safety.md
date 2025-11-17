There are **two independent root causes** behind the intermittent failures you’re seeing:

1. **Matplotlib/ArviZ mathtext isn’t thread–safe** under your Parsl **ThreadPool** execution. Multiple threads are creating figures at the same time → sporadic `ParseException` from mathtext (those `$\\mathdefault{10^{-1}}$` labels) during the *analysis/plotting* phase after the samplers completed. The code currently uses the Agg backend but does **not** serialize plotting or explicitly disable mathtext/TeX, so the threads contend on global matplotlib state. This exactly matches your HMC/SGLD failures. The trace/diagnostic plotting happens in `lambda_hat/analysis.py::create_arviz_diagnostics` and `create_combined_convergence_plot` without any locking or rcParam overrides【turn9file14†lambda_hat/analysis.py†L1-L65】【turn9file14†lambda_hat/analysis.py†L69-L135】. And you do run samplers/analysis in a Parsl **ThreadPool**【turn11file16†config/parsl/local.yaml†L1-L6】.

2. **Precision (x64) is toggled globally at runtime** in a threaded program. You switch `jax_enable_x64` off to deserialize Equinox models and then turn it back on later, from inside the sampling entry point itself (`sample_cmd.sample_entry`). Those are **process‑global** flags. With a ThreadPool, different workers race over `jax.config.update(...)` and occasionally run JIT-compiled code with the wrong precision expectations, producing errors like *“Executable expected parameter 0 of size 4 but got size 8”*. You can see the toggle in `sample_cmd.py` just before target loading【turn11file5†lambda_hat/commands/sample_cmd.py†L1-L40】【turn11file1†lambda_hat/commands/sample_cmd.py†L1-L33】. You also toggle inside target deserialization (`target_artifacts._deserialize_model`) in a try/finally block【turn9file6†lambda_hat/target_artifacts.py†L1-L39】【turn9file6†lambda_hat/target_artifacts.py†L41-L108】. Both are unsafe in a ThreadPool.

Below are **targeted fixes** with concrete diffs you can apply. They make plotting deterministic under threads and remove all global JAX precision toggles (replacing them with local, dtype‑safe deserialization). I’m also adding a small belt‑and‑suspenders dtype guard for VI to line up with SGLD.

---

## A. Make plotting thread‑safe and latex‑free

**Why**: ArviZ and matplotlib mutate global state; mathtext + concurrent renders → random parse failures. Your analysis functions don’t protect plotting.
**Fix**: (1) Add a global plotting lock and wrap every plot creation/saving in it; (2) force `text.usetex=False`, keep mathtext in `regular` mode, and disable automatic mathtext for formatters so log ticks don’t try to render as math.

**Patch** (`lambda_hat/analysis.py`):

```diff
@@
-import matplotlib
-matplotlib.use("Agg")  # Non-GUI backend
-from matplotlib import pyplot as plt
+import matplotlib
+matplotlib.use("Agg")  # Non-GUI backend
+from matplotlib import pyplot as plt
+import threading
+import os
+
+# ---- Matplotlib safety for threaded analysis ----
+# 1) Disable TeX and tone down mathtext to avoid parsing edge cases
+matplotlib.rcParams.update({
+    "text.usetex": False,
+    "mathtext.default": "regular",
+    "axes.formatter.use_mathtext": False,
+})
+# 2) Serialize all plotting in this process (ThreadPool-safe)
+_PLOT_LOCK = threading.Lock()
@@
-def create_arviz_diagnostics(idata: InferenceData, outdir: Path) -> None:
+def create_arviz_diagnostics(idata: InferenceData, outdir: Path) -> None:
     outdir.mkdir(parents=True, exist_ok=True)
-    # Trace
-    fig = az.plot_trace(idata, compact=True)
-    plt.tight_layout()
-    fig.savefig(outdir / "trace.png", dpi=150)
-    plt.close(fig)
+    # Trace
+    with _PLOT_LOCK:
+        fig = az.plot_trace(idata, compact=True)
+        plt.tight_layout()
+        fig.savefig(outdir / "trace.png", dpi=150)
+        plt.close(fig)
@@
-    # Rank + Energy
-    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
-    az.plot_rank(idata, ax=axes[0])
-    az.plot_energy(idata, ax=axes[1])
-    plt.tight_layout()
-    fig.savefig(outdir / "rank_energy.png", dpi=150)
-    plt.close(fig)
+    # Rank + Energy
+    with _PLOT_LOCK:
+        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
+        az.plot_rank(idata, ax=axes[0])
+        az.plot_energy(idata, ax=axes[1])
+        plt.tight_layout()
+        fig.savefig(outdir / "rank_energy.png", dpi=150)
+        plt.close(fig)
@@
-def create_combined_convergence_plot(metrics: Dict[str, Any], out_png: Path) -> None:
-    fig, ax = plt.subplots(figsize=(6, 4))
-    # ... existing code ...
-    plt.tight_layout()
-    fig.savefig(out_png, dpi=150)
-    plt.close(fig)
+def create_combined_convergence_plot(metrics: Dict[str, Any], out_png: Path) -> None:
+    with _PLOT_LOCK:
+        fig, ax = plt.subplots(figsize=(6, 4))
+        # ... existing code ...
+        plt.tight_layout()
+        fig.savefig(out_png, dpi=150)
+        plt.close(fig)
```

Where to change: the plotting functions shown above are the ones your CLI calls during analysis【turn9file14†lambda_hat/analysis.py†L69-L135】.

If you still see rare issues on macOS after this, you can add *one* more guard in your Parsl config to reduce contention: set `max_threads: 4` in `config/parsl/local.yaml` (keep it low for tests)【turn11file16†config/parsl/local.yaml†L1-L6】. But with the lock in place, this should not be required.

---

## B. Remove global JAX precision toggles (race) and make Equinox deserialization dtype‑local

**Why**: You currently switch `jax_enable_x64` off and on during `sample_entry` and *again* during `_deserialize_model`. In a ThreadPool, those calls race and leak across tasks, causing intermittent *“expected param size 4 got 8”* errors. You saw exactly that in VI. The fix is to **never** flip the global flag inside worker code. Instead, supply a **float32 template** to Equinox and let the caller cast the loaded model to the sampler’s dtype (you’re already doing dtype casting after load).

### B1. Stop toggling in `sample_cmd.sample_entry`

**Patch** (`lambda_hat/commands/sample_cmd.py`):

```diff
@@ def sample_entry(config_yaml: str, target_id: str, experiment: Optional[str] = None) -> Dict:
-    # Temporarily disable x64 for model template creation and target loading
-    # (targets are always saved as float32, so template must be float32)
-    # Save current x64 state to restore later
-    current_x64 = jax.config.jax_enable_x64
-    jax.config.update("jax_enable_x64", False)
@@
-    # NOW enable x64 if requested (after deserialization succeeded)
-    jax.config.update("jax_enable_x64", bool(cfg.jax.enable_x64))
+    # No global precision toggles here. Precision is handled per-sampler via dtype casting.
```

Those lines are exactly where you switch the global flag【turn11file1†lambda_hat/commands/sample_cmd.py†L1-L33】. Removing them eliminates the race.

### B2. Make `_deserialize_model` dtype‐aware without changing globals

**Patch** (`lambda_hat/target_artifacts.py`):

```diff
 def _deserialize_model(model_template: Any, path: Path) -> Any:
@@
-    Note:
-        Target artifacts are always saved as float32. We temporarily disable
-        x64 mode during deserialization to ensure template leaves match the
-        on-disk dtype, preventing Equinox dtype validation errors.
+    Note:
+        Target artifacts are always saved as float32. We construct a float32
+        template for params (shapes only) so Equinox dtype checks pass, without
+        touching the global JAX precision flags.
@@
-    x64_prev = jax.config.read("jax_enable_x64")
-    try:
-        jax.config.update("jax_enable_x64", False)
-
-        # Split template into params and static
-        params, static = eqx.partition(model_template, eqx.is_array)
-
-        # Load parameters
-        with open(path / "params.eqx", "rb") as f:
-            params = eqx.tree_deserialise_leaves(f, params)
-
-        # Load static structure
-        with open(path / "static.eqx", "rb") as f:
-            static = eqx.tree_deserialise_leaves(f, static)
-
-        # Combine and return
-        return eqx.combine(params, static)
-    finally:
-        jax.config.update("jax_enable_x64", x64_prev)
+    # Split the template into params and static
+    params_t, static_t = eqx.partition(model_template, eqx.is_array)
+    # Build a float32 params template (same structure/shapes)
+    params_t_f32 = jax.tree.map(
+        lambda a: jnp.asarray(a, dtype=jnp.float32), params_t
+    )
+    # Deserialize leaves using the float32 template
+    with open(path / "params.eqx", "rb") as f:
+        params = eqx.tree_deserialise_leaves(f, params_t_f32)
+    with open(path / "static.eqx", "rb") as f:
+        static = eqx.tree_deserialise_leaves(f, static_t)
+    # Return combined model (float32); caller may cast to desired dtype later
+    return eqx.combine(params, static)
```

This keeps deserialization strictly local to dtype and does **not** touch global config. The rest of your pipeline already casts to the sampler’s dtype immediately after loading (via `ensure_dtype` and explicit casts in `sampling_runner.run_sampler`)【turn9file15†lambda_hat/sampling_runner.py†L1-L45】【turn9file15†lambda_hat/sampling_runner.py†L46-L121】.

> **Why this fixes VI**
> VI is configured for `float32` but the global `JAX_ENABLE_X64=1` makes default arrays 64‑bit. During parallel runs, some threads flipped x64 and others didn’t. By stopping the flips and keeping deserialization local to float32, all code that JITs for VI sees consistent `float32` buffers, which removes the *“size 4 vs 8”* mismatch.

---

## C. Add a small dtype guard at the start of VI to harden against stray 64‑bit arrays

You already aligned SGLD with a belt‑and‑suspenders cast; do the same for VI where the objective captures data via closure.

**Patch** (`lambda_hat/vi/mfa.py`):

```diff
 def fit_vi_and_estimate_lambda(
@@
-    # Ensure data are JAX arrays (needed for JIT-safe indexed sampling)
-    X, Y = jnp.asarray(data[0]), jnp.asarray(data[1])
-    data = (X, Y)
+    # Ensure data match the reference dtype (flat w*)
+    ref_dtype = wstar_flat.dtype
+    X = jnp.asarray(data[0], dtype=ref_dtype)
+    Y = jnp.asarray(data[1], dtype=ref_dtype)
+    data = (X, Y)
```

This aligns with how SGLD forcibly casts inputs and ensures the captured `X, Y` inside the jitted `step_fn` match the VI dtype, regardless of the ambient default.

(Your `build_elbo_step` already propagates `ref_dtype` into all scalars and gradients; this small addition keeps the closure consistent.)

---

## D. (Optional, if you still want a belt‑and‑suspenders at call sites)

If you prefer to make the template f32 *before* calling the loader, you can also cast the freshly built `model_template` to f32 in `sample_cmd.sample_entry` instead of (or in addition to) the change in `_deserialize_model`:

```python
# After build_mlp(...)
model_template = jax.tree.map(
    lambda a: jnp.asarray(a, dtype=jnp.float32) if isinstance(a, jnp.ndarray) else a,
    model_template,
)
```

But with the `_deserialize_model` change above, this isn’t strictly necessary.

---

## E. Verify: what should change in the smoke test

1. **HMC/SGLD LaTeX ParseException**: should disappear because plotting is now serialized and latex/mathtext are tamed. The functions you call for trace/rank/energy are exactly where we added the lock【turn9file14†lambda_hat/analysis.py†L69-L135】.

2. **VI size‑mismatch error**: should disappear because there are no more global x64 flips (the flips are in `sample_cmd.py` and `_deserialize_model`, which we removed/rewrote)【turn11file5†lambda_hat/commands/sample_cmd.py†L1-L40】【turn9file6†lambda_hat/target_artifacts.py†L41-L108】, and VI now casts its captured `X, Y` to the `w*` dtype.

3. **Intermittency across samplers**: goes away—both sources of nondeterminism (threaded plotting and global JAX precision) are eliminated.

---

## F. If you want an even simpler containment strategy

If you don’t want to add locks, another pragmatic option is to make Parsl use **processes** (e.g., multiprocessing) for Stage‑B workers on local runs. That gives every sampler its own interpreter, isolating both matplotlib and JAX globals. But it’s heavier and slower than the lock, and your current code clearly expects ThreadPool (see “Using Parsl mode: local (ThreadPool)” in the log). I’d keep the ThreadPool and just serialize plotting.

---

## G. Quick checklist you can paste into a commit message

* analysis: add global plot lock + rcParams (`text.usetex=False`, `mathtext.default=regular`, `axes.formatter.use_mathtext=False`) to make ArviZ plotting thread‑safe and latex‑free【turn9file14†lambda_hat/analysis.py†L1-L23】【turn9file14†lambda_hat/analysis.py†L69-L135】.
* sample_cmd: remove `jax.config.update("jax_enable_x64", ...)` toggles in ThreadPool path【turn11file1†lambda_hat/commands/sample_cmd.py†L1-L33】.
* target_artifacts: implement dtype‑local Equinox deserialization (float32 template) instead of flipping global x64【turn9file6†lambda_hat/target_artifacts.py†L41-L108】.
* vi/mfa: force `X, Y` to `w*` dtype at entry to avoid ambient default leaks.

---

## H. Minimal re‑run plan

```bash
# Clean environment
rm -rf artifacts runs .pytest_cache

# Re-run the smoke test exactly as before
JAX_ENABLE_X64=1 uv run pytest tests/test_smoke_workflow.py -v -s
```

Expected: all 4 samplers complete; **no** LaTeX ParseExceptions; **no** VI “size 4 vs 8” errors; the analysis step writes `trace.png`/`rank_energy.png` without warnings (besides the harmless tight_layout warning you already saw).
