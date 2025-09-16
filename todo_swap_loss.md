right now `test_quadratic_sanity.py` has to **monkey‑patch** `losses.make_loss_fns`, which is brittle and forces your “diagnostic” quadratic to impersonate the NN loss. The clean fix is to make the *target model* a first‑class, **swappable** component that provides `(loss_full, loss_minibatch)` and an initial center `θ₀`. Then the pipeline can keep doing “generous sampling analysis” (SGLD/HMC/MCLMC, ESS, diagnostics, artifacts) unchanged, while you swap in an NN, a quadratic, a deep‑linear model, etc.

Below is a careful, minimal refactor you can hand to a simplistic AI coding assistant. It keeps signatures stable where they matter (notably `grad_logpost_minibatch` still takes a minibatch shaped like `(Xb, Yb)` because `posterior.make_logpost_and_score` expects it), so we don’t churn your samplers or diagnostics.&#x20;

---

## What we are changing (and what we are not)

**Keep as‑is:**

* The samplers & their knobs, evaluation cadence, and LLC estimator (all the way through `make_logpost_and_score → runners → diagnostics`).&#x20;
* The SGLD/HMC/MCLMC wiring and the expectation that `loss_minibatch(theta, Xb, Yb)` exists (so `posterior.make_logpost_and_score` can form the local posterior without any code change).&#x20;

**Change:**

* Introduce a **Target** layer that builds the loss fns and `θ₀` for one of: `"mlp"` (current default) or `"quadratic"`.
* Let the pipeline ask the Target for: `d`, `θ₀(f32,f64)`, `(loss_full, loss_minibatch)` (both dtypes), and *placeholder data* `(X, Y)` if the target doesn’t need data (so your SGLD minibatch plumbing stays untouched).

Consequence: you can run *exactly the same* samplers and diagnostics while swapping the target with one config flag.

---

## Step‑by‑step refactor (small patches)

> Order: Config → new `targets.py` → pipeline → CLI (optional) → test.

### 1) Add a target selector to the config

**File:** `llc/config.py` — add two fields (default keeps current behavior).

```diff
@@
 from dataclasses import dataclass
-from typing import Optional, Literal, List
+from typing import Optional, Literal, List
@@
 @dataclass
 class Config:
+    # ---- Target model selection ----
+    # "mlp": current neural network target (default)
+    # "quadratic": analytical diagnostic L_n(θ) = 0.5 ||θ||^2 (ignores data)
+    target: Literal["mlp", "quadratic"] = "mlp"
+    # For 'quadratic', you can set the parameter dimension here; if None we fall
+    # back to target_params (else in_dim).
+    quad_dim: Optional[int] = None
```

This is a no‑op for existing runs (`target="mlp"`).&#x20;

---

### 2) Add a tiny target factory

**New file:** `llc/targets.py`

```python
# llc/targets.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .config import Config
from .data import make_dataset
from .models import infer_widths, init_mlp_params
from .losses import make_loss_fns, as_dtype
from .experiments import train_erm

@dataclass
class TargetBundle:
    d: int
    theta0_f32: jnp.ndarray
    theta0_f64: jnp.ndarray
    # loss(theta) -> scalar
    loss_full_f32: Callable[[jnp.ndarray], jnp.ndarray]
    loss_minibatch_f32: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    loss_full_f64: Callable[[jnp.ndarray], jnp.ndarray]
    loss_minibatch_f64: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    # data for minibatching (even if ignored by the target)
    X_f32: jnp.ndarray
    Y_f32: jnp.ndarray
    X_f64: jnp.ndarray
    Y_f64: jnp.ndarray
    L0: float  # L_n at theta0 (f64)

def _identity_unravel(theta: jnp.ndarray):
    # For analytic targets with a flat parameter vector
    return theta

def build_target(key, cfg: Config) -> TargetBundle:
    """Return a self-contained target for the pipeline to consume."""
    if (cfg.target or "mlp") == "mlp":
        # ----- Existing NN path -----
        # Data + teacher
        X, Y, teacher_params, teacher_forward = make_dataset(key, cfg)

        # Init student params & train to ERM (θ⋆), then center prior at θ⋆
        key, subkey = jax.random.split(key)
        widths = cfg.widths or infer_widths(
            cfg.in_dim, cfg.out_dim, cfg.depth, cfg.target_params, fallback_width=cfg.hidden
        )
        w0_pytree = init_mlp_params(
            subkey, cfg.in_dim, widths, cfg.out_dim, cfg.activation, cfg.bias, cfg.init
        )
        theta_star_f64, unravel_star_f64 = train_erm(
            w0_pytree, cfg, X.astype(jnp.float64), Y.astype(jnp.float64)
        )
        params_star_f64 = unravel_star_f64(theta_star_f64)
        params_star_f32 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float32), params_star_f64)
        theta0_f32, unravel_star_f32 = ravel_pytree(params_star_f32)

        # Cast data to both dtypes
        X_f32, Y_f32 = as_dtype(X, cfg.sgld_dtype), as_dtype(Y, cfg.sgld_dtype)
        X_f64, Y_f64 = as_dtype(X, cfg.hmc_dtype), as_dtype(Y, cfg.hmc_dtype)

        # Loss fns for each dtype
        loss_full_f32, loss_minibatch_f32 = make_loss_fns(unravel_star_f32, cfg, X_f32, Y_f32)
        loss_full_f64, loss_minibatch_f64 = make_loss_fns(unravel_star_f64, cfg, X_f64, Y_f64)

        L0 = float(loss_full_f64(theta_star_f64))
        d = int(theta0_f32.size)
        return TargetBundle(
            d=d,
            theta0_f32=theta0_f32,
            theta0_f64=theta_star_f64,
            loss_full_f32=loss_full_f32,
            loss_minibatch_f32=loss_minibatch_f32,
            loss_full_f64=loss_full_f64,
            loss_minibatch_f64=loss_minibatch_f64,
            X_f32=X_f32, Y_f32=Y_f32, X_f64=X_f64, Y_f64=Y_f64,
            L0=L0,
        )

    elif cfg.target == "quadratic":
        # ----- Analytic diagnostic: L_n(θ) = 0.5 ||θ||^2 -----
        d = int(cfg.quad_dim or cfg.target_params or cfg.in_dim)
        theta0_f64 = jnp.zeros((d,), dtype=jnp.float64)
        theta0_f32 = theta0_f64.astype(jnp.float32)

        # loss_full(θ) = 0.5 ||θ||^2 ; minibatch ignores Xb,Yb but keeps signature
        def _lf(theta):      return 0.5 * jnp.sum(theta * theta)
        def _lb(theta, Xb, Yb):  # <— keep (theta, Xb, Yb) to match posterior.py
            return _lf(theta)

        # Provide trivial data so SGLD minibatching works without special cases
        n = int(cfg.n_data)
        X_f32 = jnp.zeros((n, 1), dtype=jnp.float32)
        Y_f32 = jnp.zeros((n, 1), dtype=jnp.float32)
        X_f64 = jnp.zeros((n, 1), dtype=jnp.float64)
        Y_f64 = jnp.zeros((n, 1), dtype=jnp.float64)

        L0 = 0.0  # L_n at θ0=0
        return TargetBundle(
            d=d,
            theta0_f32=theta0_f32,
            theta0_f64=theta0_f64,
            loss_full_f32=lambda th: _lf(th.astype(jnp.float32)).astype(jnp.float32),
            loss_minibatch_f32=lambda th, Xb, Yb: _lb(th.astype(jnp.float32), Xb, Yb).astype(jnp.float32),
            loss_full_f64=lambda th: _lf(th.astype(jnp.float64)).astype(jnp.float64),
            loss_minibatch_f64=lambda th, Xb, Yb: _lb(th.astype(jnp.float64), Xb, Yb).astype(jnp.float64),
            X_f32=X_f32, Y_f32=Y_f32, X_f64=X_f64, Y_f64=Y_f64,
            L0=L0,
        )
    else:
        raise ValueError(f"Unknown target: {cfg.target}")
```

Notes:

* We **preserve the minibatch signature** `(theta, Xb, Yb)` so `posterior.make_logpost_and_score` stays untouched.&#x20;
* For the quadratic target we feed placeholder `(X, Y)` arrays so the SGLD minibatch indexing still works, but they are simply ignored by the loss. This avoids touching any sampler plumbing.&#x20;

---

### 3) Use the target in the pipeline

**File:** `llc/pipeline.py` — replace the early “build data → init → train → loss” block with `build_target`, then proceed exactly as before.

```diff
@@
-from .models import infer_widths, init_mlp_params
-from .data import make_dataset
-from .losses import as_dtype, make_loss_fns
+from .targets import build_target
@@
-    logger.info("Building teacher and data")
-    stats = RunStats()
+    logger.info("Building target")
+    stats = RunStats()
@@
-    X, Y, teacher_params, teacher_forward = make_dataset(key, cfg)
-
-    # Initialize student network parameters
-    key, subkey = random.split(key)
-    widths = cfg.widths or infer_widths(
-        cfg.in_dim, cfg.out_dim, cfg.depth, cfg.target_params, fallback_width=cfg.hidden
-    )
-    w0_pytree = init_mlp_params(
-        subkey, cfg.in_dim, widths, cfg.out_dim, cfg.activation, cfg.bias, cfg.init
-    )
-
-    stats.t_build = toc(t0)
-
-    # Train to empirical minimizer (ERM) - center the local prior there
-    logger.info("Training to empirical minimizer...")
-    t1 = tic()
-    theta_star_f64, unravel_star_f64 = train_erm(
-        w0_pytree, cfg, X.astype(jnp.float64), Y.astype(jnp.float64)
-    )
-    stats.t_train = toc(t1)
-
-    # Create proper f32 unravel function (rebuild around f32 params)
-    params_star_f64 = unravel_star_f64(theta_star_f64)
-    params_star_f32 = jax.tree_util.tree_map(
-        lambda a: a.astype(jnp.float32), params_star_f64
-    )
-    theta_star_f32, unravel_star_f32 = ravel_pytree(params_star_f32)
-
-    # Center the local prior at θ⋆, not at the teacher
-    theta0_f64, unravel_f64 = theta_star_f64, unravel_star_f64
-    theta0_f32, unravel_f32 = theta_star_f32, unravel_star_f32
-
-    # Create dtype-specific data versions
-    X_f32, Y_f32 = as_dtype(X, cfg.sgld_dtype), as_dtype(Y, cfg.sgld_dtype)
-    X_f64, Y_f64 = as_dtype(X, cfg.hmc_dtype), as_dtype(Y, cfg.hmc_dtype)
-
-    dim = theta0_f32.size
+    # Build a self-contained target (NN, quadratic, …)
+    bundle = build_target(key, cfg)
+    stats.t_build = toc(t0)
+
+    theta0_f32 = bundle.theta0_f32
+    theta0_f64 = bundle.theta0_f64
+    X_f32, Y_f32, X_f64, Y_f64 = bundle.X_f32, bundle.Y_f32, bundle.X_f64, bundle.Y_f64
+    dim = bundle.d
     print(f"Parameter dimension: {dim:,d}")
@@
-    # Create loss functions for each dtype
-    loss_full_f32, loss_minibatch_f32 = make_loss_fns(unravel_f32, cfg, X_f32, Y_f32)
-    loss_full_f64, loss_minibatch_f64 = make_loss_fns(unravel_f64, cfg, X_f64, Y_f64)
+    # Loss functions supplied by the target
+    loss_full_f32, loss_minibatch_f32 = bundle.loss_full_f32, bundle.loss_minibatch_f32
+    loss_full_f64, loss_minibatch_f64 = bundle.loss_full_f64, bundle.loss_minibatch_f64
@@
-    # Recompute L0 at empirical minimizer (do this in float64 for both samplers)
-    L0 = float(loss_full_f64(theta0_f64))
-    print(f"L0 at empirical minimizer: {L0:.6f}")
+    # L0 is provided by the target (at θ0)
+    L0 = float(bundle.L0)
+    print(f"L0 at reference θ0: {L0:.6f}")
```

Nothing else in the pipeline needs to change: `compute_beta_gamma`, `make_logpost_and_score`, samplers, and all diagnostics keep working because they still receive the same shapes and call signatures. (Critically, `make_logpost_and_score` still calls `loss_minibatch(theta, Xb, Yb)`, which our quadratic target implements as a no‑op on `(Xb, Yb)`.)&#x20;

---

### 4) (Optional) expose the target via CLI

If you use the CLI, add two flags.

**File:** `llc/cli.py`

```diff
@@ def add_run_arguments(parser: argparse.ArgumentParser) -> None:
     # Model architecture
     parser.add_argument("--depth", type=int, help="Number of hidden layers")
     parser.add_argument("--width", type=int, help="Hidden layer width")
     parser.add_argument("--target-params", type=int, help="Target parameter count")
+    # Target selection
+    parser.add_argument("--target", choices=["mlp", "quadratic"], help="Target model")
+    parser.add_argument("--quad-dim", type=int, help="Parameter dimension for quadratic target")
@@ def override_config(cfg: Config, args: argparse.Namespace) -> Config:
     for attr in direct_mappings:
         value = getattr(args, attr.replace("-", "_"), None)
         if value is not None:
             overrides[attr] = value
+    # Target selection overrides
+    if getattr(args, "target", None) is not None:
+        overrides["target"] = args.target
+    if getattr(args, "quad_dim", None) is not None:
+        overrides["quad_dim"] = args.quad_dim
```

The rest of the CLI stays intact.&#x20;

---

### 5) Delete the monkey‑patch and make the quadratic test use the swappable target

**File:** `test_quadratic_sanity.py` — simplify to a true black‑box test:

```diff
@@
-from llc.config import Config
-from llc.pipeline import run_one
+from llc.config import Config
+from llc.pipeline import run_one
@@
-def make_quadratic_config(d: int = 4, beta: float = 1.0) -> Config:
-    """Create config for pure quadratic test L_n(θ) = 0.5||θ||²"""
-    return Config(
-        # Model: just parameters (no actual network)
-        in_dim=d,
-        out_dim=1,
-        depth=0,  # No hidden layers
-        widths=[],
-        target_params=d,  # d parameters total
-
-        # Data: dummy (won't be used with our custom loss)
-        n_data=100,
-
-        # Posterior: tempering only, no spatial prior
-        beta_mode="fixed",
-        beta0=beta,
-        gamma=0.0,  # No spatial localization
-
-        # Quick sampling for test
-        chains=2,
-        sgld_steps=500,
-        sgld_warmup=100,
-        sgld_eval_every=10,
-        hmc_draws=200,
-        hmc_warmup=50,
-        hmc_eval_every=2,
-        mclmc_draws=300,
-        mclmc_eval_every=3,
-
-        # Use all samplers
-        samplers=["sgld", "hmc", "mclmc"],
-
-        # Save results
-        save_plots=False,
-        save_manifest=False,
-        save_readme_snippet=False,
-    )
+def make_quadratic_config(d: int = 4, beta: float = 1.0) -> Config:
+    """Config for pure quadratic test L_n(θ) = 0.5||θ||²"""
+    return Config(
+        target="quadratic",
+        quad_dim=d,
+        # Data count only affects the tempering scale nβ and SGLD batching
+        n_data=100,
+        beta_mode="fixed",
+        beta0=beta,
+        gamma=0.0,        # no spatial prior
+        # sampling budget (keep generous if you like)
+        chains=2,
+        sgld_steps=500, sgld_warmup=100, sgld_eval_every=10,
+        hmc_draws=200, hmc_warmup=50, hmc_eval_every=2,
+        mclmc_draws=300, mclmc_eval_every=3,
+        samplers=["sgld","hmc","mclmc"],
+        save_plots=False, save_manifest=False, save_readme_snippet=False,
+    )
@@
-    cfg = make_quadratic_config(d)
-
-    # We need to monkey-patch the loss function to use pure quadratic
-    # This is a bit of a hack but allows us to test without major refactoring
-    original_make_loss_fns = None
-
-    try:
-        from llc import losses
-        original_make_loss_fns = losses.make_loss_fns
-
-        def quadratic_loss_fns(unravel_fn, cfg, X, Y):
-            """Pure quadratic loss: L_n(θ) = 0.5||θ||²"""
-            def loss_full(theta_flat):
-                return 0.5 * jnp.sum(theta_flat ** 2)
-
-            def loss_batch(theta_flat, batch):
-                # For quadratic, batch doesn't matter
-                return loss_full(theta_flat)
-
-            return loss_full, loss_batch
-
-        # Monkey patch
-        losses.make_loss_fns = quadratic_loss_fns
-
-        # Run the test
-        result = run_one(cfg, save_artifacts=False, skip_if_exists=False)
+    cfg = make_quadratic_config(d)
+    # Run the test with the swappable target
+    result = run_one(cfg, save_artifacts=False, skip_if_exists=False)
@@
-    finally:
-        # Restore original function
-        if original_make_loss_fns is not None:
-            losses.make_loss_fns = original_make_loss_fns
+    # no monkey-patching anymore
```

The expected value **stays** `d/2` because with `γ=0` and `L_n(θ)=0.5‖θ‖²`, the tempered target is `N(0, I/(nβ))`, so $E[L_n]=\frac{d}{2nβ}$ and $\hat\lambda=nβ\,E[L_n]=d/2$. If any sampler reports ≈`d` you still have the classic “×2” bug. ✔️&#x20;

---

## Why this design is correct and minimal

* **No churn in SGMCMC plumbing.** We keep *exactly* the minibatch gradient API that `posterior.make_logpost_and_score` relies on (minibatch is passed as `(Xb, Yb)`), avoiding a cross‑cutting signature change. The quadratic target simply ignores the batch.&#x20;
* **All sampler knobs remain visible.** We don’t touch the config surface for SGLD/HMC/MCLMC or the diagnostic machinery; `run_one` still computes LLC, ESS‑aware SEs, saves NetCDF traces, etc.&#x20;
* **Targets are truly swappable.** The pipeline now depends on a self‑contained `TargetBundle`, so adding more targets (e.g., deep‑linear with analytic LLC, logistic regression, a synthetic quartic) is a 30‑line addition to `targets.py`—no changes to samplers or diagnostics.

---

## Usage examples

**Quadratic sanity test (no monkey patch):**

```bash
uv run python test_quadratic_sanity.py
# or via CLI
uv run python main.py run --target=quadratic --quad-dim=8 --beta-mode=fixed --beta0=1.0 --gamma=0
```

**Back to NN target (default):**

```bash
uv run python main.py run --preset=quick
```

---

## A few “gotchas” I’m pre‑empting

* **Don’t change the loss scale.** Your samplers assume `L_n` is an *average* (not a sum). The quadratic diagnostic is defined as `0.5||θ||²`—no extra 1/2σ², no sum over data—so that the factor‑of‑2 detection is meaningful. If you change the scale, $\hat\lambda$ shifts.&#x20;
* **Set `gamma=0` when you want the closed‑form Gaussian check.** Otherwise the local prior contributes a quadratic term and shifts the expected LLC away from `d/2`. The test config above does this explicitly.&#x20;
* **We *intentionally* feed dummy `(X, Y)`.** SGLD samples minibatches; supplying zero arrays avoids branching in every sampler while keeping shapes stable. The quadratic loss simply ignores them.&#x20;

---

## Optional niceties you might want (later)

1. **Target registry:** instead of `if cfg.target == ...` in `build_target`, keep a dict `{name: builder}` and a `register_target(name, fn)` helper. That lets experiments add ad‑hoc diagnostics without touching the core file.

2. **Expose `θ₀` choice in config:** for MLP you currently center at the ERM `θ⋆`. If you want to replicate the *teacher‑centered* estimator from the benchmark paper *or* compare both, add
   `prior_center={"erm"|"teacher"|"zero"}` plus `reference_for_L0` (you already have those fields—use them). Then set `θ₀` and `L0` consistently per choice.&#x20;

---

If you apply the patches above, the quadratic diagnostic becomes a first‑class target, your test stops monkey‑patching, and you can trivially add more “diagnostic targets” while keeping your generous sampling analysis identical.
