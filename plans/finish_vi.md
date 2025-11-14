Below is a concrete, implementation‑ready plan to upgrade the **VI** path (mixture of low‑rank factor analyzers) across **configurability**, **diagnostics**, **stability**, and **docs/UX**. I’ve grounded the plan in the current repo’s behavior and structure (pointing to exact files/structures), and I’ve included API diffs, code‑level hooks, tests, and a playbook for tuning.

---

## 0) Where VI stands today (baseline)

* **Family**: mixture of factor analyzers centered at (w^*); shared diagonal (D), low‑rank factors (A_m), Woodbury identities; STL gradients for continuous parameters, RB gradient for mixture weights. Implemented in `lambda_hat/variational.py`; orchestrated by `run_vi` in `lambda_hat/sampling.py`.
* **Config**: `sampler.vi` exposes `M, r, steps, batch_size, lr, gamma, eval_every, eval_samples, dtype, use_whitening` via YAML & dataclass. (See `lambda_hat/conf/sample/sampler/vi.yaml`, `lambda_hat/config.py::VIConfig`.)
* **Diagnostics**: traces carry `elbo`, `elbo_like`, `logq`, `radius2`, `resp_entropy`, `cumulative_fge`; the final LLC is computed with an **HVP control variate** and then broadcast as the “llc” trace for ArviZ. Plots are created by `lambda_hat/analysis.py`. There is **no TensorBoard** yet.
* **Gap noted in docs**: “`use_whitening: true` — currently uses identity; future support for Adam/RMSProp‑based whitening.” (i.e., geometry whitening is still a TODO).

The plan below addresses: **(1)** richer hyper‑parameterization, **(2)** diagnosability (TensorBoard + richer traces), **(3)** “finish” items from the VI plan (notably geometry whitening) and stability tweaks, and **(4)** a user playbook for improving performance.

---

## 1) Make VI more configurable (expose the right knobs cleanly)

### 1.1 Config surface (backward‑compatible)

**Add fields** to `VIConfig` (`lambda_hat/config.py`) and propagate through `lambda_hat/conf/sample/sampler/vi.yaml` and `run_vi`:

```python
# lambda_hat/config.py
@dataclass
class VIConfig:
    M: int = 8
    r: int = 2
    steps: int = 5_000
    batch_size: int = 256
    lr: float = 1e-2
    eval_every: int = 50
    gamma: float = 1e-3
    eval_samples: int = 64
    dtype: str = "float32"
    use_whitening: bool = True

    # NEW — mixture & rank control
    r_per_component: Optional[List[int]] = None       # e.g. [4,2,1,...]
    mixture_cap: Optional[int] = None                 # upper bound for M (enable pruning)
    prune_threshold: float = 1e-3                     # drop components with π < threshold
    alpha_dirichlet_prior: Optional[float] = None     # >1.0 discourages collapse

    # NEW — optimizer & schedules
    clip_global_norm: Optional[float] = 5.0           # optax.clip_by_global_norm
    lr_schedule: Optional[str] = "cosine"             # "none"|"cosine"|"linear_decay"
    lr_warmup_frac: float = 0.05                      # warmup steps fraction

    # NEW — annealing / entropy
    entropy_bonus: float = 0.0                        # add λ * H(q) to ELBO target term
    alpha_temperature: float = 1.0                    # softmax temperature on α

    # NEW — whitening options
    whitening_mode: str = "none"                      # "none"|"rmsprop"|"adam"|"hvp_diag"
    whitening_decay: float = 0.99                     # EMA decay for diag moments
```

**YAML example** (`lambda_hat/conf/sample/sampler/vi.yaml`):

```yaml
vi:
  M: 16
  r: 2
  steps: 10000
  batch_size: 512
  lr: 0.005
  eval_every: 50
  gamma: 0.001
  eval_samples: 128
  dtype: float32
  use_whitening: true

  # NEW
  r_per_component: [4,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1]
  mixture_cap: 16
  prune_threshold: 1e-3
  alpha_dirichlet_prior: 1.1

  clip_global_norm: 5.0
  lr_schedule: cosine
  lr_warmup_frac: 0.05

  entropy_bonus: 0.0
  alpha_temperature: 1.0

  whitening_mode: rmsprop
  whitening_decay: 0.99
```

* **Per‑component rank budgets**: implement as padded tensors with an **activity mask**. Keep `A` as `(M, d, r_max)`; store `active_r[m]`; mask inactive columns when computing `K_m = D^{1/2} A_m`. Backward‑compatible if `r_per_component` is `None`. Update `init_vi_params`, `sample_q`, and `logpdf_components_and_resp` to honor masks.
* **Mixture budget/pruning**: after each `eval_every` steps, compute π; prune components with π < `prune_threshold` (renormalize π; keep arrays compact or leave holes and skip inactive components). Expose a simple **“static M with soft pruning”** first (no split/merge).
* **Simple prior on π**: add `alpha_dirichlet_prior` to the objective as ((\alpha_0-1)\sum_j \log \pi_j). Implement as an **additive term to `ell`** (or add to the logq/entropy side), and add its gradient to `g_alpha`. This helps avoid single‑component collapse.
* **Gradient clipping & LR schedules**: wrap `optax.adam` with `optax.chain(optax.clip_by_global_norm(...), optax.scale_by_adam(), optax.scale_by_schedule(...))`. Provide cosine or linear decay schedules with warmup fraction. (The training side already uses Optax; add the same style here.)
* **Entropy & α‑temperature**: allow temporary exploration: multiply logits by `1/alpha_temperature`; optionally add `entropy_bonus * H(q)` to the target term for a few hundred steps (keep default zero to keep the estimator unchanged by default).

> **Where to change**: `lambda_hat/variational.py` (init, sampling, ELBO step), `lambda_hat/sampling.py::run_vi` (read config and pass), `lambda_hat/conf/sample/sampler/vi.yaml`, `lambda_hat/config.py::VIConfig`.

---

## 2) Diagnostics & TensorBoard (scalars + images)

You already generate plots via ArviZ and save diagnostics under each run directory. We will add **optional TensorBoard logging** using **Flax’s `SummaryWriter`** (already in deps), and enrich the trace/plots with VI‑specific signals.

### 2.1 What to log (scalars)

Every `eval_every` steps, log:

* `vi/elbo`, `vi/elbo_like`, `vi/logq`
* `vi/radius2` (mean and quantiles), `vi/resp_entropy`
* `vi/pi_min`, `vi/pi_max`, `vi/pi_entropy`
* `vi/D_sqrt_min`, `vi/D_sqrt_max`, `vi/D_sqrt_med`
* `vi/A_col_norm_p95` (per‑component), `vi/grad_norm` (global)
* `vi/cumulative_fge` (work), `vi/lr` (if scheduled)
* Final **evaluation**: `vi/Eq_Ln_mc`, `vi/Eq_Ln_cv`, `vi/variance_reduction`, `vi/L0`, and the final `llc`

### 2.2 What to log (images)

Once per `k * eval_every` steps (low frequency), log:

* **Mixture weights bar chart** (π), sorted; helpful to see pruning / collapse.
* **Per‑component “scree”**: ( |A_m[:, j]|^2 ) vs rank index (reveals rank usage).
* **Histogram of (D^{1/2})** (log scale).
* **Radius trace** overlay (running stats) to help match the intended localizer radius.

### 2.3 How to integrate

* Add a `tensorboard` block in the VI run dir: `runs/targets/<tid>/run_vi_<rid>/diagnostics/tb/`.
* Use `flax.metrics.tensorboard.SummaryWriter(logdir)` from `lambda_hat/sampling.py::run_vi` **after** we receive arrays from `fit_vi_and_estimate_lambda`: emit all **stored traces** offline (no live callbacks required, which keeps JIT simple). This avoids host callbacks in JAX.
* Optionally gate with `env` or a new `sampler.vi.tensorboard: true` flag.

> **Where to change**: `lambda_hat/sampling.py::run_vi` (post‑hoc TB writing), and optionally small helpers under `lambda_hat/analysis.py` to reuse existing plotting code to produce PNGs that are then added to TB with `writer.image`.

---

## 3) “Complete the plan”: geometry whitening & stability

### 3.1 Geometry whitening (currently a TODO)

Docs indicate `use_whitening: true` but the implementation uses identity; add functional support:

* **RMSProp/Adam whitening** (`whitening_mode: rmsprop|adam`):
  Build a **diagonal preconditioner (A)** via an EMA of **squared gradients of the full loss at (w^*)** computed on small random minibatches for a short pre‑pass (e.g., one pass over a few thousand parameter‑gradient samples). Produce a vector (A_{\text{diag}} \approx \mathbb{E}[g^2]); pass it to `make_whitener(A_diag)`, which is already implemented. (You already have the minibatch gradient infra in SGLD; reuse the gradient function.) Store the EMA in memory and feed it to VI’s `whitener`.
* **HVP diagonal whitening** (`whitening_mode: hvp_diag`):
  Use a small set of random directions (v) (Rademacher or Gaussian), approximate diag(H) via Hutchinson‑style: `diag(H) ≈ mean( (H v) ⊙ v )`. This is heavier but matches the control‑variate machinery you already have for `hvp_at_wstar`. Optionally smooth by EMA.
* **Plumbing**: Add computation in `run_vi` before calling `fit_vi_and_estimate_lambda`:

  * If `whitening_mode != "none"`, produce `A_diag` (matching precision) and call `vi.make_whitener(A_diag)`.
  * Pass that whitener (instead of `None`).
* **Acceptance tests**: (i) `resp_entropy` should **increase** early (less premature collapse), (ii) **variance reduction** of the CV should not degrade, and (iii) **ELBO becomes smoother** (fewer spikes) for the same `lr`. Add a unit test that toggles whitening and asserts finite traces and non‑worse ELBO drift on a fixed seed.

### 3.2 Numeric stability reinforcements

Existing safeguards (D‑sqrt clipping, ridge on (I + A^T A), column‑norm clipping, log‑stability for (D)) are good. We’ll add:

* **Global gradient clipping** (exposed via `clip_global_norm`).
* **Bounded α temperature** (`>= 0.5`), and apply a **softmax‑with‑temperature** to reduce extreme logits early in training.
* **Guard on `logdet` and Cholesky**: if Cholesky fails (rare), add a slightly larger ridge (`1e-5 → 5e-5`) for that step only; record a counter.
* **Mask‑aware Woodbury**: ensure masked ranks don’t contribute to (A^T A) or (A^T x).
* **Pruning**: never prune the last active component; and never remove all ranks in a component (keep at least rank 1 if component survives).

---

## 4) Instrumentation in traces / ArviZ compatibility

Extend `lambda_hat/analysis.py` to optionally include **VI‑specific traces** as `sample_stats` so they appear alongside HMC/SGLD: `resp_entropy`, `elbo_like`, `logq`. Presently they’re stored but not rendered by ArviZ. Add a small block that, when present and with >1 finite values, includes them in `sample_stats`. This will make them show up in rank/trace plots and in the combined convergence view you already generate.

---

## 5) Minimal code diffs (illustrative, not exhaustive)

> **A. Schedules & clipping** (`lambda_hat/variational.py`, in `fit_vi_and_estimate_lambda`)

```python
import optax

def make_vi_optimizer(cfg):
    # schedule
    if cfg.lr_schedule == "cosine":
        sched = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.lr,
            warmup_steps=max(1, int(cfg.lr_warmup_frac * cfg.steps)),
            decay_steps=cfg.steps,
            end_value=cfg.lr * 0.1,
        )
        scale = optax.scale_by_schedule(sched)
    elif cfg.lr_schedule == "linear_decay":
        sched = optax.linear_schedule(cfg.lr, cfg.lr * 0.1, cfg.steps)
        scale = optax.scale_by_schedule(sched)
    else:
        scale = optax.identity()

    transforms = []
    if cfg.clip_global_norm is not None:
        transforms.append(optax.clip_by_global_norm(cfg.clip_global_norm))
    transforms += [optax.scale_by_adam(), scale, optax.scale(-1.0)]
    return optax.chain(*transforms)
```

Wire this optimizer where `optax.adam(lr)` is used now. Log `current_lr` via TB (if enabled) using the schedule output cached per step.

> **B. Per‑component ranks** (mask) — inside `build_elbo_step` and friends, compute `A_m_masked = A_m[:, :active_r[m]]` and pad‑safe operations (or multiply by a 0/1 column mask). This keeps shapes static for JIT while respecting rank budgets.

> **C. Dirichlet prior on π** — add to `ell` and `g_alpha`:

```python
alpha0 = jnp.asarray(cfg.alpha_dirichlet_prior or 1.0, dtype=ref_dtype)
if alpha0 > 1.0:
    # encourage spread: (alpha0 - 1) * sum(log pi)
    pi = jax.nn.softmax(params.alpha / cfg.alpha_temperature)
    ell = ell + (alpha0 - 1.0) * jnp.sum(jnp.log(pi + 1e-10))
# RB gradient addendum for alpha: derivative of prior wrt logits via softmax chain rule
```

> **D. Whitening pre‑pass** (`lambda_hat/sampling.py::run_vi`): compute `A_diag` based on `whitening_mode` by reusing your minibatch gradient function used in SGLD (already implemented), then call `vi.make_whitener(A_diag)` instead of identity.

> **E. TensorBoard write‑out** (`lambda_hat/sampling.py::run_vi`):

```python
from flax.metrics.tensorboard import SummaryWriter

tb_dir = run_dir / "diagnostics" / "tb"   # run_dir already known here
writer = SummaryWriter(tb_dir.as_posix())

T = traces["elbo"].shape[1]
for t in range(T):
    writer.scalar("vi/elbo",            float(traces["elbo"][0, t]), t)
    writer.scalar("vi/resp_entropy",    float(traces["resp_entropy"][0, t]), t)
    writer.scalar("vi/radius2",         float(traces["radius2"][0, t]), t)
    writer.scalar("vi/cumulative_fge",  float(traces["cumulative_fge"][0, t]), t)
# final metrics
writer.scalar("vi/lambda_hat", float(work["lambda_hat_mean"]), T)
writer.scalar("vi/Eq_Ln_cv",   float(work["Eq_Ln_mean"]), T)
writer.scalar("vi/L0",         float(work["Ln_wstar"]), T)
writer.flush()
```

(If you want per‑chain logs, loop over chains or log means/quantiles.)

---

## 6) Tests (extend `tests/`)

Add targeted tests (small problems, few steps) alongside `tests/test_vi_mlp.py` & `test_vi_quadratic.py`:

1. **Per‑component ranks**: `test_vi_r_per_component_mask()` — ensure masked columns don’t change the ELBO vs. an equivalent dense model with zero columns; ensure shapes are static and finite.
2. **Whitening**: `test_vi_whitening_rmsprop_improves_stability()` — compare ELBO variance (or radius spikes) with and without `whitening_mode="rmsprop"`; assert no degradation and fewer NaN guards triggered.
3. **Dirichlet prior**: `test_vi_dirichlet_prior_discourages_collapse()` — run with tiny `M`, verify `pi_entropy` increases vs. baseline.
4. **TB smoke**: `test_vi_tensorboard_writes()` — after a short run, check that an events file exists under `diagnostics/tb`.
5. **Pruning**: `test_vi_component_pruning_monotone_M()` — start with `M=6`, `prune_threshold=0.2`, verify `effective_M` does not increase and stays ≥ 1.

---

## 7) Documentation updates (user‑facing)

Update `docs/vi.md` and `docs/sweeps.md`:

* **New options table** for VI (`r_per_component`, `alpha_dirichlet_prior`, `whitening_mode`, schedules, clipping, pruning).
* **“Reading diagnostics”**: explain ELBO vs ELBO‑like, `resp_entropy` (component collapse detector), `radius2` (match (\mathbb{E}|w-w^*|^2) and (\gamma)), π bar chart, D‑sqrt histogram.
* **TensorBoard quickstart**: `tensorboard --logdir runs/targets/<tid>/run_vi_<rid>/diagnostics/tb`.
* **Sweep examples** that vary `gamma`, `M`, `r`, whitening, and LR schedules.

---

## 8) How to use these knobs + visualizations to improve VI performance (practical playbook)

**Goal:** minimize bias and variance of (\hat\lambda) at fixed work (FGEs), and keep optimization stable.

1. **Localizer strength ((\gamma))**

* Start at (10^{-3}). If `radius2` drifts high or ELBO jitters → increase (\gamma) (tighter tether). If LLC appears under‑estimated (very small radii), back off (\gamma). Watch TB: `vi/radius2` trend and `vi/elbo` smoothness.

2. **Mixture size (M) & rank (r / r_per_component)**

* Start with modest `M=8, r=1–2`. If `resp_entropy` ↓ (collapse) and `Eq_Ln_cv` still biased vs HMC/SGLD references, increase either **M** or **rank on a few components** via `r_per_component` (e.g. `[4,2,2,1,1,...]`). Use the **scree** image to see if extra rank is used (flat scree → extra columns don’t help). Prune weak components automatically with `prune_threshold`.

3. **Whitening**

* Turn on `whitening_mode="rmsprop"` for most targets. You should see **smoother ELBO**, higher **resp_entropy** early, and fewer pathological radii spikes. If gradients are too noisy, raise `whitening_decay`. HV‑diag whitening is heavier but can help when curvature is very anisotropic.

4. **Optimizer & schedules**

* If ELBO is noisy: enable `clip_global_norm=5.0`, reduce `lr`, and switch to `lr_schedule=cosine` with a small warmup. In TB, `vi/elbo` should stop exploding and `vi/radius2` should stop spiking.

5. **Prevent component collapse**

* If π collapses: add a tiny `alpha_dirichlet_prior=1.05–1.2` and/or set `alpha_temperature=1.5` for the first ~10–20% of steps. Watch `pi_entropy` rise. Turn back to 1.0 later.

6. **Control variate quality**

* If `variance_reduction < 1.0`, increase `eval_samples` modestly and check whitening; poor whitening and poor subspace rank often hurt the HVP‑CV. The TB scalars `vi/Eq_Ln_cv` vs `vi/Eq_Ln_mc` help verify the CV is doing work.

7. **Comparing against references**

* Use the existing combined convergence & WNV plots to compare VI vs HMC/SGLD at equal **FGEs**; VI should reach comparable confidence with far fewer FGEs when tuned and whitened.

---

## 9) Acceptance criteria / “done means”

* **Config**: New fields compile; old configs remain valid.
* **Whitening**: With `whitening_mode=rmsprop`, VI runs with fewer ELBO/radius spikes (quantified: reduced 95th‑percentile absolute ELBO step‑to‑step change) on the “tiny MLP” tests.
* **Diagnostics**: TensorBoard shows the scalars/images listed; events file present under `diagnostics/tb/`.
* **Stability**: No NaNs with default settings on the provided test suites; tests pass (`pytest -q`).
* **Performance**: On a small benchmark target, WNV (work‑normalized variance) for VI improves relative to current baseline (measured by your `llc_wnv.png` logic).

---

## 10) File‑by‑file checklist

* `lambda_hat/config.py`: extend `VIConfig` (above).
* `lambda_hat/conf/sample/sampler/vi.yaml`: add new keys with sane defaults.
* `lambda_hat/sampling.py`:

  * In `run_vi`: compute optional **whitener**; pass config to VI; write TensorBoard logs post‑hoc from `traces`/`extras`.
  * Export `resp_entropy`, `elbo_like`, etc., as `sample_stats` in `analysis.py` (minor).
* `lambda_hat/variational.py`:

  * `init_vi_params` → handle `r_per_component` (mask).
  * `sample_q`, `logpdf_components_and_resp` → honour masks & α temperature.
  * `build_elbo_step` → add Dirichlet prior term, entropy bonus, gradient clipping via optimizer, and schedule; keep STL/RB separation intact.
  * Component pruning after `eval_every` (cheap pass).
* `lambda_hat/analysis.py`: include VI extras in ArviZ `sample_stats` when present; no behavior change for other samplers.
* `docs/vi.md`, `docs/sweeps.md`: detail new options and TB usage; add sweep examples.
* `tests/…`: add the five tests noted above.

---

### Closing note

This plan keeps your **variational family** intact while adding the **control** (rank/mixture budgets, schedules, priors), **geometry** (whitening), and **observability** (TB + richer traces) you need to systematically improve performance and stability. It is aligned with how VI is written and integrated today (JAX/Optax, Optax schedules, ArviZ, config‑driven orchestration), so the changes are incremental and well‑scoped across `variational.py`, `sampling.py`, configs, and docs.

If you want, I can draft the concrete diffs for `VIConfig` + `run_vi` + the TB writer first, then proceed to the whitening pre‑pass and the mask‑based per‑component ranks.
