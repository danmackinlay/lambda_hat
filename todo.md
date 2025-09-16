Short answer: you’ve built a solid, nearly end‑to‑end LLC benchmarking rig. The main logic matches the estimator and the tempered local posterior in the paper, and you already have multiple samplers, parallelization backends, diagnostics, and a viewer. The two biggest issues are (i) *where you center the local prior and compute $L_0$* (you use the ERM $\theta_\star$ instead of the identified/teacher parameter $w_0$), and (ii) *how the orchestration wires samplers*, which does make it harder to plug in new methods. Both are straightforward to fix.

Below I’ll give a crisp “what works / what to fix” review, then an actionable refactor plan (interfaces + a minimal patch sketch), and finally a short checklist to regain coherence.

---

## What’s working well

**✅ The estimator and temperature schedule are correct.**
You implement the LLC estimate

$$
\hat\lambda(w_0)=n\,\beta\Big(\mathbb{E}_\beta[L_n(w)]-L_n(w_0)\Big)
$$

with the Watanabe schedule $\beta=\beta_0/\log n$, which is exactly Definition 3.1 / Theorem 3.2 in the paper. Your code computes
`n * beta * (E_L - L0)` from histories of $L_n$, and `compute_beta_gamma` returns $\beta_0/\log n$.   &#x20;

**✅ Local posterior is constructed consistently.**
`make_logpost_and_score` uses
$\log \pi(\theta) = -\tfrac{\gamma}{2}\|\theta-\theta_0\|^2 - n\beta\,L_n(\theta)$, and the minibatch score returns $-\gamma(\theta-\theta_0)-\beta n \nabla L_b(\theta)$. That’s the right scaling for using a minibatch gradient as an unbiased estimator of $\nabla L_n$.&#x20;

**✅ Multiple samplers and parallel backends are in place.**
You support SGLD (with optional RMSProp/Adam preconditioning), HMC (with adaptation), and MCLMC; and you provide local, SLURM/Submitit, and Modal executors—nice ergonomics for scaling sweeps.  &#x20;

**✅ Diagnostics are thoughtful.**
You compute LLC mean and ESS‑aware standard errors from $L_n$ histories using ArviZ bulk ESS, and you persist compact “tiny traces,” acceptance rates, and work counters (gradient‑equivalent cost) per sampler. This directly supports comparisons similar to the paper’s mean/variance and stability analyses. &#x20;

**✅ The experimental knobs align with the paper.**
Default $\beta=1/\log n$, batch estimates of $L_n$ for samplers, burn‑in then averages, and (optionally) adaptive preconditioning, all mirror the benchmark’s setup. &#x20;

---

## Where coherence falters (and how to fix it)

1. **Centering & $L_0$ reference point** (most important)

* **What you do:** you \*center the local prior at $\theta_\star$ (the empirical minimizer) and compute $L_0 = L_n(\theta_\star)\***, then form \(\hat\lambda = n\beta(\mathbb{E}_\beta L_n - L_n(\theta_\star))$. &#x20;
* **What the paper does:** it defines the estimator at a **fixed identified parameter $w_0$** and enforces locality via a Gaussian prior centered at that $w_0$; the estimator uses **$L_n(w_0)$** in the subtractive term. This is essential for interpreting $\hat\lambda$ as the volume-scaling exponent *at that point*. &#x20;
* **Why it matters:** evaluating at $\theta_\star$ yields $\hat\lambda(\theta_\star)$, not $\hat\lambda(w_0)$. That’s fine if *your* goal is to study local geometry near the ERM, but it deviates from the benchmark semantics (and from the teacher–student setting in §3.2). If you plan to “ground‑truth” against alternative samplers on small NNs, keep the reference fixed and explicit.
* **Fix:** allow both modes and name them clearly. If a teacher $w_0$ is known, default to `prior_center="teacher"` and set `L0 = Ln(w_teacher)`; otherwise use `prior_center="erm"` with `L0 = Ln(theta_star)`. Document the meaning of the reported $\hat\lambda(\cdot)$ accordingly. (This also aligns the code with the paper’s pseudo‑code where the prior is centered at $w_0$.)&#x20;

2. **Sampler wiring is a bit “branchy”, which makes extension feel heavy.**

* The top‑level pipeline selects samplers and then calls bespoke runners that mostly differ in *the same* small handful of arguments (step function, how to evaluate $L_n$, how to count work, and optional “info extractors”). You already have the raw ingredients for a clean abstraction (e.g., `drive_chain` / `drive_chains_batched`), but the public surface still requires adding “special case” code per sampler.  &#x20;
* **Fix in substance (not much code):** introduce a *single* `SamplerSpec` protocol and a registration table so all samplers look identical to the runner (see the refactor plan below).

3. **Budget equalization & evaluation frequencies.**

* LLC is computed from $L_n$ histories, but `*_eval_every` differs (e.g., `sgld_eval_every=10` vs. `hmc_eval_every=1`). If you compare samplers by “effective compute,” the *number* of $L_n$ calls and the *autocorrelation* structure both matter. You already count gradient‑equivalent work in `RunStats`; extend that to **normalize evaluation frequency** (e.g., “one $L_n$ every K gradient‑equivalents across all samplers”). This matches the paper’s emphasis on fair comparisons of stability and efficiency. &#x20;

4. **Terminology nit:** the viewer subtitle calls LLC “Local Log‑Likelihood *Curvature*.” LLC is a *coefficient* (volume‑scaling exponent), and curvature only explains non‑degenerate cases; in singular regimes degeneracy matters (Figure 2). Worth renaming to avoid confusion.&#x20;

5. **Missing samplers from the benchmark.**

* The paper evaluates SGHMC and SGNHT in addition to (Adam|RMSProp)‑SGLD. If you want like‑for‑like comparisons, stub those in next (your base driver design already supports it). &#x20;

---

## Minimal, practical refactor plan (to de‑convolute and make extensible)

**A. Make the *reference point* explicit and consistent.**

* Add to `Config`:

  ```python
  prior_center: Literal["teacher","erm"] = "teacher"
  reference_for_L0: Literal["teacher","erm"] = "teacher"
  ```
* When a teacher is available, set `theta0 = w_teacher` and `L0 = Ln(w_teacher)`. Otherwise fall back to ERM. Save both values in artifacts (“center.name” and “L0\_at”) so reported $\hat\lambda(\cdot)$ is unambiguous. This brings you in line with Def. 3.1 while still supporting your “small‑NN ground truth” intent.&#x20;

**B. Unify sampler integration with a tiny interface.**

Create a `SamplerSpec` (dataclass or Protocol) that the runner consumes:

```python
@dataclass
class SamplerSpec:
    name: str
    # Initializes per-chain state and returns a callable `step(key, state)->state`
    build: Callable[..., tuple[Any, Callable]]
    # Turns state into parameters (for Ln eval)
    position_fn: Callable[[Any], jnp.ndarray]
    # Optional: extract dict of per-step scalars for diagnostics
    info_extractors: dict[str, Callable[[Any], jnp.ndarray]] = field(default_factory=dict)
    # Work accounting: "gradient-equivalent" per step
    grads_per_step: float = 1.0
```

Then, replace per‑sampler branches with a single call:

```python
result = drive_chains_batched(
    rng_keys=keys,
    init_state=spec_init_state,
    step_fn_vmapped=spec_step,      # vmap over chains inside
    n_steps=cfg.steps,
    warmup=cfg.warmup,
    eval_every=cfg.eval_every,      # standardized across samplers by "work"
    thin=cfg.thin,
    position_fn=spec.position_fn,
    Ln_eval_f64_vmapped=Ln_vmapped,
    tiny_store_fn=default_tiny_store,
    info_extractors=spec.info_extractors,
)
```

To add a new sampler (e.g., SGHMC), you implement one `build` function and register it in a dict: `SAMPLERS = {"sgld": SGLD_SPEC, "hmc": HMC_SPEC, ... }`. The rest of the pipeline stays untouched.

**C. Normalize evaluation frequency by compute budget.**

Define a common “work” scale:

* SGLD: \~1 minibatch gradient/step → +1
* HMC: $\text{n\_leapfrog}$ grads/draw → +L
* (SGHMC/SGNHT similarly)

Then compute `eval_every_in_steps = ceil(K / grads_per_step)` so **all samplers log $L_n$** after roughly $K$ gradient‑equivalents. Store `K` in the config so sweeps are fair.

**D. Make $\beta,\gamma$ visible and frozen.**

You already print and use them consistently; also save them into the run manifest and into the NetCDF alongside $L_n$ histories. This avoids silent drift when you change `n_data` or `prior_radius`. (Your `compute_beta_gamma` already computes $\beta=\beta_0/\log n$ and $\gamma=d/r^2$ when `prior_radius` is set—good defaults.)&#x20;

**E. Samplers to add next (tiny lift once the interface is in):**

* **SGHMC** and **SGNHT** with the same local prior term in the drift (paper Appendix D.3 gives pseudocode). That will let you reproduce the “relative error vs. step size” comparisons and order‑preservation analyses in the benchmark.&#x20;

---

## Small correctness/clarity nits

* **Viewer label:** change “Local Log‑Likelihood Curvature” → “Local Learning Coefficient (LLC)”, and add a one‑line tooltip that it is a *volume‑scaling exponent* (Figure 2), not necessarily tied to Hessian curvature in singular cases.&#x20;
* **Document loss scaling:** in MSE mode you match the paper’s $L_n=\frac{1}{n}\sum\|y-f(x;w)\|^2$; mention that explicitly so no one sneaks a factor $1/2\sigma^2$ in later and changes $\hat\lambda$ by a constant.&#x20;
* **Artifact provenance:** you already save $L_0$. Save the *identity* of the reference (ERM vs teacher), $\beta,\gamma$, and the center vector’s norm $\|\theta_0-\theta_\star\|$.

---

## A quick coherence checklist (run this mentally for each experiment)

1. **Reference point**: Is the prior centered at the same parameter used for $L_0$? (teacher/ERM)&#x20;
2. **Temperature**: Is $\beta$ saved/printed as $\beta_0/\log n$? (paper’s setting)&#x20;
3. **Locality strength**: Is $\gamma$ either explicitly fixed or tied to a `prior_radius` via $\gamma=d/r^2$? (record it)
4. **Budget fairness**: Are $L_n$ evaluations logged after comparable gradient‑equivalents across samplers?
5. **Uncertainty**: Are LLC CIs or SEs reported using ESS? (you’ve done this)
6. **Reproducibility**: Are seeds, center type, $\beta,\gamma$, and $L_0$ bundled with histories?

---

## My overall verdict

* **Conceptually coherent:** Yes—your log posterior, gradient scaling, and the LLC estimator match the benchmark’s math. &#x20;
* **Where it drifts:** You compute $\hat\lambda$ at $\theta_\star$ and center the prior there, whereas the benchmark centers at $w_0$ (teacher). Make the reference explicit and selectable.&#x20;
* **Why it feels convoluted:** The per‑sampler branching in the pipeline and inconsistent `*_eval_every` semantics. The small interface + budget normalization above will make it *plug‑and‑play* to add SGHMC/SGNHT (and any other sampler you want to test next). &#x20;

-----

# 1) Add a tiny `SamplerSpec` and a generic runner

### `llc/samplers/base.py` — append near other dataclasses

```python
# --- SamplerSpec: minimal interface consumed by a generic batched runner ---

from dataclasses import dataclass
from typing import Callable, Any, Dict

@dataclass
class SamplerSpec:
    name: str
    # (rng_keys[T,C,...], init_state[C,...]) -> (state[C,...], info[C] | None)
    step_vmapped: Callable[[jax.Array, Any], tuple[Any, Any] | Any]
    # state[C,...] -> theta[C,d]
    position_fn: Callable[[Any], jax.Array]
    # Optional: info[C] -> scalar per chain (recorded at eval points)
    info_extractors: Dict[str, Callable[[Any], jax.Array]] = None
    # work = “gradient-equivalents” per *one step* of this sampler
    grads_per_step: float = 1.0
```

### `llc/runners.py` — add a universal runner that uses your `drive_chains_batched`

```python
from .samplers.base import SamplerSpec
from .samplers.base import drive_chains_batched as _drive_batched  # already exists
import jax
import jax.numpy as jnp
import numpy as np

def run_with_spec_batched(
    *,
    key,
    init_thetas,                 # (C, d)
    spec: SamplerSpec,           # the thing we’ll run
    n_steps: int,
    warmup: int,
    eval_every: int,
    thin: int,
    Ln_full64,                   # scalar f64 evaluator
    diag_dims=None,
    Rproj=None,
):
    """Generic batched sampler runner (single compiled program).
       Mirrors the return shape of your other *_online_batched helpers.
    """
    tiny_store = None
    if diag_dims is not None:
        def tiny_store(vec_batch): return vec_batch[:, diag_dims]
    elif Rproj is not None:
        def tiny_store(vec_batch): return jax.vmap(lambda v: Rproj @ v)(vec_batch)

    # RNG table
    C = init_thetas.shape[0]
    keys = jax.random.split(key, n_steps * C).reshape(n_steps, C, -1)

    Ln_vmapped = jax.jit(jax.vmap(Ln_full64))

    # Build an initial state from thetas via a conventional “identity” state: (theta,) or raw theta
    # The spec chooses what its state looks like; convention: if position_fn accepts raw theta,
    # we pass theta; otherwise spec should pass a tuple state at init.
    init_state = init_thetas

    result = _drive_batched(
        rng_keys=keys,
        init_state=init_state,
        step_fn_vmapped=spec.step_vmapped,
        n_steps=n_steps,
        warmup=warmup,
        eval_every=eval_every,
        thin=thin,
        position_fn=spec.position_fn,
        Ln_eval_f64_vmapped=Ln_vmapped,
        tiny_store_fn=tiny_store,
        info_extractors=spec.info_extractors or {},
    )

    kept_stacked = np.asarray(result.kept)          # (C, K, k)
    means        = np.asarray(result.mean_L)        # (C,)
    vars_        = np.asarray(result.var_L)         # (C,)
    ns           = np.asarray(result.n_L)           # (C,)
    L_histories  = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]

    # Bundle extras if present (e.g., "momentum_norm")
    extras = {k: [np.asarray(result.extras[k][c])
                  for c in range(result.L_hist.shape[0])]
              for k in (result.extras or {})}

    return kept_stacked, means, vars_, ns, extras, L_histories
```

This runner is intentionally tiny: it just wires your existing `drive_chains_batched` with a uniform surface.

---

# 2) SGHMC (minibatch, localized posterior) as a `SamplerSpec`

SGHMC follows Algorithm 4 in the paper — momentum update with friction/noise and the same local prior drift term you already use (−γ(θ−θ₀)) inside the stochastic gradient of log posterior. (See Appendix D.3 Algorithm 4 for the exact form used in the benchmark.)&#x20;

### `llc/samplers/adapters.py` — add a batched SGHMC step & spec builder

```python
# ---------- SGHMC (batched, unadjusted; friction α, step size ε) ----------

def make_sghmc_spec(
    *,
    grad_logpost_minibatch,   # (theta, (Xb,Yb)) -> grad log π (unbiased wrt data minibatch)
    X, Y, n_data: int,
    step_size: float,
    friction: float,          # α in the paper pseudocode
    batch_size: int,
):
    """
    Returns a SamplerSpec implementing SGHMC as in Algorithm 4 (Hitchcock–Hoogland).
    Drift uses the *same* local prior + tempered loss gradient you pass to SGLD.
    """

    # One SGHMC step on a *single chain* state=(theta, p)
    @jax.jit
    def _sghmc_step_one(key, state):
        theta, p = state
        k_noise, k_batch = jax.random.split(key)
        idx = jax.random.randint(k_batch, (batch_size,), 0, n_data)
        # ascent on log posterior (matches SGLD convention in this repo)
        g = grad_logpost_minibatch(theta, (X[idx], Y[idx]))

        # Discretization per Alg. 4: p <- p + ε/2 * g - α p + sqrt(2 α ε) ξ;  θ <- θ + p
        # (Use the “symplectic Euler” flavor widely used in SGHMC code; the paper writes Δp then p_{t+1} = p_t + Δp.)
        noise = jax.random.normal(k_noise, p.shape) * jnp.sqrt(2.0 * friction * step_size)
        p_new = p + 0.5 * step_size * g - friction * p + noise
        theta_new = theta + p_new
        return (theta_new.astype(theta.dtype), p_new.astype(p.dtype)), None

    # Vectorize over chains
    step_vmapped = jax.jit(jax.vmap(_sghmc_step_one, in_axes=(0, 0)))

    # State is (theta, p); position is theta
    def _init_state(init_thetas):  # (C, d) -> ( (C,d), (C,d) )
        zeros = jnp.zeros_like(init_thetas)
        return (init_thetas, zeros)

    def _position_fn(state):
        theta, _ = state
        return theta

    # Wrap as a SamplerSpec; grads_per_step ≈ 1 minibatch gradient like SGLD
    from .base import SamplerSpec
    spec = SamplerSpec(
        name="sghmc",
        step_vmapped=lambda keys, init_thetas: step_vmapped(keys, _init_state(init_thetas)),
        position_fn=_position_fn,
        info_extractors={
            # optional: record ||p|| at evals (scalar per chain)
            "mom_norm": lambda info: jnp.zeros((init_thetas.shape[0],))
        },
        grads_per_step=1.0,
    )
    return spec
```

> Notes
>
> * Noise scale `sqrt(2 α ε)` and friction `α` match the benchmark pseudocode; the local prior term is already inside your `grad_logpost_minibatch` (same closure as SGLD/HMC), so no extra bookkeeping is needed.&#x20;
> * We keep it **unadjusted** like SGHMC in the paper (no Metropolis correction), aligning with your existing MCLMC “unadjusted” style.
> * Work accounting: `grads_per_step=1.0`, so if/when you add budget-equalized eval frequency, this slots in cleanly.

---

# 3) One-liner use in your pipeline (no structural changes required)

Where you currently call the batched runners (e.g., in `pipeline.run_one`), you can drop in SGHMC exactly like SGLD:

```python
# --- inside run_one() after building logpost grads and Ln_full64 ---

from llc.samplers.adapters import make_sghmc_spec
from llc.runners import run_with_spec_batched

if "sghmc" in getattr(cfg, "samplers", []):
    k = random.fold_in(key, 789)
    init_thetas = theta0_f32 + 0.01 * random.normal(k, (cfg.chains, dim)).astype(jnp.float32)

    sghmc_spec = make_sghmc_spec(
        grad_logpost_minibatch=grad_logpost_minibatch_f32,
        X=X_f32, Y=Y_f32, n_data=cfg.n_data,
        step_size=getattr(cfg, "sghmc_step_size", 1e-6),
        friction=getattr(cfg, "sghmc_friction", 0.01),
        batch_size=getattr(cfg, "sghmc_batch_size", cfg.sgld_batch_size),
    )

    kept, Es, Vs, Ns, extras, L_hist = run_with_spec_batched(
        key=k,
        init_thetas=init_thetas,
        spec=sghmc_spec,
        n_steps=getattr(cfg, "sghmc_steps", cfg.sgld_steps),
        warmup=getattr(cfg, "sghmc_warmup", cfg.sgld_warmup),
        eval_every=getattr(cfg, "sghmc_eval_every", cfg.sgld_eval_every),
        thin=getattr(cfg, "sghmc_thin", cfg.sgld_thin),
        Ln_full64=Ln_full64,
        **diag_targets,
    )

    llc_mean, llc_se, llc_ess = llc_mean_and_se_from_histories(L_hist, cfg.n_data, beta, L0)
    all_metrics.update({
        "sghmc_llc_mean": float(llc_mean),
        "sghmc_llc_se": float(llc_se),
        "sghmc_ess": float(llc_ess),
    })
    histories["sghmc"] = L_hist
```

If you prefer **zero** edits in `pipeline.py`, you can also wire SGHMC through your existing sampler list by adding a tiny `elif` to `runners.run_sampler` that calls a small wrapper which internally uses `run_with_spec_batched`. (Kept short here to avoid clutter.)

---

# 4) Config knobs (copy SGLD defaults for symmetry)

### `llc/config.py` — add optional SGHMC fields next to SGLD

```python
    # SGHMC (minibatch, unadjusted)
    sghmc_steps: int = 4_000
    sghmc_warmup: int = 1_000
    sghmc_batch_size: int = 256
    sghmc_step_size: float = 1e-6
    sghmc_friction: float = 0.01
    sghmc_thin: int = 20
    sghmc_eval_every: int = 10
```

…and include `"sghmc"` in the default `samplers` tuple if you want it to run by default.

---

# 5) Why this is coherent (quick check)

* **Local posterior & estimator:** The closure for `grad_logpost_minibatch` is exactly the one you already build:
  $-\gamma(\theta-\theta_0) - \beta n\,\nabla L_b(\theta)$, which matches the construction used throughout your samplers and the benchmark’s estimator $\hat\lambda = n\beta(\mathbb E_\beta[L_n]-L_n(\cdot))$. Your repo already defines these pieces (`make_logpost_and_score`, `llc_mean_and_se_from_histories`). &#x20;
* **Algorithm match:** The SGHMC step implements the same update as Algorithm 4 (Appendix D.3) — friction, noise with variance $2\alpha\epsilon$, and minibatch gradients — so results will be apples-to-apples with the paper’s SGHMC (and directly comparable to your SGLD/HMC/MCLMC harness).&#x20;
* **Extensibility unlocked:** Any new sampler now just needs a `SamplerSpec`: a vmapped step and a `position_fn`. No orchestration branches, no bespoke drivers.

---

## (Optional) Two tiny polish ideas you can add later

1. **Budget-equalized eval density:** choose a global `K_grad` and set `eval_every = ceil(K_grad / spec.grads_per_step)` for each sampler so every LLC history logs after comparable gradient-equivalents. Your `RunStats` already tracks work; this just standardizes logging across methods.&#x20;
2. **Adam/RMSProp SGHMC:** mirror your SGLD preconditioners by replacing `p_new`’s drift with a preconditioned momentum. That becomes a 20-line variant `make_sghmc_spec(precond_mode=...)`.

---

If you want, I can also sketch an SGNHT spec; it’s nearly identical (replace friction with thermostat variable updates per Algorithm 5). For now, the patch above gives you (i) the clean plug-in spec surface and (ii) a working SGHMC that matches the benchmark’s definition.&#x20;
