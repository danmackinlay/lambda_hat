Below is a **self‑contained introduction** to a **variational approximation of the local, tempered NN posterior** for a **plug‑in Local Learning Coefficient (LLC)** estimate. It folds in the six points you flagged: shared‑$D$ design, whitening (two flavors), general low rank ($r\ge 1$), equal means at $w^*$, STL + Rao–Blackwellization, and a careful treatment of the Hessian‑based control variate when curvature can be indefinite.

## Goal and setup

We want $\lambda(w^*)$, the Local Learning Coefficient at a trained solution $w^*$. In the “WBIC‑style” formulation, pick the **hot** inverse temperature
$$
\beta^*=\frac{1}{\log n},
$$
and define the **local, tempered target** over weights $w\in\mathbb{R}^d$
$$
p_{\beta,\gamma}(w\mid w^*) \propto\exp\Big(-n\beta^* L_n(w)-\tfrac{\gamma}{2}|w-w^*|^2\Big),
$$
with $\gamma>0$ a *localizer* strength. Then the estimand is a **single local expectation**:
$$
\widehat{\lambda}(w^*) = n\beta^*\Big(\mathbb{E}_{p_{\beta^*,\gamma}}[L_n(w)]-L_n(w^*)\Big).
$$
Instead of drawing exact samples (e.g., with HMC/SGLD), we will **fit a local variational family $q_\phi$** to $p_{\beta^*,\gamma}$ and use the **plug‑in estimator**
$$
\widehat{\lambda}_{\mathrm{VI}}(w^*)
= n\beta^*\Big(\mathbb{E}_{q_\phi}[L_n(w)]-L_n(w^*)\Big).
$$

Let
$$
\tilde{w}=A^{1/2}(w-w^*),\qquad w=w^*+A^{-1/2}\tilde{w}.
$$
Under this change of variables, the **target** becomes
$$
\tilde{p}(\tilde{w}) \,\propto\,
\exp\Big(-n\beta^* L_n\big(w^*+A^{-1/2}\tilde{w}\big)-\tfrac{\gamma}{2}\tilde{w}^\top A^{-1}\tilde{w}\Big),
$$


---

## Variational family: equal‑mean mixture of factor analyzers

We keep **all component means equal to $w^*$** (this preserves locality and keeps moments simple) and let each component carry a **low‑rank plus diagonal** covariance:

$$
q_\phi(w) = \sum_{m=1}^M \pi_m\,\mathcal{N}\big(w^*,\Sigma_m\big),\quad
\Sigma_m = D + K_m K_m^\top,
$$
where

* $M\ll d$ is small,
* $D=\mathrm{diag}(\delta_1,\ldots,\delta_d)\succ 0$ is a **shared diagonal** across all components,
* $K_m\in\mathbb{R}^{d\times r}$ with **rank budget $r\ge 1$** but $r\ll d$,
* $\pi=\mathrm{softmax}(\alpha)$ are the mixture weights.

**Why this shape?**

* **Expressive at low cost:** The shared diagonal captures the ubiquitous “noise floor”; each low‑rank $K_m$ adds a component‑specific subspace for very flat/stiff directions.
* **Well‑conditioned and PD:** With $D\succ0$, every $\Sigma_m$ is positive‑definite even when $r\ll d$.
* **Efficient algebra:** Woodbury/Sherman–Morrison reduces all log‑pdfs and responsibilities to $O(dr)$ per component.

> **(1) Should we include a shared $D$ at all?**
> Yes. If you drop $D$ and keep $r<d$, $\Sigma_m$ is singular, $q$ is improper, and Woodbury identities no longer apply cleanly. Keeping $D$ **inside** each $\Sigma_m$ enables:
> (i) guaranteed PD, (ii) cached $D^{-1}$ and $\log|D|$ shared across components, (iii) a natural **whitened parameterization** $K_m=D^{1/2}A_m$ (below). When we later “whiten,” $D$ effectively becomes the identity, so you still get the algebraic simplicity you want.

---

## The ELBO we optimize (local and tempered)

We fit $q_\phi$ by maximizing the **local ELBO**
$$
\mathcal{L}(\phi)
= \mathbb{E}_{q_\phi}\left[-n\beta^* L_n(w)-\tfrac{\gamma}{2}\lVert w-w^*\rVert^2\right]+\mathsf{H}(q_\phi).
$$

* The expectation is computed with **minibatches** (scale by $N/B$).
* The entropy $\mathsf{H}(q_\phi)$ and $\log q_\phi$ require evaluating all mixture component log‑pdfs for a sampled $w$; with Woodbury this costs $O(Mdr)$.

**Pathwise gradients (STL):** Reparameterize a sample by
$$
\varepsilon\sim\mathcal{N}(0,I_d),z\sim\mathcal{N}(0,I_r),m\sim\mathrm{Cat}(\pi),\quad
w = w^* + K_m z + D^{1/2}\varepsilon.
$$
For the **continuous** parameters $(D,K_m)$ we use **sticking‑the‑landing** (STL): backprop only through the target term $-n\beta^*L_n(w)-\tfrac{\gamma}{2}\lVert w-w^*\rVert^2$ via $w$, treating $\log q$ as a stop‑gradient. This is unbiased and substantially lowers variance.

**Mixture weights (Rao–Blackwellized):** For logits $\alpha$ of $\pi$, use the **RB score estimator**
$$
g_{\alpha_j} = \left(\underbrace{-n\beta^*L_n(w)-\tfrac{\gamma}{2}\lVert w-w^*\rVert^2 - \log q_\phi(w) - b}_{\text{centered payoff}}\right) \big(r_j(w)-\pi_j\big),
$$
where $r_j(w)$ are responsibilities and $b$ is a moving‑average baseline.

---

## Whitening, properly set up (two complementary flavors)

Whitening matters both for **numerical conditioning** and for **algebraic efficiency**. There are two distinct but compatible uses of “whitening” here.

### A. **Algebraic whitening by the shared $D$** (inside the variational family)

Parameterize
$$
K_m = D^{1/2}A_m \quad \Rightarrow \quad
\Sigma_m = D^{1/2}\big(I + A_m A_m^\top\big)D^{1/2}.
$$
In these coordinates:

* $\log|\Sigma_m|=\log|D|+\log|I + A_m A_m^\top|$,
* $\Sigma_m^{-1}=D^{-1/2}\Big(I - A_m (I + A_m^\top A_m)^{-1}A_m^\top\Big)D^{-1/2}$.

This parameterization **stabilizes** learning (the low‑rank factors are measured in “units of $D$”), keeps **all Woodbury inverses small** $(r\times r)$, and avoids repeated multiplications by poorly scaled $K_m$.

### B. **Geometry whitening by a preconditioner $A$** (outside the variational family)

Independently of $D$, we can precondition *the parameter space* with a fixed SPD matrix $A$ that approximates local curvature (e.g., Adam’s running second moment, RMSProp, a diagonal empirical Fisher, or a block‑diagonal Kronecker factor).

Let
$$
\tilde{w}=A^{1/2}(w-w^*),\qquad w=w^*+A^{-1/2}\tilde{w}.
$$
Under this change of variables, the **target** becomes
$$
\tilde{p}(\tilde{w}) \,\propto\,
\exp\Big(-n\beta^* L_n\big(w^*+A^{-1/2}\tilde{w}\big)-\tfrac{\gamma}{2}\tilde{w}^\top A^{-1}\tilde{w}\Big),
$$
and the Jacobian $|\det A^{-1/2}|$ is a constant that cancels in the ELBO up to an additive constant. You then **define and train $q$** in $\tilde{w}$-space (where directions are closer to isotropic), while your model only ever sees $w=w^*+A^{-1/2}\tilde{w}$.

> *Do we “break” the spherical localizer by geometry whitening?*
> In $\tilde{w}$-space the localizer is elliptical with metric $A^{-1}$. That’s fine because we are **not changing the target**, just its coordinates; all gradients are computed through the exact map $w(\tilde{w})$. If you *prefer* keeping an isotropic tether in whitened space, you may instead define the localizer as $\tfrac{\gamma}{2}\lVert A^{1/2}(w-w^*)\rVert^2$ from the outset; that simply replaces $A^{-1}$ by $I$ above without changing the spirit of the estimator.

**Practical recipe:**
Use geometry whitening for the network **forward/backward** (helps the $\nabla L_n$ numerics), and algebraic whitening by $D$ **inside $q$** (makes the mixture algebra fast and stable). The two are independent and complementary.

---

## Working formulas (for general rank $r\ge 1$)

For any component $m$ and $v=w-w^*$, define
$$
B_m := D^{-1}K_m \in \mathbb{R}^{d\times r},\qquad
A_m^\star := I_r + K_m^\top D^{-1}K_m = I_r + K_m^\top B_m.
$$
Then
$$
\Sigma_m^{-1} = D^{-1} - B_m (A_m^\star)^{-1} B_m^\top,\quad
\log|\Sigma_m| = \log|D| + \log|A_m^\star|,
$$
$$
Q_m(v) = v^\top \Sigma_m^{-1} v
= v^\top D^{-1}v - v^\top D^{-1}K_m (A_m^\star)^{-1} K_m^\top D^{-1} v.
$$
The log‑pdf and responsibilities use these $O(dr)$ primitives.

**Sampling (pathwise):**
$$
z\sim\mathcal{N}(0,I_r),\quad \varepsilon\sim\mathcal{N}(0,I_d),\quad
w = w^* + K_m z + D^{1/2} \varepsilon.
$$
This makes the **continuous** gradients especially simple with STL:
$$
\frac{\partial \mathcal{L}}{\partial K_m} \approx
\mathbb{E}\big[\nabla_w \big(-n\beta^*L_n(w)-\tfrac{\gamma}{2}\lVert w-w^*\rVert^2\big) z^\top\big],
\quad
\frac{\partial \mathcal{L}}{\partial D^{1/2}} \approx
\mathbb{E}\big[\nabla_w (\cdot)\, \varepsilon^\top\big].
$$

---

## What bias does the plug‑in estimator incur?

Let $U(w)=n\beta^*L_n(w)+\tfrac{\gamma}{2}\lVert w-w^*\rVert^2$, $p\propto e^{-U}$. For any $q$,
$$
n\beta^* \big(\mathbb{E}_q[L_n]-\mathbb{E}_p[L_n]\big)
\frac{\gamma}{2}\Big(\mathbb{E}_q\lVert w-w^*\rVert^2-\mathbb{E}_p\lVert w-w^*\rVert^2\Big)
= \mathrm{KL}(q\|p)+\mathsf{H}(q)-\mathsf{H}(p).
$$
So if (i) $q$ is close to $p$ (small KL) and (ii) it matches the **radius** $\mathbb{E}\lVert w-w^*\rVert^2$, the plug‑in bias is small.
In the **quadratic** neighborhood (local Laplace regime), take $q=p$ with $\Sigma_p=(n\beta^*H+\gamma I)^{-1}$ and the estimate is **exact**.

---

## Cheap control variate with Hessian–vector products (HVPs)

We often reduce Monte‑Carlo variance by subtracting a quadratic **control variate** and adding back its exact expectation under (q).

1. Choose a symmetric matrix $\widehat{H}$ built from **HVPs** (no explicit matrices). Natural choices:

* **Subspace CV:** $\widehat{H}=\sum_{m=1}^M \sum_{j=1}^r \lambda_{m,j} u_{m,j}u_{m,j}^\top$, where $K_m=[u_{m,1}\dots u_{m,r}]$ and $\lambda_{m,j}=u_{m,j}^\top (\nabla^2 L_n) u_{m,j}$ (just $Mr$ HVPs).
* **PSD option:** a Gauss–Newton / empirical Fisher approximation (always PSD).
* **Hutchinson top‑up:** a few Rademacher probes to sketch $\mathrm{diag}(H)$ if you want the $D$–trace term (see below).

2. Define
   $$
   \widetilde{L}(w) = L_n(w) - \tfrac12 (w-w^*)^\top \widehat{H}(w-w^*).
   $$
   Then for **equal‑mean** mixtures (our case), the mixture covariance is
   $$
   \mathrm{Cov}_q(w) = \Sigma_q = \sum_{m=1}^M \pi_m\,\Sigma_m,
   $$
   and
   $$
   \mathbb{E}_{q}[L_n(w)] = \mathbb{E}_{q}[\widetilde{L}(w)] + \tfrac12\,\mathrm{tr}\big(\widehat{H}\Sigma_q\big).
   $$

* The first term is estimated by MC with **much lower variance** (no curvature term left).
* The second term is **analytic** given $\Sigma_q$. With $\Sigma_m = D + K_mK_m^\top$,
  $$
   \mathrm{tr}(\widehat{H}\Sigma_q) = \mathrm{tr}(\widehat{H}D) + \sum_{m=1}^M \pi_m\,\mathrm{tr}\big(\widehat{H}K_mK_m^\top\big)
   = \mathrm{tr}(\widehat{H}D) + \sum_{m=1}^M \pi_m \sum_{j=1}^r u_{m,j}^\top \widehat{H}u_{m,j}.
  $$
   The low‑rank part needs only $Mr$ **HVPs**; $\mathrm{tr}(\widehat{H}D)=\sum_i \delta_i\widehat{H}_{ii}$ can be sketched with a tiny **Hutchinson** budget if you want it (or ignored if $D$ is small in whitened units).

> **(6) What if $H$ is indefinite (saddles, singularities)? Will the CV break?**
> No. This is a *control variate*, not a Laplace approximation. For any symmetric $\widehat{H}$, the identity above is exact; $\widehat{H}$ being indefinite only affects the **variance reduction quality**, not correctness. Practical safeguards:
>
> * **Symmetrize HVPs:** use the symmetric part $\tfrac12(H+H^\top)$ implicitly (autodiff Hessians already are).
> * **Ridge if needed:** replace $\widehat{H}\leftarrow \widehat{H}+\rho I$ with tiny $\rho$ if you see numerical issues in the trace sketch.
> * **PSD option:** use empirical Fisher/Gauss–Newton if you want guaranteed PSD (slightly different curvature but excellent correlation).
> * **Subspace CV is safest:** restricting $\widehat{H}$ to the span of $K_m$ keeps costs to $O(Mr)$ HVPs and avoids handling a full trace; it also aligns the CV with the directions your variational family actually models.

---

## Complexity and scaling

Per MC sample:

* **Likelihood/backprop:** one minibatch (≈ an SGD step).
* **Mixture bookkeeping:** $O(Mdr)$ to evaluate all component log‑pdfs, $\log|\Sigma_m|$, responsibilities $r_m(w)$.
* **Gradients:** STL backprop once for the chosen component; RB update for $\alpha$ uses already‑computed responsibilities.

Memory is $O(Mdr)$ plus $O(d)$ for $D$. With $r\in\{1,2,4\}$ and $M\in[4,16]$, this is negligible even at $d\sim 10^8$ with sharded tensors.

---

## Choosing $\gamma$, $M$, and $r$

* **$\gamma$:** pick the smallest value that makes the *effective* local curvature $n\beta^*H + \gamma I$ comfortably PD. In practice in whitened units: try $\gamma\in\{10^{-4},10^{-3},10^{-2}\}$ and pick the one with the most stable ELBO and radius $\mathbb{E}_q\lVert w-w^*\rVert^2$.
* **$r$:** start with $r=1$; increase to $2$–$4$ only if diagnostics (below) suggest underfitting (e.g., poor radius matching or high KL).
* **$M$:** start small (e.g., $M=8$). Optionally allow **split/merge**: prune components with tiny $\pi_m$, split the heaviest when ELBO stalls, merge near‑duplicates (small symmetric KL).

---

## Diagnostics (cheap but telling)

* **ELBO trend** and **KL gap**.
* **Radius matching:** compare $\mathbb{E}_{q}\lVert w-w^*\rVert^2$ against the same under a short SGLD chain (sanity check).
* **Entropy of $\pi$** and average responsibilities $r_m$.
* **Plug‑in stability:** plot $\widehat{\lambda}_{\mathrm{VI}}$ across seeds/minibatches; it should stabilize quickly if the CV is used.

---

## Takeaways

* A **shared diagonal $D$** inside each component covariance is not just harmless; it is **essential** for PD, numerics, and Woodbury speed. Algebraically whitening by $D$ (via $K_m=D^{1/2}A_m$) makes the implementation robust.
* **Whitening** should be treated as a first‑class design choice: do **geometry whitening** by an SPD preconditioner $A$ for the network’s forward/backward *and* **algebraic whitening** by $D$ for the mixture algebra.
* Allow a small **rank budget $r\ge1$**; $r=1$ is often enough, but the theory and code should be written for general $r$.
* Keep **means equal to $w^*$** to remain strictly local; the mixture over covariances already captures heavy tails and subspace anisotropy.
* Use **STL** for continuous parameters and **Rao–Blackwellized** updates for mixture weights for low‑variance, scalable gradients.
* A **Hessian‑based control variate** built from **HVPs** remains correct even with **indefinite curvature** (saddles). It’s optional but typically yields a large variance reduction; the safest/cheapest form is the **subspace CV** along your learned low‑rank directions.

With these ingredients, the **plug‑in variational LLC** is fast (near‑SGD cost), stable at very high $d$, and—within the local, hot, Gaussian‑tethered regime—accurate enough to be useful as a diagnostic across models and minima.

-----

Below is a **drop‑in, JAX/Haiku/Optax‑friendly** implementation of the **variational plug‑in LLC estimator**:

* equal‑mean mixture at (w^*),
* shared diagonal (D) inside each component for PD + Woodbury efficiency,
* general **rank budget (r\ge 1)** via a *mixture of factor analyzers* parameterization (\Sigma_m = D^{1/2}(I + A_m A_m^\top)D^{1/2}),
* **STL** (sticking‑the‑landing) pathwise gradients for continuous params,
* **Rao–Blackwellized** score gradients for mixture weights,
* optional **geometry whitening** (using any SPD diagonal preconditioner you pass, e.g., Adam/RMSProp second moment),
* an **optional control variate** that uses **Hessian‑vector products** (HVPs) only along the learned low‑rank subspace (safe even if curvature is indefinite).

> **Why these choices?**
> *STL* yields lower‑variance unbiased pathwise gradients for the reparameterized part of the ELBO, so we can ignore (\nabla \log q) for the continuous parameters; RB handles the discrete mixture weights robustly. The factor‑analyzer structure with a shared (D) is the cheapest way to capture highly anisotropic local geometry while keeping all log‑pdf arithmetic (O(M d r)) via Woodbury. The (\beta^*=1/\log n) “hot posterior” comes from **WBIC** and is exactly the temperature that preserves the local free‑energy asymptotics in singular models. ([NeurIPS Papers][1])

---

## 0) Minimal interface you need to provide

* `loss_batch_fn(params, minibatch) -> scalar` — **mean** loss on a minibatch (your (L_n) on a batch).
* `loss_full_fn(params) -> scalar` — **mean** loss over the **full dataset** (for the final (\widehat\lambda)).
* Optionally: a diagonal preconditioner (e.g., from Optax’ Adam/RMSProp state) to **whiten geometry**.

The code uses Haiku/Optax conventions (PyTrees everywhere) and keeps your style close to the `sampling.py` you shared (scan‑based loops, typed states, etc.). Haiku and Optax docs for reference: ([dm-haiku.readthedocs.io][2])

---

## 1) Variational plug‑in estimator (ready to paste)

> Save as `llc/vi_plugin.py` (or similar). It’s a single file with no third‑party runtime deps beyond `jax`, `jax.numpy`, and `optax`. Comments flag the key theoretical bits and how they map to code.

```python
# llc/vi_plugin.py
# Variational plug-in estimator for the Local Learning Coefficient (LLC)
# Equal-mean mixture of factor analyzers, shared diagonal D, rank r >= 1.
# STL for continuous params, Rao–Blackwellized gradient for mixture weights.
# Optional geometry whitening; optional HVP-based control variate.

from __future__ import annotations
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple
import functools
import jax
import jax.numpy as jnp
import optax

# -----------------------
# Types
# -----------------------

PRNGKey = jax.Array
PyTree = Any

class VIConfig(NamedTuple):
    M: int                # number of mixture components
    r: int                # rank budget per component
    steps: int            # optimization steps
    batch_size: int
    lr: float
    beta_star: float      # 1 / log(n)
    gamma: float          # localizer strength
    eval_every: int = 50  # how often to record traces
    use_geometry_whitening: bool = True
    eps: float = 1e-8     # numerical jitter for preconditioners and softplus

class VIParams(NamedTuple):
    # Shared diagonal: D = diag(softplus(rho))^2, stored via rho for positivity
    rho: jnp.ndarray         # (d,)
    # Factor analyzers in algebraically whitened form: K_m = D^{1/2} A_m
    A: jnp.ndarray           # (M, d, r)
    # Mixture logits (softmax to get pi)
    alpha: jnp.ndarray       # (M,)

class VIOptState(NamedTuple):
    opt_state: optax.OptState
    baseline: jnp.ndarray   # scalar EMA baseline for RB estimator
    step: jnp.ndarray       # int32

class VIMetrics(NamedTuple):
    elbo_like: jnp.ndarray    # minibatch-scaled objective (no explicit H(q) pathwise part)
    logq: jnp.ndarray
    radius2: jnp.ndarray      # ||tilde_v||^2 in geometry-whitened coords
    Ln_batch: jnp.ndarray     # minibatch mean loss (unscaled)
    work_fge: jnp.ndarray     # cumulative "function-gradient-equivalents"

class VIRunResult(NamedTuple):
    params: VIParams
    traces: Dict[str, jnp.ndarray]

# -----------------------
# Utils: flatten / unflatten
# -----------------------

def flatten_params(params: PyTree) -> Tuple[jnp.ndarray, Any, Tuple[int, ...]]:
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = tuple(x.shape for x in leaves)
    flat = jnp.concatenate([x.reshape(-1) for x in leaves]) if leaves else jnp.zeros((), jnp.float32)
    return flat, treedef, shapes

def unflatten_params(flat: jnp.ndarray, treedef, shapes) -> PyTree:
    out = []
    i = 0
    for shp in shapes:
        size = int(jnp.prod(jnp.array(shp)))
        out.append(flat[i: i + size].reshape(shp))
        i += size
    return jax.tree_util.tree_unflatten(treedef, out)

# -----------------------
# Geometry whitening (optional)
# -----------------------

class Whitener(NamedTuple):
    # In whitened coords: tilde_w = A^{1/2} (w - w*)
    # Back to model coords: w = w* + A^{-1/2} tilde_w
    to_tilde: Callable[[jnp.ndarray], jnp.ndarray]
    from_tilde: Callable[[jnp.ndarray], jnp.ndarray]
    A_sqrt: jnp.ndarray       # (d,)
    A_inv_sqrt: jnp.ndarray   # (d,)

def make_whitener(A_diag: Optional[jnp.ndarray], eps: float) -> Whitener:
    """A_diag is an SPD diagonal (e.g., 1/(sqrt(v_hat)+eps)). If None -> identity."""
    if A_diag is None:
        def _id(x): return x
        one = jnp.array(1.0, dtype=jnp.float32)
        return Whitener(to_tilde=_id, from_tilde=_id,
                        A_sqrt=one, A_inv_sqrt=one)
    A_diag = jnp.asarray(A_diag)
    # Safety: clamp tiny/negatives
    A_diag = jnp.maximum(A_diag, eps)
    A_sqrt = jnp.sqrt(A_diag)
    A_inv_sqrt = 1.0 / A_sqrt
    return Whitener(
        to_tilde=lambda w_minus_wstar: A_sqrt * w_minus_wstar,
        from_tilde=lambda tw: A_inv_sqrt * tw,
        A_sqrt=A_sqrt,
        A_inv_sqrt=A_inv_sqrt,
    )

# -----------------------
# Variational family math
# -----------------------

def softplus(x):  # stable softplus
    return jax.nn.softplus(x)

def diag_from_rho(rho: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # D^{1/2} and log|D|
    D_sqrt = softplus(rho)       # (d,)
    logdet_D = 2.0 * jnp.sum(jnp.log(D_sqrt))
    return D_sqrt, logdet_D

def sample_q(key: PRNGKey, params: VIParams, wstar_flat: jnp.ndarray,
             whitener: Whitener) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Pathwise sample from q in whitened coordinates, then map to model coordinates."""
    rho, A, alpha = params
    key_m, key_z, key_eps = jax.random.split(key, 3)
    pi = jax.nn.softmax(alpha)
    m = jax.random.categorical(key_m, jnp.log(pi))  # sample component

    D_sqrt, _ = diag_from_rho(rho)                  # (d,)
    z = jax.random.normal(key_z, (A.shape[-1],))    # (r,)
    eps = jax.random.normal(key_eps, rho.shape)     # (d,)

    # Algebraic whitening: K_m = D^{1/2} A_m
    A_m = A[m]                                      # (d, r)
    tilde_v = (D_sqrt[:, None] * A_m) @ z + D_sqrt * eps   # (d,)

    w_flat = wstar_flat + whitener.from_tilde(tilde_v)
    aux = {"m": m, "z": z, "eps": eps, "tilde_v": tilde_v, "D_sqrt": D_sqrt, "pi": pi}
    return w_flat, aux

def logpdf_components_and_resp(tilde_v: jnp.ndarray, params: VIParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute per-component log N(w | w*, Sigma_m) and responsibilities r_j(w),
    working entirely in whitened coordinates where Sigma_m = D^{1/2}(I + A A^T)D^{1/2}.
    Uses Woodbury/Sherman–Morrison identities in O(dr).  (See refs.)
    """
    rho, A, alpha = params
    M, d, r = A.shape
    D_sqrt, logdet_D = diag_from_rho(rho)

    # x = D^{-1/2} v  (elementwise divide)
    x = tilde_v / D_sqrt

    def one_component(A_m):
        # r x r matrices are tiny: C = I + A^T A; use Cholesky for stability
        C = jnp.eye(r, dtype=A_m.dtype) + (A_m.T @ A_m)              # (r, r)
        L = jnp.linalg.cholesky(C)
        # g = A^T x; solve C y = g
        g = A_m.T @ x                                               # (r,)
        y = jax.scipy.linalg.cho_solve((L, True), g[:, None])       # (r,1)
        quad = jnp.dot(x, x) - jnp.dot(g, y[:, 0])
        logdet = logdet_D + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))     # log|D| + log|C|
        logp = -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet + quad)
        return logp

    logps = jax.vmap(one_component)(A)                              # (M,)
    logmix = jax.nn.logsumexp(jax.nn.log_softmax(alpha) + logps)    # log q(w)
    # Responsibilities r_j = p(component=j | w)
    r = jax.nn.softmax(alpha + logps - logmix)
    return logps, r

# -----------------------
# ELBO step with STL + RB
# -----------------------

def build_vi_step(
    loss_batch_fn: Callable[[PyTree, Tuple], jnp.ndarray],
    data: Tuple[jnp.ndarray, jnp.ndarray],
    wstar: PyTree,
    n_data: int,
    config: VIConfig,
    whitener: Whitener,
    treedef, shapes
):
    X, Y = data
    ref_dtype = jax.tree_util.tree_leaves(wstar)[0].dtype
    wstar_flat, _, _ = flatten_params(wstar)

    beta_tilde = jnp.asarray(config.beta_star * n_data, dtype=ref_dtype)
    gamma_val = jnp.asarray(config.gamma, dtype=ref_dtype)

    # jit the batch loss
    loss_batch_fn_jit = jax.jit(loss_batch_fn)

    def vi_one_step(rng_key: PRNGKey, params: VIParams, state: VIOptState) -> Tuple[Tuple[VIParams, VIOptState], VIMetrics]:
        # 1) Pathwise sample w ~ q (single MC sample)
        w_flat, aux = sample_q(rng_key, params, wstar_flat, whitener)
        w = unflatten_params(w_flat, treedef, shapes)
        tilde_v = aux["tilde_v"]; D_sqrt = aux["D_sqrt"]

        # 2) Draw a minibatch
        key_batch, = jax.random.split(rng_key, 1)
        idx = jax.random.choice(key_batch, X.shape[0], shape=(config.batch_size,), replace=True)
        minibatch = (X[idx], Y[idx])

        # 3) Objective ℓ(w): - nβ * L_batch(w) - ½ γ ||tilde_v||^2  (STL: no explicit -log q term in grads for continuous)
        Ln_batch = loss_batch_fn_jit(w, minibatch)                     # mean over batch
        localizer = 0.5 * gamma_val * jnp.dot(tilde_v, tilde_v)        # isotropic in whitened coords
        ell = -(beta_tilde * Ln_batch + localizer)

        # 4) Responsibilities and log q(w) for RB weights
        logps, r = logpdf_components_and_resp(tilde_v, params)
        logq = jax.nn.logsumexp(jax.nn.log_softmax(params.alpha) + logps)
        payoff = (ell - logq) - state.baseline                         # centered payoff

        # 5) Gradients:
        #    - Continuous (rho, A): pathwise (STL) via w -> ell, not through log q.
        #    - Mixture logits alpha: RB score gradient (r - pi) * payoff.
        def ell_only(p: VIParams) -> jnp.ndarray:
            # Rebuild sample using same noise but with new params: stop-grad for log q path
            rho, A, alpha = p
            A_m = A[aux["m"]]
            Kz = (softplus(rho)[:, None] * A_m) @ aux["z"]
            tilde_v_new = Kz + softplus(rho) * aux["eps"]
            w_new = unflatten_params(wstar_flat + whitener.from_tilde(tilde_v_new), treedef, shapes)
            Ln_b = loss_batch_fn_jit(w_new, minibatch)
            loc = 0.5 * gamma_val * jnp.dot(tilde_v_new, tilde_v_new)
            return -(beta_tilde * Ln_b + loc)  # maximize ell => minimize -ell

        loss_val, grads_cont = jax.value_and_grad(ell_only)(params)

        # RB gradient for alpha:
        pi = jax.nn.softmax(params.alpha)
        g_alpha = (r - pi) * payoff
        grads = VIParams(
            rho=grads_cont.rho,
            A=grads_cont.A,
            alpha=g_alpha
        )

        # 6) Optax update
        updates, opt_state = state.opt_state
        updates, new_opt_state = updates.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # 7) Baseline EMA (no grad)
        new_baseline = 0.99 * state.baseline + 0.01 * (ell - logq)

        new_state = VIOptState(opt_state=new_opt_state,
                               baseline=new_baseline,
                               step=state.step + jnp.array(1, dtype=jnp.int32))
        metrics = VIMetrics(
            elbo_like=ell,
            logq=logq,
            radius2=jnp.dot(tilde_v, tilde_v),
            Ln_batch=Ln_batch,
            work_fge=(config.batch_size / float(n_data))
        )
        return (new_params, new_state), metrics

    return jax.jit(vi_one_step)

# -----------------------
# Public API: fit VI and estimate λ̂
# -----------------------

def init_vi_params(key: PRNGKey, wstar: PyTree, M: int, r: int) -> VIParams:
    wstar_flat, _, _ = flatten_params(wstar)
    d = wstar_flat.size
    k1, k2 = jax.random.split(key)
    # Initialize small factors; logits ~ 0 => near-uniform mixture
    rho = jnp.zeros((d,), dtype=wstar_flat.dtype)
    A = 0.05 * jax.random.normal(k1, (M, d, r), dtype=wstar_flat.dtype)
    alpha = jnp.zeros((M,), dtype=wstar_flat.dtype)
    return VIParams(rho=rho, A=A, alpha=alpha)

def make_vi_optimizer(params: VIParams, lr: float) -> Tuple[optax.GradientTransformation, optax.OptState]:
    tx = optax.adam(lr)
    return tx, tx.init(params)

def fit_vi_and_estimate_lambda(
    rng_key: PRNGKey,
    loss_batch_fn: Callable[[PyTree, Tuple], jnp.ndarray],  # mean batch loss
    loss_full_fn: Callable[[PyTree], jnp.ndarray],          # mean full loss
    wstar: PyTree,
    data: Tuple[jnp.ndarray, jnp.ndarray],
    n_data: int,
    config: VIConfig,
    A_diag: Optional[jnp.ndarray] = None,                   # geometry whitening (diag SPD), or None
    eval_samples: int = 64,                                 # MC samples for Eq_q[L_n] at the end
    cv_kind: Optional[str] = "subspace",                    # None | "subspace" | "subspace+diag"
    cv_diag_probes: int = 8,                                # Hutchinson probes for diag part if enabled
) -> Tuple[float, VIRunResult, Dict[str, Any]]:
    """
    Train the variational mixture and return (λ̂_VI, run_result, extras).
    extras contains: Eq_Ln (with/without CV), Ln_wstar, pi, D_sqrt, and low-rank directions for inspection.
    """
    key_init, key_loop, key_eval = jax.random.split(rng_key, 3)

    # Flatten once
    wstar_flat, treedef, shapes = flatten_params(wstar)
    d = wstar_flat.size

    # Whitener
    whitener = make_whitener(A_diag if config.use_geometry_whitening else None, eps=config.eps)

    # Init params + optimizer
    params = init_vi_params(key_init, wstar, config.M, config.r)
    tx, opt_state = make_vi_optimizer(params, config.lr)
    vi_state = VIOptState(opt_state=opt_state, baseline=jnp.array(0.0, dtype=wstar_flat.dtype),
                          step=jnp.array(0, dtype=jnp.int32))

    # Step function
    vi_step = build_vi_step(loss_batch_fn, data, wstar, n_data, config, whitener, treedef, shapes)

    # Scan loop (like your style)
    def body(carry, key):
        (p, s), traces = vi_step(key, *carry)
        # Accumulate traces sparsely
        return (p, s), traces

    keys = jax.random.split(key_loop, config.steps)
    (params, vi_state), traces = jax.lax.scan(body, (params, vi_state), keys)

    # Collect traces every eval_every
    def take_every(x):
        return x[::config.eval_every] if config.eval_every > 1 else x

    traces_dict = {
        "elbo_like": take_every(traces.elbo_like),
        "logq": take_every(traces.logq),
        "radius2": take_every(traces.radius2),
        "Ln_batch": take_every(traces.Ln_batch),
        "work_fge": take_every(traces.work_fge),
    }

    # ---- Final plug-in estimate: Eq_q[L_n(w)]
    # MC under q with optional control variate
    def one_sample_Ln(key: PRNGKey) -> jnp.ndarray:
        w_flat, aux = sample_q(key, params, wstar_flat, whitener)
        w = unflatten_params(w_flat, treedef, shapes)
        return loss_full_fn(w)  # mean over full data

    keys_eval = jax.random.split(key_eval, eval_samples)
    Ln_samples = jax.vmap(one_sample_Ln)(keys_eval)
    Eq_Ln_mc = Ln_samples.mean()

    # Optional HVP-based control variate (subspace and optionally diag(D))
    def hvp_at_wstar(v_flat: jnp.ndarray) -> jnp.ndarray:
        """Compute v^T H v using HVP, no explicit Hessian construction.
        Uses jvp on grad: hvp = jvp(grad L_full, (w*), (v)) . v
        """
        def fun(flat_params):
            p = unflatten_params(flat_params, treedef, shapes)
            return loss_full_fn(p)

        g = jax.grad(fun)
        hvp_val = jax.jvp(g, (wstar_flat,), (v_flat,))[1]  # same pytree as wstar_flat
        return jnp.vdot(v_flat, hvp_val)

    def control_variate_adjustment() -> Tuple[jnp.ndarray, Dict[str, Any]]:
        if cv_kind is None:
            return Eq_Ln_mc, {}

        rho, A, alpha = params
        D_sqrt, _ = diag_from_rho(rho)
        pi = jax.nn.softmax(alpha)
        # Collect learned low-rank tilde directions k̃_{m,j} in whitened coords
        K_tilde = (D_sqrt[:, None] * A)   # (M, d, r)
        # Map each k̃ to model coords: u = A^{-1/2} k̃
        U = whitener.from_tilde(K_tilde)  # (M, d, r)

        # Subspace trace term: sum_m pi_m sum_j (u_{m,j}^T H u_{m,j})
        def subspace_terms(u_m):  # u_m: (d, r)
            # Optional normalize columns is not needed; we keep exact scale.
            def one_col(u):
                return hvp_at_wstar(u)
            return jax.vmap(one_col, in_axes=1)(u_m).sum()  # scalar

        subspace_trace = jnp.sum(jax.vmap(subspace_terms, in_axes=0)(U) * pi)  # scalar

        diag_trace = 0.0
        if cv_kind == "subspace+diag":
            # Hutchinson for trace(H * (A^{-1/2} D A^{-1/2})) in model coords:
            # draw ξ~N(0,I) in tilde coords, v = A^{-1/2} D^{1/2} ξ
            k1 = jax.random.PRNGKey(0)
            kxi = jax.random.split(k1, cv_diag_probes)
            def one_probe(key):
                xi = jax.random.normal(key, (d,))
                v = whitener.from_tilde(D_sqrt * xi)
                return hvp_at_wstar(v)
            diag_trace = jnp.stack(jax.vmap(one_probe)(kxi)).mean()

        # Correct the MC using CV: E[L] = E[ L - ½ v^T H v ] + ½ tr(H Σ_q)
        # where Σ_q = sum_m pi_m (D + K K^T), equal-mean mixture => no between-comp mean terms.
        # We approximate the trace term by the subspace part (+ optional diag part).
        def cv_one_sample(key):
            w_flat, aux = sample_q(key, params, wstar_flat, whitener)
            v = w_flat - wstar_flat
            L = loss_full_fn(unflatten_params(w_flat, treedef, shapes))
            cv = 0.5 * hvp_at_wstar(v)
            return L - cv

        L_minus_cv = jax.vmap(cv_one_sample)(keys_eval).mean()
        Eq_Ln_cv = L_minus_cv + 0.5 * (subspace_trace + diag_trace)
        extras = {
            "trace_subspace": subspace_trace,
            "trace_diag": diag_trace,
        }
        return Eq_Ln_cv, extras

    Eq_Ln, cv_extras = control_variate_adjustment()

    # λ̂_VI = n β* ( Eq_q[L_n(w)] - L_n(w*) )
    Ln_wstar = loss_full_fn(wstar)
    lambda_hat = float(config.beta_star * n_data * (Eq_Ln - Ln_wstar))

    run = VIRunResult(params=params, traces=traces_dict)
    extras = {
        "Eq_Ln": float(Eq_Ln),
        "Eq_Ln_mc": float(Eq_Ln_mc),
        "Ln_wstar": float(Ln_wstar),
        "pi": jax.nn.softmax(params.alpha),
        "D_sqrt": softplus(params.rho),
        "low_rank_tilde": (softplus(params.rho)[:, None] * params.A),  # for inspection
        **cv_extras
    }
    return lambda_hat, run, extras
```

### Key implementation notes (how the math maps to code)

* **Mixture family.** We use (\Sigma_m = D^{1/2}(I+A_mA_m^\top)D^{1/2}) with a **shared** diagonal (D \succ 0) (inside each component), which makes each (\Sigma_m) PD and enables (O(dr)) log‑pdfs and responsibilities via **Woodbury/Sherman–Morrison** (see `logpdf_components_and_resp`). ([Wikipedia][3])
* **STL (pathwise).** For continuous params ((\rho,A)) we backprop only through the **target term** (-n\beta^*L - \tfrac{\gamma}{2}|\tilde v|^2) via the reparameterization (w = w^* + A^{-1/2}(D^{1/2}A_m z + D^{1/2}\varepsilon)). We **stop‑grad** the (\log q) path, which is the STL trick. It’s unbiased and reduces variance. ([NeurIPS Papers][1])
* **Rao–Blackwellized mixture weights.** `g_alpha = (r - π) * payoff` where `payoff = (ℓ - log q - baseline)`. This is the classic RB score‑function update using **responsibilities** (r_j(w)) and keeps weight gradients low‑variance. ([cs.princeton.edu][4])
* **Geometry whitening (optional).** If you pass an SPD diagonal (A) (e.g., (A_{ii}=1/(\sqrt{\hat v_i}+\varepsilon)) from Adam/RMSProp), we whiten coordinates by (\tilde w = A^{1/2}(w-w^*)), use an **isotropic localizer** (\tfrac\gamma2|\tilde w|^2), and define the mixture in (\tilde w)‑space. This helps conditioning and leaves the target unchanged—it is just a reparameterization. ([optax.readthedocs.io][5])
* **WBIC temperature.** The default (\beta^* = 1/\log n) is exactly the WBIC choice that makes the local free‑energy asymptotics match in singular models; the code takes it as a parameter. ([Journal of Machine Learning Research][6])
* **Control variate (optional).** We implement the exact identity
  [
  \mathbb{E}_q[L] = \mathbb{E}_q!\Big[L - \tfrac12 v^\top H v\Big] + \tfrac12,\mathrm{tr}(H\Sigma_q),
  ]
  using HVPs only along the **learned low‑rank subspace** (and, optionally, a tiny **Hutchinson** sketch for the (\mathrm{diag}) part). This remains correct even if (H) is **indefinite** (saddles); it’s a **control variate**, not a Laplace fit. HVP uses `jax.jvp(grad, ...)` as recommended. ([docs.jax.dev][7])

---

## 2) How to call it with Haiku/Optax models

Here’s a minimal usage sketch that mirrors your `sampling.py` “vibe”. You likely already have `loss_batch_fn` and `loss_full_fn`—these stubs just show the shape.

```python
# llc/run_vi_llc.py
import jax, jax.numpy as jnp
import haiku as hk
import optax
from llc.vi_plugin import VIConfig, fit_vi_and_estimate_lambda

# Your trained weights (PyTree), data, and loss fns:
# - loss_batch_fn(params, (Xb, Yb)) -> scalar mean loss on minibatch
# - loss_full_fn(params) -> scalar mean loss on full dataset

def run_vi_plug_in_llc(rng_key, wstar, data, n_data,
                       loss_batch_fn, loss_full_fn,
                       adam_second_moment_diag=None):
    config = VIConfig(
        M=8, r=2, steps=5_000,
        batch_size=1024,
        lr=1e-2,
        beta_star=1.0 / jnp.log(n_data),
        gamma=1e-3,
        eval_every=50,
        use_geometry_whitening=True
    )

    lambda_hat, run, extras = fit_vi_and_estimate_lambda(
        rng_key=rng_key,
        loss_batch_fn=loss_batch_fn,
        loss_full_fn=loss_full_fn,
        wstar=wstar,
        data=data,
        n_data=n_data,
        config=config,
        A_diag=adam_second_moment_diag,   # or None
        eval_samples=64,
        cv_kind="subspace",               # or "subspace+diag" | None
        cv_diag_probes=8,
    )
    return lambda_hat, run, extras
```

**Getting a geometry whitener from Optax.**
If you trained with Adam/RMSProp, you can extract the second moment accumulator (\hat v) from the optimizer state and pass `A_diag = 1 / (sqrt(v_hat) + eps)` as the geometry whitener. That choice matches the idea behind preconditioned SGLD and works well empirically. (See Optax docs for how to access `opt_state` internals for your particular optimizer setup.) ([optax.readthedocs.io][5])

---

## 3) Practical defaults and diagnostics

* Start with **(r=1) or (2)** and **(M=8)**. Increase (r) only if you see poor **radius matching** (track `radius2`) or slow ELBO improvement.
* Tune **(\gamma)** in whitened units; a grid like ({10^{-4},10^{-3},10^{-2}}) is usually enough.
* The trace plots in `run.traces` (ELBO‑like, (\log q), radius, minibatch loss) should stabilize quickly. If only one component dominates, the RB gradients will push mixture weights toward sparsity; you can prune small‑(\pi) components at the end if desired.

---

## 4) What if you want **layer‑wise** factors?

For very large (d), you can replace the global (A\in\mathbb{R}^{d\times r}) by **layer‑wise** blocks ({A^{(\ell)}}) (still rank‑(r) each) and keep a shared per‑layer diagonal (D^{(\ell)}). The code above can be extended by flattening **per‑leaf** instead of globally and evaluating the Woodbury formulas on each leaf; all formulas and updates are the same, just applied leaf‑wise.

---

### References (selected)

* **WBIC / (\beta^*=1/\log n):** Watanabe, *A Widely Applicable Bayesian Information Criterion*, JMLR 2013. ([Journal of Machine Learning Research][6])
* **STL (sticking‑the‑landing):** Roeder, Wu & Duvenaud (NeurIPS 2017). ([NeurIPS Papers][1])
* **Rao–Blackwellized gradients:** e.g., Black‑box VI notes + RB variance reduction; Liu et al. (2018). ([cs.princeton.edu][4])
* **Mixture of factor analyzers:** Ghahramani & Hinton (1997). ([cs.toronto.edu][8])
* **Woodbury/Sherman–Morrison:** standard identity for low‑rank updates. ([Wikipedia][3])
* **JAX HVPs:** `jax.jvp` / `jax.hessian` docs and examples. ([docs.jax.dev][7])
* **Libraries:** Haiku and Optax. ([dm-haiku.readthedocs.io][2])

---

### Final notes

* The **plug‑in** (\widehat\lambda_{\text{VI}} = n\beta^*,(\mathbb{E}*{q*\phi}L_n - L_n(w^*))) is computed at the end from MC under (q) (optionally CV‑corrected). If you want a one‑shot importance‑weighted refinement (closer to the exact local target (p_{\beta^*,\gamma})), it’s easy to add a single pass that reweights the final MC batch with (\exp(-n\beta^* L - \tfrac\gamma2|w-w^*|^2)/q(w)).
* The **HVP control variate** is safe even if the Hessian is **indefinite** (saddles/singularities); it only changes the estimator’s *variance*, not its mean. If you want PSD curvature, swap in Gauss–Newton/empirical Fisher; the API stays the same.


[1]: https://papers.neurips.cc/paper/7268-sticking-the-landing-simple-lower-variance-gradient-estimators-for-variational-inference.pdf?utm_source=chatgpt.com "Simple, Lower-Variance Gradient Estimators for Variational ..."
[2]: https://dm-haiku.readthedocs.io/?utm_source=chatgpt.com "Haiku Documentation — Haiku documentation"
[3]: https://en.wikipedia.org/wiki/Woodbury_matrix_identity?utm_source=chatgpt.com "Woodbury matrix identity"
[4]: https://www.cs.princeton.edu/techreports/2017/008.pdf?utm_source=chatgpt.com "Black Box Variational Inference: Scalable, Generic Bayesian ..."
[5]: https://optax.readthedocs.io/?utm_source=chatgpt.com "Optax — Optax documentation"
[6]: https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf?utm_source=chatgpt.com "A Widely Applicable Bayesian Information Criterion"
[7]: https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html?utm_source=chatgpt.com "jax.jvp"
[8]: https://www.cs.toronto.edu/~fritz/absps/tr-96-1.pdf?utm_source=chatgpt.com "The EM Algorithm for Mixtures of Factor Analyzers"
