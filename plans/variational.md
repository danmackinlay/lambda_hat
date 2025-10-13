Below is a **self‑contained introduction** to a **variational approximation of the (local, tempered) NN posterior** for a **plug‑in Local Learning Coefficient (LLC)** estimate. It folds in the six points you flagged: shared‑(D) design, whitening (two flavors), general low rank (r\ge 1), equal means at (w^*), STL + Rao–Blackwellization, and a careful treatment of the Hessian‑based control variate when curvature can be indefinite.

---

## Goal and setup

We want (\lambda(w^*)), the Local Learning Coefficient at a trained solution (w^*). In the “WBIC‑style” formulation, pick the **hot** inverse temperature
[
\beta^*=\frac{1}{\log n},
]
and define the **local, tempered target** over weights (w\in\mathbb{R}^d)
[
p_{\beta,\gamma}(w\mid w^*) \ \propto\ \exp!\Big(-n\beta,L_n(w)-\tfrac{\gamma}{2}|w-w^*|^2\Big),
]
with (\gamma>0) a *localizer* strength. Then the estimand is a **single local expectation**:
[
\widehat{\lambda}(w^*);=;n\beta^*,\Big(\mathbb{E}*{p*{\beta^*,\gamma}}[L_n(w)]-L_n(w^*)\Big).
]

Instead of drawing exact samples (e.g., with HMC/SGLD), we will **fit a local variational family (q_\phi)** to (p_{\beta^*,\gamma}) and use the **plug‑in estimator**
[
\widehat{\lambda}_{\mathrm{VI}}(w^*)
;=; n\beta^*\Big(\mathbb{E}*{q*\phi}[L_n(w)]-L_n(w^*)\Big).
]

---

## Variational family: equal‑mean mixture of factor analyzers

We keep **all component means equal to (w^*)** (this preserves locality and keeps moments simple) and let each component carry a **low‑rank plus diagonal** covariance:

[
q_\phi(w) ;=; \sum_{m=1}^M \pi_m;\mathcal{N}!\big(w^*,,\Sigma_m\big),\quad
\Sigma_m ;=; D;+;K_m K_m^\top,
]
where

* (M\ll d) is small,
* (D=\mathrm{diag}(\delta_1,\ldots,\delta_d)\succ 0) is a **shared diagonal** across all components,
* (K_m\in\mathbb{R}^{d\times r}) with **rank budget (r\ge 1)** but (r\ll d),
* (\pi=\mathrm{softmax}(\alpha)) are the mixture weights.

**Why this shape?**

* **Expressive at low cost:** The shared diagonal captures the ubiquitous “noise floor”; each low‑rank (K_m) adds a component‑specific subspace for very flat/stiff directions.
* **Well‑conditioned and PD:** With (D\succ0), every (\Sigma_m) is positive‑definite even when (r\ll d).
* **Efficient algebra:** Woodbury/Sherman–Morrison reduces all log‑pdfs and responsibilities to (O(dr)) per component.

> **(1) Should we include a shared (D) at all?**
> Yes. If you drop (D) and keep (r<d), (\Sigma_m) is singular, (q) is improper, and Woodbury identities no longer apply cleanly. Keeping (D) **inside** each (\Sigma_m) enables:
> (i) guaranteed PD, (ii) cached (D^{-1}) and (\log|D|) shared across components, (iii) a natural **whitened parameterization** (K_m=D^{1/2}A_m) (below). When we later “whiten,” (D) effectively becomes the identity, so you still get the algebraic simplicity you want.

---

## The ELBO we optimize (local and tempered)

We fit (q_\phi) by maximizing the **local ELBO**
[
\mathcal{L}(\phi)
;=; \mathbb{E}*{q*\phi}!\left[-,n\beta^*,L_n(w);-;\tfrac{\gamma}{2},|w-w^*|^2\right];+;\mathsf{H}(q_\phi).
]

* The expectation is computed with **minibatches** (scale by (N/B)).
* The entropy (\mathsf{H}(q_\phi)) and (\log q_\phi) require evaluating all mixture component log‑pdfs for a sampled (w); with Woodbury this costs (O(Mdr)).

**Pathwise gradients (STL):** Reparameterize a sample by
[
\varepsilon\sim\mathcal{N}(0,I_d),\ z\sim\mathcal{N}(0,I_r),\ m\sim\mathrm{Cat}(\pi),\quad
w ;=; w^* + K_m z + D^{1/2}\varepsilon.
]
For the **continuous** parameters ((D,K_m)) we use **sticking‑the‑landing** (STL): backprop only through the target term (-n\beta^*L_n(w)-\tfrac{\gamma}{2}|w-w^*|^2) via (w), treating (\log q) as a stop‑gradient. This is unbiased and substantially lowers variance.

**Mixture weights (Rao–Blackwellized):** For logits (\alpha) of (\pi), use the **RB score estimator**
[
g_{\alpha_j} ;=; \left(\underbrace{-n\beta^*L_n(w)-\tfrac{\gamma}{2}|w-w^*|^2 - \log q_\phi(w) - b}_{\text{centered payoff}}\right); \big(r_j(w)-\pi_j\big),
]
where (r_j(w)) are responsibilities and (b) is a moving‑average baseline.

---

## Whitening, properly set up (two complementary flavors)

Whitening matters both for **numerical conditioning** and for **algebraic efficiency**. There are two distinct but compatible uses of “whitening” here.

### A. **Algebraic whitening by the shared (D)** (inside the variational family)

Parameterize
[
K_m ;=; D^{1/2}A_m \quad \Rightarrow \quad
\Sigma_m ;=; D^{1/2},(I + A_m A_m^\top),D^{1/2}.
]
In these coordinates:

* (\log|\Sigma_m|=\log|D|+\log|I + A_m A_m^\top|),
* (\Sigma_m^{-1}=D^{-1/2}\Big(I - A_m (I + A_m^\top A_m)^{-1}A_m^\top\Big)D^{-1/2}).

This parameterization **stabilizes** learning (the low‑rank factors are measured in “units of (D)”), keeps **all Woodbury inverses small** ((r\times r)), and avoids repeated multiplications by poorly scaled (K_m).

### B. **Geometry whitening by a preconditioner (A)** (outside the variational family)

Independently of (D), we can precondition *the parameter space* with a fixed SPD matrix (A) that approximates local curvature (e.g., Adam’s running second moment, RMSProp, a diagonal empirical Fisher, or a block‑diagonal Kronecker factor).

Let
[
\tilde{w};=;A^{1/2}(w-w^*),\qquad w;=;w^*+A^{-1/2}\tilde{w}.
]
Under this change of variables, the **target** becomes
[
\tilde{p}(\tilde{w})\ \propto\
\exp!\Big(-,n\beta^*,L_n\big(w^*+A^{-1/2}\tilde{w}\big);-;\tfrac{\gamma}{2},\tilde{w}^\top A^{-1}\tilde{w}\Big),
]
and the Jacobian (|\det A^{-1/2}|) is a constant that cancels in the ELBO up to an additive constant. You then **define and train (q)** in (\tilde{w})-space (where directions are closer to isotropic), while your model only ever sees (w=w^*+A^{-1/2}\tilde{w}).

> *Do we “break” the spherical localizer by geometry whitening?*
> In (\tilde{w})-space the localizer is elliptical with metric (A^{-1}). That’s fine because we are **not changing the target**, just its coordinates; all gradients are computed through the exact map (w(\tilde{w})). If you *prefer* keeping an isotropic tether in whitened space, you may instead define the localizer as (\frac{\gamma}{2}|A^{1/2}(w-w^*)|^2) from the outset; that simply replaces (A^{-1}) by (I) above without changing the spirit of the estimator.

**Practical recipe:**
Use geometry whitening for the network **forward/backward** (helps the (\nabla L_n) numerics), and algebraic whitening by (D) **inside (q)** (makes the mixture algebra fast and stable). The two are independent and complementary.

---

## Working formulas (for general rank (r\ge 1))

For any component (m) and (v=w-w^*), define
[
B_m := D^{-1}K_m \ \in \mathbb{R}^{d\times r},\qquad
A_m^\star := I_r + K_m^\top D^{-1}K_m = I_r + K_m^\top B_m.
]
Then
[
\Sigma_m^{-1} = D^{-1} - B_m (A_m^\star)^{-1} B_m^\top,\quad
\log|\Sigma_m| = \log|D| + \log|A_m^\star|,
]
[
Q_m(v) = v^\top \Sigma_m^{-1} v
= v^\top D^{-1}v - v^\top D^{-1}K_m (A_m^\star)^{-1} K_m^\top D^{-1} v.
]
The log‑pdf and responsibilities use these (O(dr)) primitives.

**Sampling (pathwise):**
[
z\sim\mathcal{N}(0,I_r),\quad \varepsilon\sim\mathcal{N}(0,I_d),\quad
w ;=; w^* + K_m z + D^{1/2} \varepsilon.
]
This makes the **continuous** gradients especially simple with STL:
[
\frac{\partial \mathcal{L}}{\partial K_m}\ \approx\
\mathbb{E}!\big[\nabla_w \big(-n\beta^*L_n(w)-\tfrac{\gamma}{2}|w-w^*|^2\big); z^\top\big],
\quad
\frac{\partial \mathcal{L}}{\partial D^{1/2}}\ \approx\
\mathbb{E}!\big[\nabla_w (\cdot); \varepsilon^\top\big].
]

---

## What bias does the plug‑in estimator incur?

Let (U(w)=n\beta^*L_n(w)+\frac{\gamma}{2}|w-w^*|^2), (p\propto e^{-U}). For any (q),
[
n\beta^* \big(\mathbb{E}*{q}[L_n]-\mathbb{E}*{p}[L_n]\big)
+\frac{\gamma}{2}\Big(\mathbb{E}*{q}|w-w^*|^2-\mathbb{E}*{p}|w-w^*|^2\Big)
= \mathrm{KL}(q|p)+\mathsf{H}(q)-\mathsf{H}(p).
]
So if (i) (q) is close to (p) (small KL) and (ii) it matches the **radius** (\mathbb{E}|w-w^*|^2), the plug‑in bias is small.
In the **quadratic** neighborhood (local Laplace regime), take (q=p) with (\Sigma_p=(n\beta^*H+\gamma I)^{-1}) and the estimate is **exact**.

---

## Cheap control variate with Hessian–vector products (HVPs)

We often reduce Monte‑Carlo variance by subtracting a quadratic **control variate** and adding back its exact expectation under (q).

1. Choose a symmetric matrix (\widehat{H}) built from **HVPs** (no explicit matrices). Natural choices:

* **Subspace CV:** (\widehat{H}=\sum_{m=1}^M \sum_{j=1}^r \lambda_{m,j},u_{m,j}u_{m,j}^\top), where (K_m=[u_{m,1}\dots u_{m,r}]) and (\lambda_{m,j}=u_{m,j}^\top (\nabla^2 L_n) u_{m,j}) (just (Mr) HVPs).
* **PSD option:** a Gauss–Newton / empirical Fisher approximation (always PSD).
* **Hutchinson top‑up:** a few Rademacher probes to sketch (\mathrm{diag}(H)) if you want the (D)–trace term (see below).

2. Define
   [
   \widetilde{L}(w) ;=; L_n(w) - \tfrac12 (w-w^*)^\top \widehat{H},(w-w^*).
   ]
   Then for **equal‑mean** mixtures (our case), the mixture covariance is
   [
   \mathrm{Cov}*q(w) = \Sigma_q = \sum*{m=1}^M \pi_m,\Sigma_m,
   ]
   and
   [
   \mathbb{E}*{q}[L_n(w)] ;=; \mathbb{E}*{q}[\widetilde{L}(w)] ;+; \tfrac12,\mathrm{tr}\big(\widehat{H},\Sigma_q\big).
   ]

* The first term is estimated by MC with **much lower variance** (no curvature term left).
* The second term is **analytic** given (\Sigma_q). With (\Sigma_m = D + K_mK_m^\top),
  [
  \mathrm{tr}(\widehat{H},\Sigma_q) ;=; \mathrm{tr}(\widehat{H}D) ;+; \sum_{m=1}^M \pi_m,\mathrm{tr}\big(\widehat{H}K_mK_m^\top\big)
  ;=; \mathrm{tr}(\widehat{H}D);+; \sum_{m=1}^M \pi_m \sum_{j=1}^r u_{m,j}^\top \widehat{H},u_{m,j}.
  ]
  The low‑rank part needs only (Mr) **HVPs**; (\mathrm{tr}(\widehat{H}D)=\sum_i \delta_i,\widehat{H}_{ii}) can be sketched with a tiny **Hutchinson** budget if you want it (or ignored if (D) is small in whitened units).

> **(6) What if (H) is indefinite (saddles, singularities)? Will the CV break?**
> No. This is a *control variate*, not a Laplace approximation. For any symmetric (\widehat{H}), the identity above is exact; (\widehat{H}) being indefinite only affects the **variance reduction quality**, not correctness. Practical safeguards:
>
> * **Symmetrize HVPs:** use the symmetric part (\tfrac12(H+H^\top)) implicitly (autodiff Hessians already are).
> * **Ridge if needed:** replace (\widehat{H}\leftarrow \widehat{H}+\rho I) with tiny (\rho) if you see numerical issues in the trace sketch.
> * **PSD option:** use empirical Fisher/Gauss–Newton if you want guaranteed PSD (slightly different curvature but excellent correlation).
> * **Subspace CV is safest:** restricting (\widehat{H}) to the span of (K_m) keeps costs to (O(Mr)) HVPs and avoids handling a full trace; it also aligns the CV with the directions your variational family actually models.

---

## Complexity and scaling

Per MC sample:

* **Likelihood/backprop:** one minibatch (≈ an SGD step).
* **Mixture bookkeeping:** (O(Mdr)) to evaluate all component log‑pdfs, (\log|\Sigma_m|), responsibilities (r_m(w)).
* **Gradients:** STL backprop once for the chosen component; RB update for (\alpha) uses already‑computed responsibilities.

Memory is (O(Mdr)) plus (O(d)) for (D). With (r\in{1,2,4}) and (M\in[4,16]), this is negligible even at (d\sim 10^8) with sharded tensors.

---

## Choosing (\gamma), (M), and (r)

* **(\gamma):** pick the smallest value that makes the *effective* local curvature (n\beta^*H + \gamma I) comfortably PD. In practice in whitened units: try (\gamma\in{10^{-4},10^{-3},10^{-2}}) and pick the one with the most stable ELBO and radius (\mathbb{E}_q|w-w^*|^2).
* **(r):** start with (r=1); increase to (2)–(4) only if diagnostics (below) suggest underfitting (e.g., poor radius matching or high KL).
* **(M):** start small (e.g., (M=8)). Optionally allow **split/merge**: prune components with tiny (\pi_m), split the heaviest when ELBO stalls, merge near‑duplicates (small symmetric KL).

---

## Diagnostics (cheap but telling)

* **ELBO trend** and **KL gap**.
* **Radius matching:** compare (\mathbb{E}_{q}|w-w^*|^2) against the same under a short SGLD chain (sanity check).
* **Entropy of (\pi)** and average responsibilities (r_m).
* **Plug‑in stability:** plot (\widehat{\lambda}_{\mathrm{VI}}) across seeds/minibatches; it should stabilize quickly if the CV is used.

---

## Takeaways

* A **shared diagonal (D)** inside each component covariance is not just harmless; it is **essential** for PD, numerics, and Woodbury speed. Algebraically whitening by (D) (via (K_m=D^{1/2}A_m)) makes the implementation robust.
* **Whitening** should be treated as a first‑class design choice: do **geometry whitening** by an SPD preconditioner (A) for the network’s forward/backward *and* **algebraic whitening** by (D) for the mixture algebra.
* Allow a small **rank budget (r\ge1)**; (r=1) is often enough, but the theory and code should be written for general (r).
* Keep **means equal to (w^*)** to remain strictly local; the mixture over covariances already captures heavy tails and subspace anisotropy.
* Use **STL** for continuous parameters and **Rao–Blackwellized** updates for mixture weights for low‑variance, scalable gradients.
* A **Hessian‑based control variate** built from **HVPs** remains correct even with **indefinite curvature** (saddles). It’s optional but typically yields a large variance reduction; the safest/cheapest form is the **subspace CV** along your learned low‑rank directions.

With these ingredients, the **plug‑in variational LLC** is fast (near‑SGD cost), stable at very high (d), and—within the local, hot, Gaussian‑tethered regime—accurate enough to be useful as a diagnostic across models and minima.
