
## 0) Core modeling idea (low‑dimensional flow + thin ambient noise)

We want a sampler whose **pushforward mass stays near a (d_{\mathrm{latent}})-dimensional manifold** around (w^*) (with (d_{\mathrm{latent}}\ll d)). The most efficient way to both keep **tractable densities** (so we can optimize the reverse KL / ELBO) and allow **(d_{\mathrm{latent}}<d)** is to use an **injective (across‑dimension) flow** in the tangent directions and add **thin, easy noise** in the orthogonal directions.

**Construction.**

1. Choose an orthonormal basis (\mm{U}\in\mathbb{R}^{d\times d_{\mathrm{latent}}}) spanning the locally “flat” directions near (w^*) (how to get (\mm{U}): §2 below).
2. Let (\mm{E}*\perp\in\mathbb{R}^{d\times(d-d*{\mathrm{latent}})}) be any orthonormal complement so ([\mm{U},\mm{E}_\perp]) is orthogonal.
3. Put a **small SPD preconditioner** (\mm{D}\succ0) (diagonal or structured) to whiten/scalewise balance the posterior locally.
4. Define a **latent flow** (T_\phi:\mathbb{R}^{d_{\mathrm{latent}}}!\to!\mathbb{R}^{d_{\mathrm{latent}}}) (a standard 1D‑wise monotone triangular flow such as neural rational‑quadratic splines; see §3.2), and **orthogonal noise** with scale (\sigma_\perp) (fixed or learned, with a floor).

**Sampling.**
Draw (\rv{z}\sim\mathcal{N}(0,\mm{I}*{d*{\mathrm{latent}}})), (\rv{\varepsilon}*\perp\sim\mathcal{N}(0,\mm{I}*{d-d_{\mathrm{latent}}})).
Compute
[
\rv{s} = T_\phi(\rv{z}),\qquad
\rv{w} = w^* + \mm{D}^{1/2}\big(\mm{U},\rv{s} + \mm{E}*\perp,\sigma*\perp,\rv{\varepsilon}*\perp\big).
]
This is a **noisy injective flow** (a special case of “flows across dimensions” and SurVAE/injective flows). It preserves a tractable log density and keeps compute dominated by (O(d,d*{\mathrm{latent}})). ([arXiv][1])

**Log density.**
In the orthonormal basis ([\mm{U},\mm{E}*\perp]) scaled by (\mm{D}^{1/2}), the Jacobian of the map ((\rv{z},\rv{\varepsilon}*\perp)\mapsto \rv{w}) is **block‑triangular**, so
[
\log q_\phi(\rv{w}) =
\log \mathcal{N}(\rv{z},;,0,\mm{I})+\log \mathcal{N}(\rv{\varepsilon}*\perp,;,0,\mm{I})
-\tfrac12\log|\mm{D}| - (d-d*{\mathrm{latent}})\log \sigma_\perp

* \log\big|\det \nabla T_\phi(\rv{z})\big|.
  ]
  This makes the **entropy and the (\log q)** terms cheap: the only nontrivial determinant is the **(d_{\mathrm{latent}}\times d_{\mathrm{latent}})** Jacobian of (T_\phi).

> Why this shape?
>
> * It gives you a **full‑dimensional** (q) for the ELBO even when (d_{\mathrm{latent}}\ll d) (no singular density).
> * It aligns compute with the **intrinsic** manifold dimension.
> * As (\sigma_\perp\to 0), you concentrate onto the manifold (you said it’s fine if quality is arbitrarily bad when (d_{\mathrm{latent}}) is too small).

---

## 1) Objective and gradient estimator (reverse‑KL with pathwise/STL)

We maximize (\mathrm{ELBO}(\phi)=\Ex_{q_\phi(\rv{w})}[\log p(\rv{w})]+\H(q_\phi)). We never need the normalizer of (p).

**Pathwise (reparameterization) gradient.**
Write (\rv{w}=g_\phi(\rv{z},\rv{\varepsilon}*\perp)) as above. For any model term (f(\rv{w})),
[
\nabla*\phi \Ex[f(\rv{w})] ;=; \Ex\big[\nabla_{\rv{w}} f(\rv{w});\nabla_\phi g_\phi\big].
]
Use this for the (\Ex[\log p(\rv{w})]) term: we only need (\nabla_{\rv{w}}\log p(\rv{w})) (via backprop through the network, minibatched if needed).

**“Sticking‑the‑Landing” (STL).**
When differentiating (\Ex[\log q_\phi(\rv{w})]), stop gradient through (\rv{w}) and only differentiate the **explicit** dependence of (\log q_\phi) on (\phi) (e.g., Jacobian terms, (\log|\mm{D}|), etc.). This is **unbiased** and typically lower variance for VI; it’s a standard trick that becomes especially effective as (q) approaches (p). ([arXiv][2])

> In code: `loss = -logp(w).detach_grad_wrt_q + logq_explicit_only;` i.e., do **not** backpropagate the (-\log q) term through (\rv{w}).

**Multi‑sample/IWAE tightening (optional).**
If you want a tighter bound while keeping gradients well‑behaved as samples increase, use **DReG** (doubly‑reparameterized gradients) for the IWAE objective; this avoids the signal‑to‑noise collapse and stays unbiased. It’s a worthwhile drop‑in if you’re already drawing (K>1) samples per step. ([arXiv][3])

**Implicit/Generalized reparameterization (edge cases).**
If you ever swap the Gaussian base for hard‑to‑reparam distributions (Gamma, Dirichlet, truncated, etc.), you can still keep pathwise gradients using **implicit** or **generalized** reparameterization gradients. (You probably won’t need these for the core design here.) ([arXiv][4])

---

## 2) Geometry & preconditioning (how to pick (\mm{U}), (\mm{D}), and (\sigma_\perp))

**2.1 Obtain a tangent basis (\mm{U}).**
We want columns that span directions of **low curvature / high posterior variance** near (w^*). Cheap options:

* **Hessian‑vector products** at (w^*) (per‑batch) with a few steps of Lanczos/LOBPCG to extract top eigenvectors of the **inverse curvature** (or equivalently bottom eigenvectors of (\nabla^2 U)). Use GGN/Fisher or K‑FAC blocks if you prefer.
* **SGD path PCA** (PCA on recent iterates or gradients) also finds flat directions.

You can **optimize (\mm{U})** jointly with (\phi) on the Stiefel manifold if you want, but in practice **freezing (\mm{U})** after a short warm‑start often suffices. If you do train (\mm{U}), param with a stack of Householder/Cayley layers to maintain orthonormality with (O(d,d_{\mathrm{latent}})) cost.

**2.2 Preconditioner (\mm{D}).**
Take (\mm{D}) diagonal to start (Adam’s running second moment, diagonal Fisher, or diagonal Hessian approximation). It whitens scales for stable training and enters log density via (\tfrac12\log|\mm{D}|). You can let (\log \mathrm{diag}(\mm{D})) be trainable with a floor.

**2.3 Orthogonal noise (\sigma_\perp).**
Set a small constant, e.g. (10^{-3})–(10^{-2}) times the median diagonal of (\mm{D}), or **learn** it with (\sigma_\perp\ge \sigma_{\min}) (hard floor). Making (\sigma_\perp) small concentrates mass on the learned manifold; you explicitly said you don’t mind being arbitrarily bad if (d_{\mathrm{latent}}) is too small.

---

## 3) Flow architecture in the latent space

Because (d_{\mathrm{latent}}) is tiny, we can afford **very expressive 1D monotone transforms** inside an **autoregressive** or **coupling** template:

* **Neural spline flows** (rational‑quadratic splines): scalar‑wise monotone with easy (\log|\det|) and great flexibility.
* **Autoregressive (MAF/IAF)** or **coupling (RealNVP/Glow)** templates give triangular/block‑triangular Jacobians; with (d_{\mathrm{latent}}) this small, either is fine.
* For extra low‑rank structure, **Sylvester/planar** layers are also cheap in (O(d_{\mathrm{latent}}^2)).
  All of these come with standard, stable log‑det computations. ([arXiv][5])

**Manifold/injective‑flow perspective.**
Our construction is also a special case of **injective/noisy flows** (mapping from low to high dimension with added noise), with strong theory and estimators developed in recent years; if you later want to let the **orthogonal noise depend on (\rv{z})** (a local normal bundle), you can borrow gradients/estimators from the injective‑flows literature. ([arXiv][1])

---

## 4) Gradient estimators & variance reduction (what actually lowers wall‑clock)

**4.1 STL for the entropy term.**
Use **STL**: detach (\rv{w}) inside (-\log q_\phi(\rv{w})); only differentiate explicit (\phi)-dependence (Jacobian of (T_\phi), (\mm{D}), (\sigma_\perp)). This is typically your single biggest variance reduction. ([arXiv][2])

**4.2 Control‑variate for the model term.**
Reuse your idea: subtract a local quadratic at (w^*) using HVPs,
[
\tilde U(\rv{w}) = U(\rv{w}) - \tfrac12(\rv{w}-w^*)^\top \widehat{\mm{H}},(\rv{w}-w^*),
]
and add back (\tfrac12\mathrm{tr}(\widehat{\mm{H}}\ \Sigma_q)) analytically. This reduces variance in (\Ex[\log p(\rv{w})]) without bias.

**4.3 Antithetic sampling & QMC.**
Use antithetic pairs ((\rv{z},\rv{\varepsilon}*\perp)) and ((-\rv{z},-\rv{\varepsilon}*\perp)). With tiny (d_{\mathrm{latent}}), **Sobol** sequences for (\rv{z}) often pay off.

**4.4 Rao–Blackwellization (where possible).**
If your likelihood decomposes over data, adopt **local reparameterization** to move parameter noise to **activation noise** and reduce minibatch variance (especially for linear layers). This is orthogonal to the flow design and cuts gradient variance for (\nabla_{\rv{w}}\log p(\rv{w})). ([arXiv][6])

**4.5 Multi‑sample bounds with DReG (optional).**
If you want the IWAE objective, use **DReG** to keep gradients healthy as (K) increases. ([arXiv][3])

**4.6 Beyond‑standard pathwise gradients (rarely needed here).**
If you add exotic latent distributions or constraints, **implicit** and **transport‑equation** pathwise gradients extend reparameterization with low variance. Good to have in the toolbox. ([arXiv][4])

---

## 5) Training loop (end‑to‑end)

**Inputs.** MAP (w^*), data minibatches, HVP oracle at (w^*), batch size (B), latent dim (d_{\mathrm{latent}}), noise floor (\sigma_{\min}).

**Setup.**

1. Estimate (\mm{U}) via a few HVP‑Lanczos iterations (or SGD‑path PCA).
2. Initialize (\mm{D}) from Adam’s second moment or diagonal Fisher at (w^*).
3. Set (\sigma_\perp = \sigma_{\min}) (or learnable with lower bound).
4. Initialize (T_\phi) close to identity (small‑scale splines or small‑gain couplings).

**Each step.**

1. Sample ({(\rv{z}^{(k)},\rv{\varepsilon}*\perp^{(k)})}*{k=1}^K) (antithetic/QMC if you like).
2. Map to (\rv{w}^{(k)} = g_\phi(\rv{z}^{(k)},\rv{\varepsilon}_\perp^{(k)})).
3. On a data minibatch, compute (\log p(\rv{w}^{(k)})) (or (-U)) and (\nabla_{\rv{w}}\log p(\rv{w}^{(k)})) by backprop through the NN with weights (\rv{w}^{(k)}).
4. Compute (\log q_\phi(\rv{w}^{(k)})) using the block‑triangular formula (cheap).
5. **ELBO estimate.** Average over (k) (or IWAE aggregator).
6. **Gradients.**

   * For (T_\phi), (\mm{D}), (\sigma_\perp): pathwise gradient through (\log p(\rv{w})) plus **STL** for (-\log q).
   * For (\mm{U}) (if trainable): pathwise via (\partial \rv{w}/\partial \mm{U} = \mm{D}^{1/2},\rv{s}) and Riemannian update on the Stiefel manifold (or project via QR/Householder after each step).
7. **Variance control.** Apply the quadratic control variate around (w^*) for the (\log p) term; use a moving baseline for scalar objectives.
8. Update parameters with AdamW (small LR for (\log \mathrm{diag}(\mm{D})) and (\sigma_\perp)).

**Complexity.** Per sample, (O(d,d_{\mathrm{latent}})) dominates (matrix–vector ops with (\mm{U}) and (\mm{D}^{1/2})); latent‑flow ops scale in (d_{\mathrm{latent}}). That’s typically **comparable to an SGD step** if you keep (d_{\mathrm{latent}}\in{2,4,8,16}).

---

## 6) Practical choices that matter

* **Latent flow:** use **neural spline flows** (coupling or autoregressive); their scalar monotone transforms are stable and very expressive. ([arXiv][5])
* **Across‑dimension correctness:** our construction is a streamlined case of **injective/noisy flows**; if you decide to let the orthogonal noise depend on (\rv{z}) (learn a normal bundle), use estimators from **NIF/SurVAE** to keep likelihoods tractable. ([arXiv][1])
* **Review of flows/choices:** for a broad, up‑to‑date perspective on what’s robust in practice, Papamakarios et al. is a good unifying reference. ([Journal of Machine Learning Research][7])

---

## 7) Using the learned flow for **local curvature estimates**

You wanted curvature near (w^*). The flow gives you multiple routes:

1. **Posterior covariance restricted to the manifold.**
   Compute (\mathrm{Cov}*q[\rv{w}] \approx \mm{D}^{1/2}\Big(\mm{U},\mathrm{Cov}[T*\phi(\rv{z})],\mm{U}^\top + \sigma_\perp^2,\mm{E}*\perp\mm{E}*\perp^\top\Big)\mm{D}^{1/2}). The dominant contribution is in (\mathrm{span}(\mm{U})). If the posterior is locally Gaussian, the inverse of this (restricted) covariance approximates the **restricted Hessian**.

2. **Quadratic regression of (U(\rv{w})) under (q).**
   Use samples from (q) to fit (\tfrac12(\rv{w}-w^*)^\top \widehat{\mm{H}} (\rv{w}-w^*)) via ridge regression (or via (\Ex[\nabla^2 U(\rv{w})]) with HVPs). This produces an empirical curvature aligned with the learned manifold.

3. **Pullback curvature in latent space.**
   In latent coords, the energy is (U\big(w^*+\mm{D}^{1/2}\mm{U}T_\phi(\rv{z})\big)). The **pullback Hessian** at (\rv{z}=0) transports curvature into a small (d_{\mathrm{latent}}\times d_{\mathrm{latent}}) matrix via the chain rule (cheap to approximate with HVPs at a handful of latent points).

---

## 8) What happens if (d_{\mathrm{latent}}) is too small?

Two safe behaviors you can encourage (both ELBO‑friendly):

* Keep (\sigma_\perp) tiny but nonzero: you’ll **project** the posterior into the manifold and soak up the residual in the orthogonal noise. This preserves a finite KL and keeps training stable.
* Allow (\mm{D}) to adapt: orthogonal scales can rise if the model needs extra thickness.

You said you **don’t care** if quality is arbitrarily bad in that setting; the above ensures training stays numerically healthy without pretending to explain away the missing dimensions.

---

## 9) Small but useful details

* **Minibatch‑likelihood variance:** apply **local reparameterization** when evaluating (\log p(\rv{w})) inside linear layers to slash gradient noise (especially helpful when (d) is huge). ([arXiv][6])
* **Numerical floors:** (\sigma_\perp \ge 10^{-4}), (\log \mathrm{diag}(\mm{D})\in[\log 10^{-8}, \log 10^2]) are practical.
* **Initialization:** start (T_\phi) close to identity, e.g., splines with 3–5 bins, tiny slopes offsets.
* **Monitoring:** track (\ELBO), (\KL(q\Vert p)) components, (\mathrm{Cov}*q[\rv{w}]) trace along (\mm{U}), and **effective latent volume** via (\Ex[\log|\det\nabla T*\phi|]).
* **When multi‑modal:** if the local basin is not enough, you can either (i) mixture‑of‑flows with **relaxed** component selection (Gumbel‑Softmax) and **Rao–Blackwellized** control variates, or (ii) fit distinct local flows per basin (cheaper if modes are well separated). (STL still applies; DReG helps for multi‑sample training.)

---

## 10) Why the reparameterization choices are “SOTA enough” here

* **STL** (detach (-\log q) through samples) is still a go‑to, simple, low‑variance improvement for VI‑style training. ([arXiv][2])
* **DReG** is the standard fix for multi‑sample bounds (IWAE) to keep gradients unbiased and high‑SNR as (K) grows. ([arXiv][3])
* **Implicit/transport‑equation gradients** cover cases where you extend the base family beyond Gaussians. ([arXiv][4])
* **Injective/noisy flows & SurVAE** provide the right formalism for across‑dimension maps with exact (or bounded) likelihoods, matching our manifold‑plus‑noise construction. ([arXiv][1])
* **Neural spline flows / MAF/RealNVP/Sylvester** are robust latent‑space choices with cheap log‑dets and strong empirical performance. ([arXiv][5])
* For a unifying overview of design choices and trade‑offs in flows and their gradients, see the **JMLR review**. ([Journal of Machine Learning Research][7])

---

## 11) Minimal pseudocode (framework‑agnostic)

```
Given w_star, data, latent_dim, sigma_perp_min

# Geometry & scales
U = tangent_basis_via_HVPs(w_star, latent_dim)          # or SGD-PCA
Eperp = orthonormal_complement(U)
D = diag_from_Adam_or_Fisher(w_star)
sigma_perp = Parameter(init=sigma_perp_min, constrained >= sigma_perp_min)

# Latent flow T_phi in R^{latent_dim} (e.g., spline-coupling)
phi = init_latent_flow_identity(latent_dim)

for step in 1..T:
    # Sample base noise (use antithetic/QMC if desired)
    z, eps_perp = sample_standard_normals(latent_dim, d - latent_dim)
    s = T_phi(z)                         # latent transform
    w = w_star + D^{1/2} ( U s + Eperp (sigma_perp * eps_perp) )

    # Model term (minibatch)
    logp, grad_logp_w = evaluate_logp_and_grad(w, minibatch)

    # Density term (explicit; STL: detach w here)
    logq = log_normal(z) + log_normal(eps_perp) \
           - 0.5*logdet(D) - (d - latent_dim)*log(sigma_perp) \
           - logabsdet(Jacobian_T_phi(z))

    elbo = logp + ( - logq )     # H(q) = -E[log q], already included
    loss = -stopgrad_for_logq_through_w(elbo)

    loss.backward()               # pathwise for logp, explicit for logq
    optimizer.step()
    optimizer.zero_grad()
```

---

## 12) What you’ll get out of it

* **Samples** that live close to a curved, (d_{\mathrm{latent}})-dimensional valley around (w^*).
* **Cheap log‑densities** and **low‑variance gradients** (STL + preconditioning).
* **Local curvature summaries** from (q): variances along (\mathrm{span}(\mm{U})), posterior‑consistent covariance estimates, and inexpensive latent‑space Hessians.

---

### Pointers to the specific techniques referenced

* STL (sticking‑the‑landing) low‑variance VI gradients. ([arXiv][2])
* Implicit & generalized reparameterization gradients. ([arXiv][4])
* Injective/noisy flows and SurVAE (across dimensions, manifold + noise). ([arXiv][1])
* Neural spline flows, RealNVP/MAF/IAF, Sylvester flows. ([arXiv][5])
* Survey of normalizing flows, modeling & inference. ([Journal of Machine Learning Research][7])
* Local reparameterization trick for minibatch variance reduction. ([arXiv][6])

---

## 13) Practical JAX/vmap Implementation Learnings

**Context:** This section documents critical lessons learned from implementing Flow VI with JAX vmap for multi-chain parallel execution (Nov 2025).

### 13.1 PRNG Key Management

**Issue:** FlowJAX and JAX vmap have strict requirements for PRNG key formats.

**Solution:**
* Use **typed threefry2x32 keys globally**: Set `jax.config.update("jax_default_prng_impl", "threefry2x32")` at package import time
* **Avoid RBG**: RBG keys have unusual vmap batching semantics that can cause shape mismatches
* **Normalize keys on the host**: Convert legacy `uint32[2]` keys to typed keys *before* entering vmap/jit:
  ```python
  def ensure_typed_key(key):
      """Host-side normalization (outside jit/vmap)."""
      if isinstance(key, jax.Array) and key.dtype == jnp.uint32:
          if key.ndim == 1 and key.shape[-1] == 2:
              return jr.wrap_key_data(key, impl="threefry2x32")
      return key
  ```
* Set `JAX_DEFAULT_PRNG_IMPL=threefry2x32` in worker environments (Parsl, SLURM, etc.)

### 13.2 Vmap Compatibility: Returns Must Be Pure Data

**Critical constraint:** `jax.vmap` can only batch over **valid JAX types** (arrays, scalars). Functions, Equinox modules, or objects with static components will cause:
```
TypeError: Output from batched function ... with type <class 'jaxlib._jax.PjitFunction'> is not a valid JAX type
```

**Rules for VI algorithm returns:**
1. **Return only JAX arrays and Python scalars**
   * ✅ Good: `{"lambda_hat": jnp.array(...), "traces": {"elbo": jnp.array(...)}}`
   * ❌ Bad: `{"final_dist": flow_object}` (Equinox module with function references)

2. **No FlowJAX modules in return values**
   * FlowJAX `Flow` objects contain static parts (bijector functions like `jax.nn.softplus`)
   * These cannot pass through vmap - extract numerical arrays only

3. **Wrap all values in `jnp.asarray()` for type safety**
   ```python
   extras = {
       "Eq_Ln": jnp.asarray(E_L),           # Not just E_L
       "Ln_wstar": jnp.asarray(L0),
       "variance_reduction": jnp.asarray(1.0, dtype=E_L.dtype),  # Even scalars
   }
   ```

4. **Add vmap-safety validation during development**
   ```python
   def _all_leaves_are_arrays(x):
       leaves, _ = jax.tree_util.tree_flatten(x)
       return all(isinstance(l, (jax.Array, jnp.ndarray, int, float, bool)) for l in leaves)

   assert _all_leaves_are_arrays(result), "Non-JAX objects in returns"
   ```

### 13.3 No Python Casts Inside Traced Code

**Issue:** Cannot use `float()`, `int()`, `list()` on traced values inside `@jax.jit` or `jax.vmap`:
```python
# Inside @jax.jit function:
work_fge = jnp.asarray(batch_size / float(n_data), dtype=jnp.float64)  # ❌ FAILS
# Error: ConcretizationTypeError when n_data is traced
```

**Solution:** Let JAX handle type promotion automatically:
```python
work_fge = jnp.asarray(batch_size / n_data, dtype=jnp.float64)  # ✅ Works
```

**Rule:** Inside jit/vmap/scan, **never** call:
* `float(x)`, `int(x)` on arrays or traced values
* `len(x)` on traced arrays (use `.shape[0]` or pass length as static arg)
* `list(x)`, `tuple(x)` on traced arrays

### 13.4 STL Implementation Details

The theoretical "Sticking the Landing" gradient estimator requires **keeping gradients flowing through the model term but not through the entropy term via samples**.

**Practical implementation:**
* Return `lambda_hat` (and all results) as **JAX arrays**, not Python floats:
  ```python
  # ❌ Bad (breaks vmap, breaks STL):
  return {"lambda_hat": float(lambda_hat), ...}

  # ✅ Good (vmap-safe, STL-compatible):
  return {"lambda_hat": lambda_hat, ...}  # Keep as JAX array
  ```
* The gradient stopping happens in the loss computation, not in the return values
* Converting to Python types breaks both vmap AND gradient flow

### 13.5 Interface Consistency Across VI Algorithms

**Problem:** Different VI algorithms (MFA, Flow) must return compatible structures for the sampling infrastructure.

**Required return structure:**
```python
{
    "lambda_hat": jnp.array(...),           # Shape: scalar or (,)
    "traces": {                             # Per-iteration metrics
        "elbo": jnp.array(...),             # Shape: (num_steps,)
        "grad_norm": jnp.array(...),
        # ... other traces
    },
    "extras": {                             # Final evaluation metrics
        "Eq_Ln": jnp.array(...),            # Expected negative log-likelihood
        "Ln_wstar": jnp.array(...),         # Log-likelihood at MAP
        "cv_info": {                        # Control variate diagnostics
            "Eq_Ln_mc": jnp.array(...),
            "Eq_Ln_cv": jnp.array(...),
            "variance_reduction": jnp.array(...),
        },
    },
    "timings": {                            # Wall-clock times (Python floats OK here)
        "adaptation": 0.0,                  # Flow/MFA have no adaptation phase
        "sampling": float(...),             # Total training+eval time
        "total": float(...),
    },
    "work": {                               # FGE accounting (Python ints/floats OK)
        "n_full_loss": int(...),
        "n_grad": int(...),
        # ...
    },
}
```

**Key points:**
* `timings` uses `"adaptation"/"sampling"/"total"` keys (not `"train"/"eval"`)
* `extras` follows MFA's structure with `Eq_Ln`, `Ln_wstar`, `cv_info`
* All arrays wrapped in `jnp.asarray()` for type safety

### 13.6 Equinox Module Handling

**FlowJAX uses Equinox**, which partitions modules into:
* **Dynamic part** (arrays): Can be vmapped, differentiated
* **Static part** (functions, shapes): Cannot change during tracing

**When using Equinox flows:**
1. **Partition at the start of `run()`** (outside loops):
   ```python
   flow_params, flow_static = eqx.partition(flow, eqx.is_array)
   ```

2. **Use only `flow_params` inside jit/vmap/scan**

3. **Recombine when needed** inside traced code:
   ```python
   flow = eqx.combine(flow_params, flow_static)
   samples = flow.sample(key, ...)
   ```

4. **Never return `flow_static` or combined modules** from vmapped functions

### 13.7 Debugging Vmap Issues

**Symptoms of vmap incompatibility:**
* `TypeError: ... is not a valid JAX type`
* `ConcretizationTypeError` when calling `float()`, `int()`, etc.
* `PRNG key shape mismatch` errors

**Debugging steps:**
1. **Check return types** by adding this before your vmap call:
   ```python
   test_result = run_one_chain(chain_keys[0])  # Run single chain
   leaves, _ = jax.tree_util.tree_flatten(test_result)
   for i, leaf in enumerate(leaves):
       print(f"Leaf {i}: type={type(leaf)}, shape={getattr(leaf, 'shape', '?')}")
   ```

2. **Look for non-array types**: Functions, modules, or objects in the output

3. **Check PRNG keys**: Ensure all keys are typed and consistent:
   ```python
   key_data = jax.random.key_data(rng_key)
   print(f"Key shape: {key_data.shape}, impl: {rng_key._impl}")
   ```

4. **Use `JAX_TRACEBACK_FILTERING=off`** to see full stack traces

### 13.8 Multi-Chain Execution Patterns

**For vmap-compatible algorithms (MFA, fixed Flow):**
```python
def run_one_chain(chain_key):
    return algo.run(rng_key=chain_key, ...)

results = jax.vmap(run_one_chain)(chain_keys)  # Parallel across chains
```

**For algorithms with vmap issues (legacy pattern, now deprecated):**
```python
# Sequential fallback (slower, but works with any return type)
results_list = [run_one_chain(k) for k in chain_keys]
lambda_hats = jnp.stack([r["lambda_hat"] for r in results_list])
# ... manually stack other results
```

**Current status:** Flow VI is now vmap-compatible (as of Nov 2025 fixes).

---

If you want, I can also tailor this to **your exact training stack** (e.g., how you're computing HVPs at (w^*), what your data minibatching looks like, and whether you want to train (\mm{U}) jointly or freeze it after a warm‑start).

[1]: https://arxiv.org/abs/2006.13070 "Normalizing Flows Across Dimensions"
[2]: https://arxiv.org/abs/1703.09194 "Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference"
[3]: https://arxiv.org/abs/1810.04152 "Doubly Reparameterized Gradient Estimators for Monte Carlo Objectives"
[4]: https://arxiv.org/abs/1805.08498 "Implicit Reparameterization Gradients"
[5]: https://arxiv.org/abs/1906.04032 "Neural Spline Flows"
[6]: https://arxiv.org/abs/1506.02557 "Variational Dropout and the Local Reparameterization Trick"
[7]: https://www.jmlr.org/papers/volume22/19-1028/19-1028.pdf "Normalizing Flows for Probabilistic Modeling and Inference"
