**Several places currently assume “Markov chains that emit Ln histories” and “MCMC‑specific diagnostics.”** A variational path violates both. If you drop VI into the existing code as a fourth “sampler,” you’ll mainly break: (i) analysis that hard‑requires `Ln` traces shaped like chains, (ii) ESS/R‑hat/WNV logic that presumes Markov dependence, (iii) dtype/precision expectations wired to MCMC, and (iv) places where the code and configs use MCMC‑only fields (acceptance, energy, warmup). Below I’ve enumerated concrete breakpoints and the minimum abstraction changes to make VI a clean, first‑class option.

---

## What will break (by layer), and why

### 1) Trace shape & required keys in analysis

* `analyze_traces()` **raises** if the traces dict doesn’t contain `'Ln'` and expects an array with shape `(chains, draws)`. VI doesn’t have “chains” by default and may only produce Monte Carlo draws from `q`. If you don’t synthesize chain structure (e.g., K independent seeds or 1 pseudo‑chain), this function fails immediately.

* The ArviZ conversion helpers are designed for **ragged Ln histories across chains** and optional HMC extras (acceptance, energy). Those extras won’t exist for VI, which is OK (they’re optional), but `Ln` itself is non‑negotiable. If `Ln` is present but you pass a 1‑D vector, code already reshapes to `(1, T)`, so a **single‑chain VI path works** if you populate `Ln` properly.

### 2) ESS, R‑hat, and “work” metrics assume MCMC

* Downstream metrics compute ESS, R‑hat and work‑normalized variance from the ArviZ summary for `posterior['llc']`. That logic presumes **autocorrelated MCMC** samples. For **IID VI draws**, the ArviZ ESS may be artificially high or degenerate (especially with one “chain”), and R‑hat is meaningless. If you leave it as is, you’ll get misleading “efficiency” numbers. The function also divides by “work” that is currently defined as *full loss evals + minibatch grads × (b/n)*; you must **log “work” for VI** (optimization steps × minibatch grads, + final MC evals), or you’ll return NaNs.

### 3) Dtype and precision conventions are sampler‑coded

* The codebase hard‑codes **float32 for SGLD** and **float64 for HMC/MCLMC** (and toggles `jax_enable_x64` in configs/workflow). VI is designed to train in float32 with occasional f64 in tiny `r×r` blocks, so if you reuse the HMC path’s `enable_x64=True` or the SGLD path’s assumptions blindly, you’ll either pay unnecessary cost or run into numeric mismatches with target artifacts built under a different global precision flag. See the “JAX precision conventions” notes and `jax_enable_x64` propagation in Snakefile.

### 4) Sampler registry & CLI/config wiring

* The Stage‑B runner wires **three named samplers** and expects sampler‑specific configs in `lambda_hat/conf/sample/sampler/{sgld,hmc,mclmc}.yaml`. There is **no `vi.yaml`** and no `vi` branch. Trying to select `sampler.name=vi` will 404 the config and/or raise in the dispatch table.

* The older “online” runner uses a **`SAMPLERS` registry** that couples init/run lambdas and inserts HMC/SGLD‑specific arguments (acceptance, energy, warmup, preconditioner flags, etc.). Dropping VI into that table without adapting the environment payload will break at call‑sites.

### 6) Posterior factories & signatures

* MCMC paths use `make_logpost` / `make_logpost_and_score(...)` that combine **loss**, **θ₀**, **n**, **β**, **γ** (and a minibatch score for SGLD). The VI plan needs **`loss_full_fn`** (full‑data mean) and optionally **`loss_minibatch_fn`** for STL updates. Those are available in both stacks but under **different names/signatures** — you’ll need thin adapters so the VI code can stay agnostic to whether params are vectors or PyTrees.

### 7) Artifacts & run outputs

* Stage‑B expects to produce per‑sampler **ArviZ NetCDFs and metrics** under the run dir (e.g., `{sampler}.nc`, `metrics.json`, `L0.txt`). If the VI runner only returns a scalar plug‑in estimate and doesn’t populate `Ln_histories` (or doesn’t write `{sampler}.nc`), the **Snakefile downstream rules and plotting** will have nothing to consume. The examples and “What’s in a run” doc assume those files exist.

---

## Minimal abstraction changes to make VI a first‑class citizen

> Goal: keep Stage‑A (target artifacts) and Stage‑B (samplers) intact, and treat VI as “another engine that produces an IID `Ln` series,” not a Markov chain.

### A) Normalize the **sampler interface** around a common `RunResult`

Create/standardize a small dataclass used by **all** engines:

```python
class RunResult(NamedTuple):
    Ln_histories: list[np.ndarray]      # length C; each shape (T,)
    theta_thin:  list[np.ndarray] | None
    timings:     dict                   # {"sampling": sec, "adaptation": sec (optional)}
    work:        dict                   # {"n_full_loss": float, "n_minibatch_grads": float}
    extras:      dict                   # {"acceptance": [...], "energy": [...]} when present
```

* MCMC already fits this shape (it logs `Ln`, acceptance, energy, cumulative FGEs/time).
* **VI runner must fill**:

  * `Ln_histories=[Ln_draws]` (use **1 chain** or **K seeds** → K chains);
  * `theta_thin` optionally (e.g., projections or subset dims);
  * `work={"n_full_loss": eval_samples, "n_minibatch_grads": steps}`;
  * `timings={"sampling": optimize_time + eval_time}`.

This keeps `analyze_traces()` and figure code unchanged (they only need `Ln` and timings/work).

### B) Add a **VI config** and a branch in the Stage‑B dispatcher

* New file: `lambda_hat/conf/sample/sampler/vi.yaml` (steps, batch_size, lr, M, r, gamma_localizer, eval_samples, cv_kind). The Snakefile composes the sample config from `.../sampler/{name}.yaml`, so this is required.

* In the Stage‑B entrypoint/runner, add `"vi": {...}` to the registry (mirroring `sgld`/`hmc`/`mclmc`) and call your `fit_vi_and_estimate_lambda(...)` wrapper. Populate `RunResult` (above).

### C) Decouple **precision** from “sampler type”

* Today, HMC/MCLMC imply f64, SGLD implies f32. Introduce per‑sampler **`dtype`/`enable_x64`** in config and honor it at **kernel boundaries only** (loss fns, parameter casting), rather than driving a global `jax_enable_x64` from Snakefile for all Stage‑B runs. That way VI can stay f32‑dominant but still use small f64 blocks in its internal Cholesky/solve when asked (the plan suggests this pattern).

### D) Provide a **TargetAdapter** so VI sees a uniform interface

* Implement two shims:

  * **PyTree→flat** and **flat→PyTree** using `ravel_pytree` for the `lambda_hat.targets` path (Haiku params).
  * A no‑op adapter for the `llc.targets` path (already flat).
    The VI code you shared already uses exactly this flatten/unflatten discipline, so just **centralize it** and export `loss_batch_fn(params, (Xb, Yb))` and `loss_full_fn(params)` for both stacks.

### E) Make the **analysis layer sampler‑agnostic**

* Keep ArviZ conversion the same (store `posterior["L"]` and derived `posterior["llc"]`). For VI (IID draws), retain `(chains=1, draws=T)` or `(chains=K, draws=T)` if you run K seeds.

* In efficiency reporting, **branch on sampler**:

  * For `sampler=="vi"`, set `ess = draws_total` (IID) and **do not report R‑hat** (or set it to NaN/omit). Compute WNV using your `work` counters (`steps` × minibatch‑grads plus final MC full loss evals). The structure in `efficiency_metrics(...)` already receives `work` and `timings`; you only need the short conditional.

### F) Unify the **posterior hyper‑parameters** path

* Keep a single `compute_beta_gamma(...)` and a single “make log‑posterior or minibatch score” helper for both stacks to avoid drift (`llc/posterior.py` vs `lambda_hat/posterior.py` currently duplicate functionality). Otherwise, you’ll mis‑tune β or γ between MCMC and VI.

---

## Concrete “edit list” (smallest changes that work)

1. **Config & workflow**

   * Add `lambda_hat/conf/sample/sampler/vi.yaml` (VI hyper‑params). Wire the Snakefile to pick it up via `sampler.name: vi`.
   * In docs/README, list VI as a fourth engine so the Snakemake “N targets × M samplers” story remains true.

2. **Runner**

   * Add a `"vi"` entry to the Stage‑B registry (the same place `sgld`, `hmc`, `mclmc` are registered) that:

     * builds the whitening (optional) and calls your `fit_vi_and_estimate_lambda(...)`,
     * samples `eval_samples` points from `q` to form `Ln_histories=[Ln_samples]`,
     * fills `work={"n_minibatch_grads": steps, "n_full_loss": eval_samples}`,
     * returns a `RunResult` consumable by `analyze_traces()` / figures.

3. **Precision**

   * Accept `sampler.vi.dtype` (default float32) and **don’t** flip global `jax_enable_x64` for VI. If you need f64 just inside `r×r` Cholesky/solves, cast locally (your plan already sketches that).

4. **Analysis**

   * In `efficiency_metrics(...)`, if `idata.attrs["sampler"] == "vi"`, set:

     * `ess = C×T` (number of IID draws),
     * suppress/NaN R‑hat,
     * compute WNV with `work` (above). The function already has the hooks (`timings`, `work`, `n_data`). Add a 6‑line conditional.

5. **Duplication cleanup**

   * Pick **one** stack (`llc/*` or `lambda_hat/*`) as the Stage‑B surface. Export:

     * `TargetBundle` with **both** `loss_full` & `loss_minibatch` and **params0** (+ flatten/unflatten helpers).
     * `posterior.compute_beta_gamma`, `make_logpost...` (single source of truth).
       That avoids “works under `llc`, breaks under `lambda_hat`” issues.

---

## Checklist you can run down (order matters)

1. **Add `vi.yaml`** and a `vi` branch in the Stage‑B registry.
2. **Implement `run_vi(...)`** that returns `RunResult` with:

   * `Ln_histories=[Ln_samples]` (shape `(T,)` or `(C,T)`),
   * `timings={"sampling": optimize_time + eval_time}`,
   * `work={"n_minibatch_grads": steps, "n_full_loss": eval_samples}`.
3. **Ensure `analyze_traces()` sees `'Ln'`** → the function will create ArviZ data and compute λ̂ automatically.
4. **Guard metrics for VI** (ESS=draws, R‑hat→NaN/omit; use `work` to compute WNV).
5. **Keep VI mostly f32**; don’t toggle global `jax_enable_x64` for VI. Cast locally for `r×r` when needed.
6. **Unify posterior factories** and add a tiny **TargetAdapter** so VI sees the same `(loss_full_fn, loss_batch_fn, w*)` regardless of which stack produces the target.

---

## Notes tying to your VI plan

* Your plan’s **equal‑mean mixture of factor analyzers**, STL gradients, RB for weights, whitening, and the **HVP control variate** all plug in cleanly **as long as** Stage‑B hands you `(loss_batch_fn, loss_full_fn, w*, n, β, γ)` and a place to **return an IID `Ln` series** and **work counters**. The current pipeline already has all inputs; we only need to adapt outputs so the same analysis and Snakemake steps run. (The minimal `RunResult` above is exactly what’s missing.)

* For **diagnostics**, VI won’t emit acceptance/energy; the conversion and plotting code tolerate their absence (they’re optional), but **ESS/R‑hat need the VI special‑case** (or you risk reporting nonsense).

If you make the interface changes above, you can add VI as a one‑file engine without destabilizing Stage‑A, Snakemake, or plots.
