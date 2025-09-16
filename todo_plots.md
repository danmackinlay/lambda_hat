You’re right: the diagnostics are shaky. There are **two separate problems** mixed together:

1. **Wrong baseline in the plots (causes tiny y‑axis):**
   Your *running LLC* figure is computed as $ \lambda_t = n\beta(\bar L_n^{(t)} - L_0)$. The code does that correctly, **but the $L_0$ you pass is not the empirical minimizer**—it’s the “reference $\theta_0$” value (your new pipeline even prints “*L0 at reference θ0*”). That pushes $\bar L_n - L_0$ toward \~0, so the plot hovers in the **0.01–0.08** range even when your final scalar estimator reports something like **29.694**. In the earlier `main.py` path you trained to the ERM (you saw “Training to empirical minimizer…”), so plots and scalars matched. In the new `pipeline.run_one`, you import `train_erm` but never call it—so **L0 is wrong** in *both* the scalar estimator and the plots for runs done through the new pipeline.

2. **Diagnostics not aligned with modern ArviZ practice:**
   You’re using bulk ESS in one place, a custom running-mean plot elsewhere, and otherwise ignoring ArviZ’s standard diagnostics (rank plots, R‑hat, energy plots, autocorr, ESS evolution/quantile). You also have an unused `_idata_from_L`. You’re leaving a lot of value on the table. ArviZ expects an `InferenceData` with proper chain/draw dims and (for HMC) `sample_stats` (e.g., energy, acceptance). See ArviZ’s ESS, R‑hat, and plotting APIs. ([python.arviz.org][1])

Below I give you (A) a concrete redesign of the diagnostics (what to plot and why), (B) precise code diffs to implement it, and (C) minimal fixes to your pipeline and presets so the pictures line up with the numbers.

---

## A — What to plot (and why), aligned with ArviZ

**Variables to analyze**

* $L_n$ traces per chain (raw and centered $L_n - L_0$) for interpretability.
* The *LLC draws* $\lambda = n\beta(L_n - L_0)$ **as a first-class variable** (“llc”).
* Selected $\theta$ coordinates (thin samples) for trace/ACF sanity.

**Core diagnostics (ArviZ best practice)**

1. **Trace / rank plots for `llc`**
   Use `az.plot_trace` + `az.plot_rank`. Rank plots should look roughly uniform across chains if they target the same distribution; great at spotting chain inconsistencies. ([python.arviz.org][2])

2. **R‑hat (rank‑normalized, split)** for `llc`
   Report R‑hat and flag anything $\widehat R \ge 1.01$. This is the modern, rank‑normalized definition ArviZ exposes. ([python.arviz.org][3])

3. **ESS diagnostics**

   * Show **ESS evolution** and **ESS quantile/local** for `llc` to ensure enough effective samples not only for the mean but across the distribution. ArviZ recommends local/quantile ESS checks for intervals; bulk ESS is for location summaries. ([python.arviz.org][4])
   * As a rule of thumb, aim for **ESS ≥ 100 × (#chains)** and $\widehat R < 1.01$. (Vehtari–Gelman guidance, summarized in ArviZ/ArviZ.jl docs.) ([arviz-devs.github.io][5])

4. **Autocorrelation (ACF) for `llc` and a few $\theta$ dims**
   Visualizes mixing; slow decay ⇒ increase draws / reduce thinning / improve adaptation. ([python.arviz.org][6])

5. **HMC-specific**

   * **Energy plot** (requires `sample_stats.energy`): reveals heavy tails / poor exploration. If you can compute per‑draw Hamiltonian energy, call `az.plot_energy`. ([python.arviz.org][7])
   * **Acceptance evolution** with a target band. Stan’s default target is **0.8**; BlackJAX ChEES targets \~**0.651**; in practice acceptable \~0.6–0.9 depending on model. Show reference lines at 0.65 and 0.8. ([Stan][8])

6. **SGLD / MCLMC caveat**
   $\widehat R$ and ESS are formulated for MCMC; SGLD/MCLMC can be non‑reversible or adjusted. Still, **rank plots/ACF/ESS are informative heuristics** across chains if the estimand is well‑behaved. (ArviZ doesn’t prohibit it; interpret with caution.)

**Readability fixes**

* Always label the running-mean figure **“LLC = nβ(E\[Lₙ] − L₀)”**.
* Plot both per‑chain and pooled running LLC; add a horizontal band at **final mean ± 2·SE**.
* Center $L_n$ plots as **$L_n - L_0$** in a separate figure; this makes the scale meaningful.

---

## B — Drop‑in code changes

### 1) Fix the wrong baseline and make an ArviZ‑friendly `InferenceData`

```diff
diff --git a/llc/pipeline.py b/llc/pipeline.py
@@
-    # Build a self-contained target (NN, quadratic, …)
-    bundle = build_target(key, cfg)
+    # Build a self-contained target (NN, quadratic, …)
+    bundle = build_target(key, cfg)
@@
-    # L0 is provided by the target (at θ0)
-    L0 = float(bundle.L0)
-    print(f"L0 at reference θ0: {L0:.6f}")
+    # IMPORTANT: Use the empirical minimizer for L0
+    # (this was missing; plotting & scalars then disagreed)
+    from llc.experiments import train_erm
+    erm = train_erm(key, bundle, cfg)  # returns (w_hat, L0, …); keep your existing signature
+    L0 = float(erm.L0)
+    print(f"L0 at empirical minimizer: {L0:.6f}")
```

> That single change aligns the *running LLC* plot scale with the reported scalar estimate.

### 2) Replace your diagnostics with an ArviZ‑first suite

```diff
diff --git a/llc/diagnostics.py b/llc/diagnostics.py
@@
-ESS_METHOD = "bulk"
+ESS_METHOD = "bulk"  # bulk ESS: location summaries; use quantile/local for intervals

@@
-def _idata_from_L(Ln_histories: List[np.ndarray]) -> Tuple[Optional[Any], int]:
-    """Create ArviZ InferenceData from L_n histories"""
-    H = _stack_histories(Ln_histories)
-    return (
-        (az.from_dict(posterior={"L": H}), H.shape[1]) if H is not None else (None, 0)
-    )
+def _idata_from_L(Ln_histories: List[np.ndarray]) -> Tuple[Optional[Any], int]:
+    """Create ArviZ InferenceData from L_n histories (posterior group, dims=chain,draw)."""
+    H = _stack_histories(Ln_histories)
+    if H is None:
+        return None, 0
+    # Ensure shape (chain, draw) and proper dims
+    idata = az.from_dict(
+        posterior={"L": H},
+        coords={"chain": np.arange(H.shape[0]), "draw": np.arange(H.shape[1])},
+        dims={"L": ["chain", "draw"]},
+    )
+    return idata, H.shape[1]

+def _idata_from_llc(
+    Ln_histories: List[np.ndarray], n: int, beta: float, L0: float,
+    acceptance_rates: Optional[List[np.ndarray]] = None,
+    energy: Optional[List[np.ndarray]] = None,
+) -> Optional[Any]:
+    """Make InferenceData with `llc` variable and optional sample_stats (acceptance, energy)."""
+    H = _stack_histories(Ln_histories)
+    if H is None:
+        return None
+    llc = n * float(beta) * (H - L0)  # (chain, draw)
+    sstats = {}
+    if acceptance_rates is not None and any(len(a) > 0 for a in acceptance_rates):
+        # ragged → truncate to min length
+        m = min(len(a) for a in acceptance_rates if len(a) > 0)
+        if m > 0:
+            acc = np.stack([a[:m] for a in acceptance_rates], axis=0)
+            sstats["acceptance_rate"] = acc
+            # also trim llc to match if needed
+            llc = llc[:, :m] if llc.shape[1] >= m else llc
+    if energy is not None and any(len(e) > 0 for e in energy):
+        m2 = min(len(e) for e in energy if len(e) > 0)
+        if m2 > 0:
+            en = np.stack([e[:m2] for e in energy], axis=0)
+            sstats["energy"] = en
+            llc = llc[:, :m2] if llc.shape[1] >= m2 else llc
+    idata = az.from_dict(
+        posterior={"llc": llc},
+        sample_stats=sstats if sstats else None,
+        coords={"chain": np.arange(llc.shape[0]), "draw": np.arange(llc.shape[1])},
+        dims={"llc": ["chain", "draw"], **({k: ["chain", "draw"] for k in sstats} if sstats else {})},
+    )
+    return idata
@@
-def _running_llc(
+def _running_llc(
     Ln_histories: List[np.ndarray], n: int, beta: float, L0: float
 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
     """Compute running LLC estimates"""
     H = _stack_histories(Ln_histories)
     if H is None:
         return None, None
     cmean = np.cumsum(H, 1) / np.arange(1, H.shape[1] + 1)[None, :]
-    lam = n * float(beta) * (cmean - L0)
+    lam = n * float(beta) * (cmean - L0)
     pooled = (
         np.sum(H, 0) / H.shape[0] / np.arange(1, H.shape[1] + 1)
     )  # pooled running mean
     lam_pooled = n * float(beta) * (pooled - L0)
     return lam, lam_pooled
@@
-def plot_diagnostics(
+def plot_diagnostics(
     run_dir: str,
     sampler_name: str,
     Ln_histories: List[np.ndarray],
     samples_thin: np.ndarray,
     acceptance_rates: Optional[List[np.ndarray]] = None,
     energy_deltas: Optional[List[np.ndarray]] = None,
     n: int = 1000,
     beta: float = 1.0,
     L0: float = 0.0,
     save_plots: bool = True,
 ) -> None:
-    """Generate comprehensive diagnostic plots for a sampler"""
+    """Generate diagnostics for a sampler, aligned with ArviZ best practice."""
+    # Build idata objects
+    idata_L, _ = _idata_from_L(Ln_histories)
+    idata_llc = _idata_from_llc(Ln_histories, n, beta, L0, acceptance_rates, energy=None)
@@
-    # L_n trace plots
+    # 1) L_n trace (raw)
     if Ln_histories and any(len(h) > 0 for h in Ln_histories):
         fig, ax = plt.subplots(1, 1, figsize=(8, 4))
         for i, hist in enumerate(Ln_histories):
             if len(hist) > 0:
                 ax.plot(hist, alpha=0.7, label=f"Chain {i}")
         ax.set_xlabel("Evaluation")
         ax.set_ylabel("L_n")
         ax.set_title(f"{sampler_name} L_n Traces")
         ax.legend()
         ax.grid(True, alpha=0.3)
         if save_plots:
             _finalize_figure(fig, f"{run_dir}/{sampler_name}_Ln_trace.png")
         plt.close(fig)
+        # centered L_n - L0
+        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
+        for i, hist in enumerate(Ln_histories):
+            if len(hist) > 0:
+                ax.plot(np.asarray(hist) - L0, alpha=0.7, label=f"Chain {i}")
+        ax.set_xlabel("Evaluation")
+        ax.set_ylabel("L_n - L_0")
+        ax.set_title(f"{sampler_name} Centered L_n")
+        ax.legend(); ax.grid(True, alpha=0.3)
+        if save_plots:
+            _finalize_figure(fig, f"{run_dir}/{sampler_name}_Ln_centered.png")
+        plt.close(fig)
@@
-    # Running LLC plot
+    # 2) Running LLC plot (per-chain + pooled, with ±2·SE band)
     lam, lam_pooled = _running_llc(Ln_histories, n, beta, L0)
     if lam is not None:
         fig, ax = plt.subplots(1, 1, figsize=(8, 4))
         for i in range(lam.shape[0]):
             ax.plot(lam[i], alpha=0.7, label=f"Chain {i}")
         if lam_pooled is not None:
             ax.plot(lam_pooled, "k-", linewidth=2, label="Pooled")
+        # final mean±2SE band using ESS-based estimator
+        mean_llc, se_llc, _ess = llc_mean_and_se_from_histories(Ln_histories, n, beta, L0)
+        if np.isfinite(se_llc):
+            ax.axhline(mean_llc, linestyle="--", linewidth=1)
+            ax.fill_between(np.arange(len(lam_pooled)), mean_llc-2*se_llc, mean_llc+2*se_llc, alpha=0.1)
         ax.set_xlabel("Evaluation")
-        ax.set_ylabel("Local Learning Coefficient")
-        ax.set_title(f"{sampler_name} Running LLC")
+        ax.set_ylabel("LLC = n·β·(E[Lₙ] − L₀)")
+        ax.set_title(f"{sampler_name} Running LLC")
         ax.legend()
         ax.grid(True, alpha=0.3)
         if save_plots:
             _finalize_figure(fig, f"{run_dir}/{sampler_name}_running_llc.png")
         plt.close(fig)
@@
-    # Theta trace plots (if available)
+    # 3) Theta trace plots (if available)
     idata_theta, theta_idx = _idata_from_theta(samples_thin)
     if idata_theta is not None and theta_idx:
         n_dims_to_plot = min(4, len(theta_idx))
         sel = {"theta_dim": theta_idx[:n_dims_to_plot]}
         fig, axes = plt.subplots(2, n_dims_to_plot, figsize=(12, 6), squeeze=False)
         az.plot_trace(
             idata_theta,
             var_names=["theta"],
             coords=sel,
             axes=axes,
             backend_kwargs={"constrained_layout": True},
         )
         plt.suptitle(f"{sampler_name} Parameter Traces")
         plt.tight_layout()
         if save_plots:
             _finalize_figure(fig, f"{run_dir}/{sampler_name}_theta_trace.png")
         plt.close(fig)
@@
-    # Acceptance rate plot (HMC only)
+    # 4) Acceptance rate plot (HMC only)
     if acceptance_rates is not None and any(len(acc) > 0 for acc in acceptance_rates):
         fig, ax = plt.subplots(1, 1, figsize=(8, 4))
         for i, acc in enumerate(acceptance_rates):
             if len(acc) > 0:
                 ax.plot(acc, alpha=0.7, label=f"Chain {i}")
         ax.set_xlabel("Draw")
         ax.set_ylabel("Acceptance Rate")
         ax.set_title(f"{sampler_name} Acceptance Rate")
         ax.legend()
         ax.grid(True, alpha=0.3)
-        ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="Target")
+        # reference targets from Stan (~0.8) and ChEES (~0.651)
+        ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="Target 0.8")
+        ax.axhline(0.651, color="gray", linestyle="--", alpha=0.5, label="Target 0.651")
         if save_plots:
             _finalize_figure(fig, f"{run_dir}/{sampler_name}_acceptance.png")
         plt.close(fig)
@@
-    # Energy delta histogram (MCLMC only)
+    # 5) Energy delta histogram (MCLMC only)
     if energy_deltas is not None and any(len(e) > 0 for e in energy_deltas):
         fig, ax = plt.subplots(1, 1, figsize=(8, 4))
         all_deltas = np.concatenate([e for e in energy_deltas if len(e) > 0])
         ax.hist(all_deltas, bins=50, alpha=0.7, density=True)
         ax.set_xlabel("Energy Change")
         ax.set_ylabel("Density")
         ax.set_title(f"{sampler_name} Energy Changes")
         ax.grid(True, alpha=0.3)
         if save_plots:
             _finalize_figure(fig, f"{run_dir}/{sampler_name}_energy_hist.png")
         plt.close(fig)
+
+    # 6) ArviZ-first plots for llc: rank, autocorr, ESS, and summary text
+    if idata_llc is not None:
+        # rank plot
+        fig = az.plot_rank(idata_llc, var_names=["llc"])
+        if save_plots: _finalize_figure(fig.figure, f"{run_dir}/{sampler_name}_llc_rank.png"); plt.close(fig.figure)
+        # autocorr
+        fig = az.plot_autocorr(idata_llc, var_names=["llc"])
+        if save_plots: _finalize_figure(fig[0].figure, f"{run_dir}/{sampler_name}_llc_autocorr.png"); plt.close(fig[0].figure)
+        # ESS evolution and quantile/local
+        fig = az.plot_ess(idata_llc, var_names=["llc"], kind="evolution")
+        if save_plots: _finalize_figure(fig.figure, f"{run_dir}/{sampler_name}_llc_ess_evolution.png"); plt.close(fig.figure)
+        fig = az.plot_ess(idata_llc, var_names=["llc"], kind="quantile")
+        if save_plots: _finalize_figure(fig.figure, f"{run_dir}/{sampler_name}_llc_ess_quantile.png"); plt.close(fig.figure)
+        # R-hat/ESS summary table (to console; you can persist as CSV if desired)
+        summ = az.summary(idata_llc, var_names=["llc"], kind="all", filter_vars=None)
+        print(f"[{sampler_name}] R-hat = {summ['r_hat'].item():.4f}, ESS_bulk = {summ['ess_bulk'].item():.1f}, ESS_tail = {summ['ess_tail'].item():.1f}")
```

* This makes `_idata_from_L` **used** (for raw $L_n$), and adds `_idata_from_llc` so you can use *all* of ArviZ’s diagnostics on **`llc`** directly (trace, rank, autocorr, ESS, R‑hat). R‑hat/ESS references and plots: ([python.arviz.org][3])
* If you later store **energy** into `sample_stats` for HMC, `az.plot_energy` will work out of the box. ([python.arviz.org][7])
* The acceptance plot shows both **0.651** (ChEES/BlackJAX) and **0.8** (Stan default) targets. ([blackjax-devs.github.io][9])

### 3) Fail fast when plots can’t exist (saves time)

Add to `scripts/promote_readme_images.py` (where you skip no‑PNG runs) a specific message:

```python
if not pngs and (run_dir / "metrics.json").exists():
    print("No plots saved (save_plots=False or wrong L0?). Re-run with --preset=quick or --save-plots.")
    sys.exit(2)
```

---

## C — Data sufficiency & preset tweaks

* **Chains:** Two chains (your current `quick`) is **not enough** for robust R‑hat and rank plots. Make **4 chains** the default for `quick`. Stan’s docs: *“For robust diagnostics, we recommend running 4 chains.”* ([Stan][10])

* **ESS targets:** As a heuristic, aim for **ESS ≥ 100 × chains** and $\widehat R < 1.01$ for `llc`. If you fall short, increase draws or reduce thinning; for SGLD raise steps/lower step size; for HMC increase draws/warmup/target\_accept. ([arviz-devs.github.io][5])

* **Evaluation frequency:** Your `sgld_eval_every=100` in quick is so sparse that the running‑mean curves are jagged and short. Use **10–20** for SGLD in quick so `llc` has enough points for ArviZ diagnostics.

* **Acceptance targets:** In HMC, Stan targets \~0.8 (`delta`), while BlackJAX ChEES adapts step size to \~**0.651** acceptance on average; acceptable practical window **0.6–0.9**. If you’re consistently at 0.95+, you’re taking steps too small (inefficient); if <0.5, too large (divergences likely). ([Stan][8])

**Preset patch**

```diff
diff --git a/llc/cli.py b/llc/cli.py
@@ def apply_preset(cfg: Config, preset: str) -> Config:
-            chains=2,
+            chains=4,
-            sgld_eval_every=100,
+            sgld_eval_every=20,
```

---

## Why your SGLD running‑LLC axis looked \~0.01–0.08

* With `n≈1000`, `β≈0.14`, $nβ≈140$. If you subtract the **wrong** $L_0$, $\bar L_n - L_0$ ends up \~$10^{-4}$–$10^{-3}$, so $\lambda$ lands around **0.01–0.1**. That is **exactly** the scale you saw. Fixing $L_0$ (train ERM first) restores $\lambda$ into the tens where your scalar estimator sits.

---

## Optional (but recommended) HMC energy

If you can log the *Hamiltonian energy* per draw (potential + kinetic), store it in `sample_stats["energy"]`. Then you can call `az.plot_energy(idata)` to diagnose heavy tails or poor exploration visually; this is a standard HMC diagnostic in ArviZ. ([python.arviz.org][7])

---

## Quick validation checklist

1. `llc run --preset=quick` (now **4 chains**, plots enabled).
2. Confirm log prints **“L0 at empirical minimizer”**.
3. In `runs/<id>/`: `*_running_llc.png` has a y‑axis in the tens; `*_llc_rank.png` looks roughly uniform per chain; printed `R-hat` < 1.01 and `ESS_bulk` ≥ 400 (for 4 chains).
4. `*_llc_ess_evolution.png` grows to a plateau; `*_llc_autocorr.png` decays reasonably fast.
5. HMC: acceptance sits in the **0.6–0.9** band (or fix adaptation). ([Stan][8])

---

## Closing bluntly

* The weird y‑axis wasn’t “mislabelled”—it was **the wrong $L_0$** in the new pipeline.
* Your current diagnostics underuse ArviZ. The patch above makes **`llc` a first‑class ArviZ variable** and gives you the standard tools: **rank plots, R‑hat, ESS evolution/quantile, ACF, energy**. That’s the toolkit reviewers expect. ([python.arviz.org][3])

If you want, I can also wire `sample_stats.energy` from your BlackJAX HMC state to enable `az.plot_energy` immediately; you’ll likely need to propagate per‑step Hamiltonian from the kernel (available in GHMC/NUTS state APIs). ([blackjax-devs.github.io][11])

[1]: https://python.arviz.org/en/stable/api/diagnostics.html?utm_source=chatgpt.com "Diagnostics — ArviZ 0.22.0 documentation"
[2]: https://python.arviz.org/en/stable/api/generated/arviz.plot_trace.html?utm_source=chatgpt.com "arviz.plot_trace — ArviZ 0.22.0 documentation"
[3]: https://python.arviz.org/en/stable/api/generated/arviz.rhat.html?utm_source=chatgpt.com "arviz.rhat — ArviZ 0.22.0 documentation"
[4]: https://python.arviz.org/en/stable/api/generated/arviz.plot_ess.html?utm_source=chatgpt.com "arviz.plot_ess — ArviZ 0.22.0 documentation"
[5]: https://arviz-devs.github.io/ArviZ.jl/stable/api/diagnostics/?utm_source=chatgpt.com "Diagnostics · ArviZ.jl"
[6]: https://python.arviz.org/en/stable/api/generated/arviz.plot_autocorr.html?utm_source=chatgpt.com "arviz.plot_autocorr — ArviZ 0.22.0 documentation"
[7]: https://python.arviz.org/en/stable/api/generated/arviz.plot_energy.html?utm_source=chatgpt.com "arviz.plot_energy — ArviZ 0.22.0 documentation"
[8]: https://mc-stan.org/docs/cmdstan-guide/mcmc_config.html?utm_source=chatgpt.com "MCMC Sampling using Hamiltonian Monte Carlo - Stan"
[9]: https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/chees_adaptation/index.html?utm_source=chatgpt.com "blackjax.adaptation.chees_adaptation"
[10]: https://mc-stan.org/docs/2_25/cmdstan-guide/mcmc-intro.html?utm_source=chatgpt.com "4 MCMC Sampling | CmdStan User's Guide"
[11]: https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/ghmc/index.html?utm_source=chatgpt.com "blackjax.mcmc.ghmc"
