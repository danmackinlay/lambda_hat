Yes—build a **sampler‑agnostic diagnostics layer** that uses the same artifacts for both methods and then adds method‑specific panels on top. ArviZ is still the backbone, but you need a couple of extras so VI is *meaningfully* comparable to MCMC.

Below is a spec you can implement once and use for all engines (HMC/MCLMC/SGLD/VI).

---

## 1) What to record for **every** method (the “common contract”)

For each kept draw (or draw batch) record:

1. **A draw of parameters** `θ` (optionally thinned / projected).
2. **Full‑data loss** on that draw, `L_n(θ)` (mean over the dataset).
3. **Unnormalized target log‑density** for your *local tempered* target
   `log p̃(θ) = - n β* L_n(θ) - ½ γ ||θ - θ*||²`.
   (Constant terms don’t matter—use the same definition for all methods.)
4. **Wall‑clock cumulative time** and **work counters** (e.g., full‑loss evals, minibatch grads) to compute efficiency.
5. **Sampler tag**: one of `{"hmc","mclmc","sgld","vi"}`.

Store all of that in a single ArviZ `InferenceData`:

* `posterior`: draws of the parameters you want to inspect.
* `sample_stats`:

  * `log_p_tilde` (float, shape = draws×chains)
  * `L_n` (float)
  * `cumulative_time_sec` (float)
  * `cumulative_work_fge` (float)  ⟵ “function‑gradient equivalents”
  * `sampler` (string attr on the idata)

For **VI only**, also store:

* `log_q` (the variational log‑density of the draw)
* any training scalars you care about at evaluation points: `elbo`, `radius2`, `entropy_pi`, etc. Put them in `sample_stats`.

For **HMC**, also store: `diverging`, `energy`, etc.
For **SGLD**, no extras are required.

> Why this matters
> If you always log `L_n`, `log_p_tilde`, and work/time, you can define **one** set of comparison metrics (below) that works across all samplers. You do **not** need R‑hat or acceptance to compare VI vs MCMC; those remain MCMC‑only panels.

---

## 2) The **unified scoreboard** (works for both)

Compute these from the captured traces at the **same evaluation checkpoints**:

1. **Point estimate of LLC**
   [
   \hat\lambda = n,\beta^* \left(\overline{L_n}-L_n(\theta^*)\right)
   ]
   where (\overline{L_n}) is the mean of `L_n` over the kept draws.

2. **Monte‑Carlo SE of (\hat\lambda)**

   * For MCMC: (\text{MCSE}(\hat\lambda) = n\beta^* \sqrt{\operatorname{Var}(L_n)/\text{ESS}}) using bulk ESS from the `L_n` series.
   * For VI (IID draws): (\text{MCSE}(\hat\lambda) = n\beta^* ,\text{sd}(L_n)/\sqrt{N}).
     This makes variance comparisons **fungible** without pretending R‑hat means anything for VI.

3. **Work‑normalized variance (WNV)**
   [
   \text{WNV} = \operatorname{Var}(L_n) \times \text{(wall‑time or FGEs)}
   ]
   Report both **per second** and **per FGE**. This lets you rank algorithms by “precision per budget”.

4. **Importance diagnostics to the *true local target*** (MCMC & VI)

   * Define log‑weights ( \ell w_i = \log p̃(\theta_i) - \log q(\theta_i) ).

     * For **MCMC**, set (\log q \equiv \text{const}) so weights are all equal.
     * For **VI**, compute (\log q) from the variational family.
   * Run **PSIS** on those weights and report (\hat{k}) (Pareto shape).
     Thresholds: ( \hat{k} < 0.5) good; (0.5!-!0.7) ok; (>0.7) trouble.
     This is your **bias proxy** for VI and a sanity check for all methods.
   * Also report **SNIS‑reweighted** LLC:
     [
     \hat\lambda_{\text{SNIS}} = n\beta^*\left(\frac{\sum w_i L_n(\theta_i)}{\sum w_i} - L_n(\theta^*)\right)
     ]
     and its MCSE using PSIS‑ESS. The gap (\hat\lambda_{\text{SNIS}}-\hat\lambda) is an actionable VI bias indicator.

5. **Stability across seeds**
   Keep a small number of independent seeds per method. Show mean ± sd of (\hat\lambda), MCSE, WNV, and (\hat{k}) across seeds.

These five numbers let you compare **precision, cost, and (approximate) bias** for MCMC and VI on the same footing.

---

## 3) Method‑specific panels (keep them, but segregate)

* **MCMC‑only:** R‑hat, (bulk/tail) ESS, E‑BFMI, divergence histograms, energy plots.
* **VI‑only:** ELBO trajectory, `radius2` (in whitened coords), mixture entropy / responsibilities, and (if you use a control variate) the variance reduction ratio.

Put them in separate tabs/sections so nobody tries to compare R‑hat to ELBO.

---

## 4) One logging surface: ArviZ **plus** TensorBoard tags

* **Always write** an ArviZ `.nc`. Everything above can be derived from it.
* **Also log** the same metrics to TensorBoard under a consistent namespace:

  * `common/ln_mean`, `common/ln_sd`, `common/mcse_lambda`, `common/wnv_time`, `common/wnv_fge`, `common/psis_k`, `common/psis_ess`
  * `mcmc/rhat_ln`, `mcmc/ess_bulk_ln`, `mcmc/ebfmi`, `mcmc/divergences`
  * `vi/elbo`, `vi/logq_mean`, `vi/radius2`, `vi/entropy_pi`, `vi/cv_gain`
* **Stop rules (shared):** stop when `common/mcse_lambda <= τ` *and* (for VI) `common/psis_k < 0.7`. This gives a fungible convergence criterion.

---

## 5) Minimal implementation changes to your codebase

You don’t have to rewrite ArviZ, and you don’t have to give VI fake chains. Do this:

1. **At each evaluation point** (for *all* samplers), compute and store:

   * `L_n(θ)` and `log_p_tilde(θ)`.
   * For VI **only**, also `log_q(θ)` (exact under your mixture‑of‑factor‑analyzers).

2. **Add a tiny PSIS utility** that takes `log_weight = log_p_tilde - log_q` (or zeros for MCMC) and returns:

   * (\hat{k}), PSIS weights, PSIS‑ESS. (20 lines if you already use a stats lib; otherwise a small Pareto fit + smoothing.)

3. **Unify the work counters** you already track (minibatch grads, full‑loss evals, HVPs) into `sample_stats.cumulative_work_fge`. For VI, increment by 1 per minibatch gradient step and by 1 per **full‑loss** eval used in diagnostics / plugin estimate.

4. **Compute the scoreboard** once per evaluation and write it both to:

   * `idata.attrs['metrics']` (and/or a `metrics.json`), and
   * TensorBoard with the tags above.

5. **Keep MCMC extras and VI extras** as method‑specific TB tags; don’t try to shoehorn them into the common scoreboard.

---

## 6) What *not* to do

* Don’t report **R‑hat** or **ESS** for VI—IIDs from (q) will make them look perfect and they tell you nothing about **approximation** error.
* Don’t compare raw wall‑times without **WNV** or MCSE—precision per budget is the only fair comparison.
* Don’t skip **PSIS** for VI. ELBO is a training objective; (\hat{k}) is a diagnostic about *mismatch to the actual target you care about*.

---

## 7) Quick formulas you’ll need (for clarity)

* (\displaystyle \hat\lambda = n\beta^*\big(\overline{L_n} - L_n(\theta^*)\big))
* (\displaystyle \text{MCSE}(\hat\lambda) = n\beta^*\times
  \begin{cases}
  \sqrt{\operatorname{Var}(L_n)/\text{ESS}} & \text{(MCMC)}[2pt]
  \text{sd}(L_n)/\sqrt{N} & \text{(VI)}
  \end{cases})
* (\displaystyle \hat\lambda_{\text{SNIS}} = n\beta^*\left(\frac{\sum w_i L_n(\theta_i)}{\sum w_i} - L_n(\theta^*)\right),\quad w_i \propto \exp(\log p̃_i - \log q_i))

---

## 8) Deliverables checklist (so this actually happens)

* [ ] Add `log_p_tilde`, `L_n`, and (for VI) `log_q` to `sample_stats`.
* [ ] Add PSIS utility; log `common/psis_k` and PSIS‑ESS.
* [ ] Compute `common/mcse_lambda`, `common/wnv_time`, `common/wnv_fge` at every eval.
* [ ] Write one `.nc` per run; mirror scoreboard into TensorBoard.
* [ ] Keep MCMC‑only and VI‑only panels in separate TB namespaces.
* [ ] Gate “converged” on `mcse_lambda<=τ` (+ `psis_k<0.7` for VI).

Adopt that contract and both families of methods become directly comparable on: **precision, cost, and mismatch to the local tempered target**. That’s the only comparison that matters for your LLC workflow.
