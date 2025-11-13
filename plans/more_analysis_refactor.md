You’re right: the `if sampler_name == "vi"` in `analysis.py` is a smell, but it’s also telling you exactly what’s missing from your abstractions. The good news is you can clean this up with one or two small patterns, not a grand redesign.

Let me outline the core problem and then suggest concrete refactors.

---

## What’s actually wrong with `analysis.py` right now?

In the current implementation:

* `analyze_traces(...)` assumes **“I always get Ln and I know how to turn Ln into λ”**, *except* when `sampler_name == "vi"`, in which case it special-cases and reads `work["lambda_hat_mean"]`.
* `_compute_metrics_from_idata(...)` also branches on `sampler_name == "vi"` to decide whether to use ArviZ ESS/R̂ or treat samples as IID.

So `analysis.py` currently knows:

1. How each sampler *semantically* defines its “samples” (Markov vs IID).
2. How each sampler’s raw traces should be turned into λ̂.

That’s the leaky abstraction: **sampling semantics live in analysis.**

You want `analysis` to do “given a time series of LLC values + some work counters, make metrics and plots”, not “oh, and also derive LLC from Ln, except for samplers that don’t actually give Ln”.

---

## Target abstraction: analysis works on a *uniform* schema

The minimal abstraction that keeps things maintainable:

**“Every sampler must hand analysis a `llc` series already in the traces.”**

Then:

* `analysis.analyze_traces` no longer needs to know how to compute λ̂ from Ln for different samplers.
* The only remaining sampler-specific thing is “are these draws IID or Markov?”, which is a much smaller and more stable distinction.

Concretely:

* For HMC/SGLD/MCLMC: you still compute
  `llc = n * beta * (Ln - L0)`
  but you do it *inside the sampler layer* (or just before analysis), not in `analysis.py`.
* For VI: you never try to derive λ̂ from Ln at all; you construct a synthetic `llc` array using the plug-in estimator you already computed.

Then `analysis.py` becomes:

```python
def analyze_traces(traces, L0, n_data, beta, ..., sampler_flavour: str):
    # 1) Extract llc
    if "llc" in traces:
        llc_values_np = np.asarray(traces["llc"])
    else:
        # backwards-compat fallback ONLY
        Ln_values = ...
        llc_values_np = float(n_data) * float(beta) * (Ln_post_warmup - L0)

    # 2) Build idata from llc_values_np (and maybe L)
    # 3) Call _compute_metrics_from_idata(idata, llc_values_np, timings, work, sampler_flavour)
```

Note that `sampler_name` is now only used to interpret **dependence structure**, not numeric meaning.

---

## Concretely, how to get there in your code

### 1. Push λ̂ computation to the sampler layer

Right now, the VI plug-in λ̂ lives in `work["lambda_hat_mean"]`, and HMC/SGLD/MCLMC only provide `Ln`.

Refactor so **all** samplers do this:

* **In `run_hmc` / `run_sgld` / `run_mclmc`**:
  After you have `traces["Ln"]` and before returning, compute:

  ```python
  Ln = traces["Ln"]          # (C, T)
  llc = float(n_data) * float(beta) * (Ln - L0)
  traces["llc"] = llc
  ```

  (You know `n_data` and `beta` in `sampling_runner.run_sampler`; just pass them in or compute there, then attach `llc` to the traces dictionary before calling analysis.)

* **In `run_vi`** (or just after it in `sampling_runner.run_sampler`):

  * Use the VI plug-in λ̂ to fabricate a consistent `llc` series:

    ```python
    lambda_hat_mean = work["lambda_hat_mean"]  # scalar
    chains = cfg.sampler.chains
    T = num_effective_samples  # steps // eval_every

    llc = jnp.full((chains, T), lambda_hat_mean, dtype=...)
    traces["llc"] = llc
    ```

  If you have per-chain λ̂’s, even better: assign row-wise instead of duplicating.

Once you do this, **analysis no longer needs the special VI case to reconstruct λ̂**.

### 2. Simplify `analyze_traces`: “if llc exists, trust it”

Change the front of `analyze_traces` to:

```python
def analyze_traces(..., sampler_name: str | None = None):
    # 1. Get llc
    if "llc" in traces:
        llc_values = traces["llc"]
        # canonicalize to (C,T)
        if llc_values.ndim == 1:
            llc_values = llc_values[None, :]
        chains, draws = llc_values.shape
        Ll = None
    else:
        # fallback path: compute from Ln
        Ln_values = traces["Ln"]
        if Ln_values.ndim == 1:
            Ln_values = Ln_values[None, :]
        chains, draws = Ln_values.shape
        Ll = Ln_values

        llc_values = float(n_data) * float(beta) * (Ln_values - L0)

    # 2. Warmup slicing etc.
    llc_post = llc_values[:, warmup:]
    draws_post = draws - warmup

    # 3. Posterior vars
    posterior_data = {"llc": np.array(llc_post)}
    if Ll is not None:
        posterior_data["L"] = np.array(Ll[:, warmup:])
```

Now the VI special-case in `analyze_traces` disappears entirely: VI sets `traces["llc"]`, analysis happily consumes it.

### 3. Replace `sampler_name` by a sampler *flavour*

You currently encode “this is VI” by string equality:

```python
if sampler_name == "vi":
    # treat as IID
else:
    # run ArviZ ESS / Rhat
```

That’s brittle and exactly the kind of dependency you don’t want to spread.

Instead define a tiny, explicit flavour:

* In `sampling_runner.run_sampler`, add a key:

  ```python
  sampler_flavour = "iid" if sampler_name == "vi" else "markov"
  return {
      "traces": traces,
      "timings": timings,
      "work": work,
      "sampler_config": ...,
      "beta": beta,
      "gamma": gamma,
      "sampler_flavour": sampler_flavour,
  }
  ```

* In `entrypoints/sample.py`, pass that through:

  ```python
  result = run_sampler(...)
  sampler_flavour = result.get("sampler_flavour", "markov")
  metrics, idata = analyze_traces(
      traces, L0, n_data, beta,
      timings=timings,
      work=work,
      sampler_flavour=sampler_flavour,
  )
  ```

* In `_compute_metrics_from_idata`, switch on *flavour*:

  ```python
  def _compute_metrics_from_idata(..., sampler_flavour: str | None):
      ...
      if sampler_flavour == "iid":
          # treat draws as IID: ESS = C*T, Rhat NaN
      else:
          # MCMC path: ArviZ ESS/Rhat
  ```

Now if you add a Laplace sampler, bootstraps, or anything else that gives IID λ̂ draws, you mark it as `"iid"` and you’re done. `analysis` no longer cares about specific sampler names.

This is the same **“strategy / tag”** pattern you see in things like Stan/ArviZ integrations: there’s a high-level “this is MCMC” vs “this is posterior predictive / VI / importance samples” flag, not a switch on every individual algorithm.

---

## How this compares to other tools’ patterns

Other ecosystems that juggle VI and MCMC effectively do something equivalent to:

* **TFP / Edward2**: “posterior objects” vs “samples”; VI gives you a `tfp.distributions.Distribution`, MCMC gives you a `SampleChain`. Downstream plotting tools work with *samples of some named variable*, not directly with the inference method.
* **PyMC + ArviZ**: ArviZ works with an `InferenceData` whose `posterior["theta"]` is just an array. How that was obtained (NUTS, ADVI, Laplace) lives upstream; analysis basically assumes “here’s a chain-shaped array, maybe Markov, maybe not” and optionally uses ESS / R̂ when that makes sense.

You’re already halfway there:

* You standardised on ArviZ `InferenceData`.
* You have a `traces` dict with common keys (`Ln`, `cumulative_fge`, etc.).
* You’ve added VI by jamming its semantics into the MCMC path.

The two tweaks above (“put llc in traces; tag sampler flavour as iid/markov”) line you up with that pattern, without inventing a whole new abstraction layer.

---

## Is it worth doing more abstraction than that?

Given your “simplicity > premature abstraction” rule, I would *not* introduce:

* A full hierarchy of `SamplerAdapter` classes per algorithm.
* Separate analysis modules per sampler.

You don’t need it. The two changes above:

1. Move λ̂ construction into the sampling layer, and
2. Replace `sampler_name == "vi"` with a tiny `sampler_flavour` tag,

keep the code simple, make `analysis.py` mostly sampler-agnostic, and avoid the combinatorial explosion of branches as you add samplers.

If you want one more small nicety later, you could split `analyze_traces` into:

* `build_idata(traces, ...) -> (idata, llc_values_np)`
* `_compute_metrics_from_idata(idata, llc_values_np, ...)`

but that’s just readability, not a new abstraction.

---

## Summary of what I’d actually change now

Minimal, high-leverage refactors:

1. **Have every sampler write `traces["llc"]` itself**

   * HMC/SGLD/MCLMC: derive from Ln and L0 upstream.
   * VI: populate from `lambda_hat_mean` (or per-chain values).

2. **Change `analyze_traces` to prefer `traces["llc"]`**

   * Only derive from Ln as a backwards-compat fallback.

3. **Add a `sampler_flavour` tag (`"markov"` vs `"iid"`)**

   * `_compute_metrics_from_idata` switches on that, instead of hard-coding `"vi"`.

That keeps the visualisation logic non-intrusive, makes it easy to add more samplers (or more VI flavours) without touching `analysis.py` again, and matches the design patterns you see in ArviZ-style ecosystems.
