Short version: you can do \~everything you want with the current repo, but two tiny bits of glue will make it painless:

1. add a **dimension sweep** over `target_params` (that’s the knob that fixes the parameter count), and
2. make the pipeline **log WNV** (work‑normalised variance) so the sweep summary CSV contains both **WNV‑time** and **WNV‑grad** columns.

Below I give drop‑in diffs and the exact commands to run. I’ll also show one plotting snippet that ingests the sweep CSV and produces the comparisons you asked for.

---

## What’s already there (and what’s missing)

* A sweep harness exists (`main.py sweep` → parallel backends, seeds, CSV output). It builds a worklist from `llc.experiments.sweep_space()` and writes `llc_sweep_results.csv`.
* The pipeline already records **per‑sampler wall‑clock (warmup/sampling) and “gradient‑equivalent” work counters** into `metrics.json` (e.g. `hmc_n_leapfrog_grads`, `sgld_n_steps`, `mclmc_n_steps`).
* LLC mean/SE/ESS are computed from the $L_n$ histories using ArviZ bulk ESS (the SE is the right thing to square to get a variance estimate).

**Missing bit:** the pipeline doesn’t currently *fill* the WNV fields (even though your gallery/README talk about them), so the sweep CSV won’t include them unless we compute & log them when a run finishes. The raw ingredients are all there (SE, sampling time, work counters).

---

## 1) Add a dimension sweep (over `target_params`)

You control total parameter count by setting `Config.target_params`; the pipeline then infers widths to hit that size (while `hidden` is kept only for backward compatibility). The sweep module already sweeps over depth/width/activation/etc.; add one more sweep entry.

**Edit `llc/experiments.py` → `sweep_space()`** (append this new block at the top of the `"sweeps"` list):

```python
# Architecture size (parameter count) sweep
{"name": "dim", "param": "target_params",
 "values": [500, 1_000, 2_000, 5_000, 10_000]},
```

The function already returns a dict with `"base"` and `"sweeps"`, and the sweep builder will emit configs across seeds for each value.

> Why `target_params` and not `hidden`? Because `target_params` explicitly fixes **dimension $d$** and lets the repo’s width‑inference do the bookkeeping, which is what you asked for (grow up to \~10k parameters) without fiddling layer shapes by hand. The pipeline does call the width inference when `widths=None` and `target_params` is set. (You already rely on this in normal runs.)

---

## 2) Log WNV (time and gradient‑equivalent)

Define WNV as:

* `WNV_time = Var[ λ̂ ] × t_sampling`, and
* `WNV_grad = Var[ λ̂ ] × (gradient‑equivalent sampling work)`,

with `Var[ λ̂ ] = (llc_se)**2` from the ArviZ ESS‑aware SE you already compute,
`t_sampling` from `stats.t_*_sampling`, and the work from the counters you already track (SGLD steps ≈ minibatch grads; HMC ≈ leapfrog grad evals; MCLMC ≈ integrator steps).

**Edit `llc/pipeline.py`** where you currently compute and store metrics per sampler (look for the existing `all_metrics.update({...})` blocks for `sgld`, `hmc`, `mclmc`). Add the WNV fields right after you compute each sampler’s `*_llc_se`:

```python
def _wnv(se, t_sampling, work_sampling):
    var = float(se) * float(se)
    return float(var * float(t_sampling)), float(var * float(work_sampling))

# --- SGLD ---
# (existing) llc_sgld_mean, se_sgld, ess_sgld computed above
# Approximate sampling-only work by splitting warmup/sample proportionally to steps:
sgld_sample_frac = (
    max(0, cfg.sgld_steps - cfg.sgld_warmup) / max(1, cfg.sgld_steps)
)
sgld_work_sampling = float(stats.n_sgld_minibatch_grads) * sgld_sample_frac
sgld_wnv_time, sgld_wnv_grad = _wnv(se_sgld, stats.t_sgld_sampling, sgld_work_sampling)
all_metrics.update({
    "sgld_wnv_time": sgld_wnv_time,
    "sgld_wnv_grad": sgld_wnv_grad,
})

# --- HMC ---
# (existing) se_hmc already computed; HMC counts sampling grads explicitly
hmc_work_sampling = float(stats.n_hmc_leapfrog_grads)
hmc_wnv_time, hmc_wnv_grad = _wnv(se_hmc, stats.t_hmc_sampling, hmc_work_sampling)
all_metrics.update({
    "hmc_wnv_time": hmc_wnv_time,
    "hmc_wnv_grad": hmc_wnv_grad,
})

# --- MCLMC ---
mclmc_work_sampling = float(stats.n_mclmc_steps)
mclmc_wnv_time, mclmc_wnv_grad = _wnv(se_mclmc, stats.t_mclmc_sampling, mclmc_work_sampling)
all_metrics.update({
    "mclmc_wnv_time": mclmc_wnv_time,
    "mclmc_wnv_grad": mclmc_wnv_grad,
})
```

* The SGLD split is the same proportional trick you already use for **time** when subtracting eval overhead (you apportion the net sampling wall‑clock by `warmup/num_steps`). It’s a mild approximation, but consistent with your current accounting; if you want exact sampling‑only minibatch counts, wire **separate warmup vs sampling counters** into the SGLD runner later. Right now SGLD bumps `n_sgld_minibatch_grads` by 1 per step and records warmup/sampling *time* split only.
* HMC already separates warmup grads and sampling grads (warmup via adaptation extras → `n_hmc_warmup_leapfrog_grads`, sampling via the `work_bump(L+1)` during draws), so the sampling work is exact.
* MCLMC uses “steps” as its unit of work (that’s how you designed the counter); we keep that definition. It’s recorded via `work_bump(1)` per step.

Your artifact gallery already expects to show WNV if present, so these keys will also light up the HTML summary automatically. (The metrics save happens right afterwards.)

---

## 3) Put WNV into the sweep CSV

Your `sweep` command currently writes a narrow summary: means/stdevs of LLC across seeds. To analyse WNV vs dimension, extend the writer to **pull `metrics.json`** from each finished run and append relevant fields.

**Edit `main.py → handle_sweep_command`**, after `results = executor.map(...)`, before the “Save results summary” section: load each run’s `metrics.json` if `run_dir` is present, and build a *long* dataframe (one row per sampler per run) with the columns you care about.

Minimal patch sketch:

```python
# After 'results = executor.map(...)' and any Modal artifact extraction
rows = []
for r in results:
    run_dir = r.get("run_dir")
    cfg = r.get("config", {})  # include if you return it; otherwise join later
    if not run_dir:
        continue
    import json, os
    metrics_path = os.path.join(run_dir, "metrics.json")
    config_path = os.path.join(run_dir, "config.json")
    try:
        with open(metrics_path) as f:
            M = json.load(f)
        with open(config_path) as f:
            C = json.load(f)
    except Exception:
        continue

    # Which samplers ran?
    for s in ("sgld", "hmc", "mclmc"):
        if f"{s}_llc_mean" not in M:
            continue
        rows.append({
            # sweep keys
            "sweep": "dim",
            "target_params": C.get("target_params"),
            "depth": C.get("depth"),
            "activation": C.get("activation"),
            "sampler": s,
            "seed": C.get("seed"),
            # accuracy
            "llc_mean": M.get(f"{s}_llc_mean"),
            "llc_se": M.get(f"{s}_llc_se"),
            "ess": M.get(f"{s}_ess"),
            # cost
            "t_sampling": M.get(f"{s}_timing_sampling"),
            "work_grad": (
                M.get(f"{s}_n_leapfrog_grads")
                or M.get(f"{s}_n_steps")  # sgld/mclmc name
                or 0
            ),
            # efficiency (the ones you asked for)
            "wnv_time": M.get(f"{s}_wnv_time"),
            "wnv_grad": M.get(f"{s}_wnv_grad"),
            # bookkeeping
            "run_dir": run_dir,
        })

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("llc_sweep_results.csv", index=False)
print("Saved llc_sweep_results.csv with WNV fields.")
```

Now the CSV has exactly what you want for plotting and tabulation.

---

## 4) Run it

Parallel local sweep with 3 seeds:

```bash
uv run python main.py sweep \
  --backend=local --workers=4 \
  --n-seeds=3 --preset=quick
```

That will run the new **“dim”** sweep across `target_params ∈ {500, 1k, 2k, 5k, 10k}`, saving per‑run artifacts (including `metrics.json`) and a **long** `llc_sweep_results.csv`.

If you want to lock **budget fairness** across samplers as dimension grows, do one of these:

* **Time‑budgeted** fairness: leave your per‑sampler draw/step counts fixed and compare **WNV\_time** across samplers; it directly answers “variance per second”.
* **Gradient‑budgeted** fairness: set HMC `eval_every` and SGLD/MCLMC `eval_every` to achieve comparable **logging cadence per gradient‑equivalent** while keeping total work fixed. (You already track per‑step grads; you can pick a global `K_grad` and set `eval_every = ceil(K_grad / grads_per_step)` per sampler.)

---

## 5) Analyse: DataFrame + plots

Once the sweep finishes:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("llc_sweep_results.csv")

# Sanity: ensure numeric dtype
for col in ["target_params","llc_mean","llc_se","wnv_time","wnv_grad","t_sampling","work_grad"]:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Aggregate over seeds (mean ± SE across seeds)
grp = (df
  .groupby(["sampler","target_params"], as_index=False)
  .agg(llc_mean=("llc_mean","mean"),
       llc_se=("llc_se","mean"),
       wnv_time=("wnv_time","mean"),
       wnv_grad=("wnv_grad","mean"),
       t_sampling=("t_sampling","mean"),
       work_grad=("work_grad","mean"),
       n=("llc_mean","count"))
)
grp["llc_mean_se_over_seeds"] = grp["llc_se"] / np.sqrt(grp["n"].clip(lower=1))

# --- Plot WNV vs dimension (lower is better) ---
def lineplot(x, y, hue, title, ylabel):
    plt.figure(figsize=(7,4))
    for s, sub in grp.pivot(index="target_params", columns="sampler", values=y).items():
        plt.plot(sub.index, sub.values, marker="o", label=s)
    plt.xlabel("Parameter count (target_params)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

lineplot("target_params","wnv_time","sampler","WNV (time-normalised) vs dimension","Var(λ̂) × seconds")
lineplot("target_params","wnv_grad","sampler","WNV (gradient-normalised) vs dimension","Var(λ̂) × grad-equiv")
```

(If you prefer to see uncertainty bands on `llc_mean`, use `grp["llc_mean_se_over_seeds"]` in an errorbar plot; **don’t** put error bars on WNV unless you bootstrap them—WNV already depends on a variance estimate.)

---

## A few judgement calls (so you don’t trip later)

* **SGLD warmup work**: today you only have a single `n_sgld_minibatch_grads` counter. I split it to sampling using the same fraction you used for time (warmup/steps). If you want exact sampling‑only grad counts, thread distinct warmup vs sampling `work_bump`s through `run_sgld_chain` and record them separately. Not urgent; the proportional split is consistent with your current timing split.

* **HMC work units**: you correctly count \~`L+1` gradient evals per draw (Velocity‑Verlet). You already include warmup grads in a separate counter during adaptation; I used **sampling‑only** for WNV. That’s what you want.

* **MCLMC work units**: you defined one “step” as one work unit. That’s fine; just remember the “grad‑equivalent” interpretation is method-specific. You are explicit about this in the manifest/galleries, so no ambiguity for you or readers.

* **Dimension setting**: using `target_params` is the right abstraction. Don’t try to back‑solve widths by hand for each depth: the repo already does that inference, and the pipeline prints the actual parameter dimension at run start so you can verify it.

---

## TL;DR action list (copy/paste)

1. **Add the “dim” sweep** over `target_params` to `llc/experiments.py` → `sweep_space()`.
2. **Compute & log WNV** in `llc/pipeline.py` right after the `*_llc_se` is known for each sampler, using the timing/work counters you already persist.
3. **Extend the sweep writer** in `main.py → handle_sweep_command` to read each run’s `metrics.json` and write a long `llc_sweep_results.csv` with `sampler`, `target_params`, `wnv_time`, `wnv_grad`, etc.
4. Run:

   ```bash
   uv run python main.py sweep --backend=local --workers=4 --n-seeds=3 --preset=quick
   ```
5. Plot using the snippet above.

That’s it. After (2)–(3), you’ll have a single CSV per sweep with **accuracy (λ̂, SE)** and **efficiency (WNV‑time, WNV‑grad)**, ready to slice by dimension and sampler.
