Great—most of the plumbing you need is already in the repo:

* Every sampler run writes convergence plots into `runs/targets/<tid>/run_<sampler>_<rid>/diagnostics/` (trace, rank, energy, combined “LLC vs FGEs/Time”, and WNV).
* You already ship a tiny “promotion” CLI (`lambda-hat-promote`) that can copy the newest plot per sampler into a stable assets folder and optionally emit a README-ready gallery snippet.

Below is a drop-in way to make **Snakemake** generate and then **promote** those convergence graphs automatically.

---

# 1) Add VI to the promoter config (one-liner)

```yaml
# lambda_hat/conf/promote.yaml
runs_root: runs
samplers: sgld,hmc,mclmc,vi    # <- include VI
outdir: assets
plot_name: trace.png
```

(This file is already in-tree; just add `vi`.)

---

# 2) Teach the Snakefile about “promotion” jobs

Append the following to your `Snakefile` (near the top where other presets are loaded):

```python
# Load promoter config
PROM = OmegaConf.load(CONF / "promote.yaml")
SAMPLERS = sorted({s for _, s, _, _ in RUNS})    # e.g. ['hmc','mclmc','sgld','vi']

def _sampler_run_analysis(wc):
    # All analysis.json for a given sampler across all targets/runs
    return [f"{STORE}/targets/{tid}/run_{wc.sampler}_{rid}/analysis.json"
            for (tid, s, rid, _cfg) in RUNS if s == wc.sampler]
```

Now add **three tiny rules**:

### A) Promote the classic trace plot (per sampler)

```python
rule promote_trace:
    input:
        _sampler_run_analysis
    output:
        png = lambda wc: f"{PROM.outdir}/trace_{wc.sampler}.png"
    params:
        samplers=lambda wc: wc.sampler,
        plot="trace.png"
    wildcard_constraints:
        sampler="|".join(SAMPLERS)
    shell:
        r"""
        mkdir -p {PROM.outdir}
        uv run lambda-hat-promote single \
          --runs-root {STORE} \
          --samplers {params.samplers} \
          --outdir {PROM.outdir} \
          --plot-name {params.plot}
        mv {PROM.outdir}/{params.samplers}.png {output.png}
        """
```

### B) Promote the combined “LLC vs FGEs/Time” plot (per sampler)

`entrypoints/sample.py` already writes `diagnostics/llc_convergence_combined.png` for each run; this copies the newest one per sampler:

```python
rule promote_convergence:
    input:
        _sampler_run_analysis
    output:
        png = lambda wc: f"{PROM.outdir}/convergence_{wc.sampler}.png"
    params:
        samplers=lambda wc: wc.sampler,
        plot="llc_convergence_combined.png"
    wildcard_constraints:
        sampler="|".join(SAMPLERS)
    shell:
        r"""
        mkdir -p {PROM.outdir}
        uv run lambda-hat-promote single \
          --runs-root {STORE} \
          --samplers {params.samplers} \
          --outdir {PROM.outdir} \
          --plot-name {params.plot}
        mv {PROM.outdir}/{params.samplers}.png {output.png}
        """
```

### C) Optional README gallery (one shot across samplers)

```python
rule promote_gallery:
    input:
        expand(f"{STORE}/targets/{{tid}}/run_{{sampler}}_{{rid}}/analysis.json", tid=[t for t,_,_,_ in RUNS], sampler=[s for _,s,_,_ in RUNS], rid=[r for _,_,r,_ in RUNS])
    output:
        md = f"{PROM.outdir}/gallery_trace.md"
    params:
        samplers=",".join(SAMPLERS),
        plot="trace.png"
    shell:
        r"""
        mkdir -p {PROM.outdir}
        uv run lambda-hat-promote gallery \
          --runs-root {STORE} \
          --samplers {params.samplers} \
          --outdir {PROM.outdir} \
          --plot-name {params.plot} \
          --snippet-out {output.md}
        """
```

Finally, include these in your top-level `rule all` so a single `snakemake` run builds targets, runs samplers, **and** promotes the plots:

```python
rule all:
    input:
        [f"{STORE}/targets/{tid}/run_{sampler}_{rid}/analysis.json" for tid,sampler,rid,_ in RUNS],
        expand(f"{PROM.outdir}/trace_{{sampler}}.png", sampler=SAMPLERS),
        expand(f"{PROM.outdir}/convergence_{{sampler}}.png", sampler=SAMPLERS),
        f"{PROM.outdir}/gallery_trace.md"
```

---

# 3) Why this works (and what you already had)

* Plot **generation** is already done in `lambda_hat/entrypoints/sample.py` via `create_arviz_diagnostics`, `create_combined_convergence_plot`, and `create_work_normalized_variance_plot`, so the Snakemake promotion step only needs to **copy** the newest run per sampler into `assets/`.
* The `lambda-hat-promote` CLI (`entrypoints/promote.py` + `promote/core.py`) already implements “single” and “gallery” modes and discovers the “newest run” by mtime of `diagnostics/<plot_name>`. We’re just calling it from rules, with proper inputs so scheduling waits for runs to finish.

---

# 4) Small paper cuts to avoid

* Make sure `samplers:` in `lambda_hat/conf/promote.yaml` includes **vi**, otherwise the gallery will omit VI.
* Keep `plot_name` aligned with what you actually save (`trace.png`, `llc_convergence_combined.png`, `llc_wnv.png`). If you want WNV public too, add a third rule identical to **B** but with `plot="llc_wnv.png"` and output `wnv_{sampler}.png`.

That’s it—after this, a single:

```bash
uv run snakemake -j 4
```

will build targets → run samplers → write diagnostics → **promote** trace & convergence graphs into `assets/` and emit a tiny gallery markdown you can paste in a README.
