High level: Parsl gives you *Python-native DAGs* that submit work to SLURM/PBS/etc. You keep your existing `lambda_hat` library (targets + samplers) and replace Snakemake with a small set of Parsl apps and a “driver” script that builds the (N \times M) grid and dispatches jobs.

I’ll outline:

1. What you get vs Snakemake
2. Core architecture for your repo
3. Sketch of actual code (no hand-waving)
4. How this plays with SLURM/PBS
5. Maintenance trade-offs for you

I’ll assume we keep **OmegaConf + your dataclasses as the only config system** (no Hydra, no Metaflow config).

---

## 1. What Parsl buys you over Snakemake

Snakemake:

* Excellent declarative DAG
* Slightly painful dynamic parameter sweeps / ad-hoc overrides
* Good SLURM/PBS support via profiles, but you’re fighting its YAML/CLI config dance

Parsl:

* **DAG is just Python code**: a function call graph with futures.
* Execution backends for **SLURM, PBSPro, LSF, Cobalt, HTCondor**, etc via providers.
* You write:

  ```python
  @python_app
  def build_target(cfg_dict): ...

  @python_app
  def run_sampler(target_artifact, sampler_cfg): ...
  ```

  and Parsl handles job submission / dependencies.

Downside: you take over more of the bookkeeping (IDs, manifests, retries) that Snakemake was doing, but your repo already has that logic implemented in Python (`lambda_hat.target_artifacts`, `entrypoints`, etc.), so you’re halfway there.

Given you’re the only dev and already comfortable with Python, it’s a good fit.

---

## 2. Architecture tailored to *this* repo

You already have:

* Entry points:

  * `lambda-hat-build-target` → `entrypoints/build_target.py`
  * `lambda-hat-sample` → `entrypoints/sample.py`
* Clear artifact story: `runs/targets/tgt_*/...` with `meta.json`, `analysis.json`, etc.
* A content-addressed ID scheme wired into `Snakefile` and `target_artifacts.py`.

I would *not* re-implement target building from scratch; just call the existing entrypoints from Parsl.

### Components

1. **Config loader (OmegaConf)**

   * Reads `config/experiments.yaml` (or a new `config/parsl_experiments.yaml`).
   * Computes the (N) targets and (M) sampler configs (`targets`, `samplers` as now).
   * Computes target IDs and run IDs using the same functions as the Snakefile (we can factor them into a small shared module).

2. **Parsl apps**

   Two top-level apps:

   ```python
   @bash_app
   def build_target_app(cfg_yaml_path, target_id, target_dir, stdout, stderr): ...

   @bash_app
   def run_sampler_app(cfg_yaml_path, target_id, run_dir, stdout, stderr): ...
   ```

   Each app just shells out to `uv run lambda-hat-build-target` or `lambda-hat-sample` using the same CLI you already have.

   You can also use `@python_app` and call into `lambda_hat.entrypoints.*.main()` directly, but using your CLI scripts preserves behaviour and guardrails with minimal code.

3. **Driver script** (replaces Snakefile): `flows/parsl_llc.py`

   * Builds the grid of all (target, sampler) combinations.
   * For each unique target: call `build_target_app(...)`.
   * For each (target, sampler): call `run_sampler_app(..., inputs=[target_future])` to express the dependency.
   * Wait on all futures; at the end, read all `analysis.json` into a **single pandas DataFrame** and write `results/llc_runs.parquet`.

4. **Parsl config file** (for HPC)

   * One `parsl_config_slurm.py` or `parsl_config_pbs.py` describing how to submit workers to your cluster.

---

## 3. Code sketch (reasonably concrete)

### 3.1 Factor out the ID logic

Right now `target_id_for` and `run_id_for` live in `Snakefile`. Move those into a tiny Python module, say `lambda_hat/id_utils.py`, so both Snakemake and Parsl can share them.

Example (paraphrasing your Snakefile):

```python
# lambda_hat/id_utils.py
import hashlib, json
from omegaconf import OmegaConf

def _fingerprint_payload_build(cfg):
    c = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    for k in ["runtime", "store"]:
        c.pop(k, None)
    return c

def target_id_for(cfg):
    blob = json.dumps(_fingerprint_payload_build(cfg), sort_keys=True)
    return "tgt_" + hashlib.sha256(blob.encode()).hexdigest()[:12]

def run_id_for(cfg):
    blob = json.dumps(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
                      sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]
```

Keep the existing Snakefile using this module; the Parsl flow imports the same functions.

### 3.2 Parsl config (SLURM example)

This sits in `parsl_config_slurm.py`:

```python
# parsl_config_slurm.py
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.addresses import address_by_hostname

config = Config(
    executors=[
        HighThroughputExecutor(
            label="htex_slurm",
            address=address_by_hostname(),
            max_workers=1,  # 1 worker per node; each worker runs your bash_app
            provider=SlurmProvider(
                partition="normal",
                nodes_per_block=1,
                init_blocks=0,
                min_blocks=0,
                max_blocks=50,
                walltime="04:00:00",
                scheduler_options="",  # extra SBATCH lines
                worker_init="source ~/.bashrc; conda activate llc-env || source .venv/bin/activate",
            ),
        )
    ],
    retries=1,
)
```

You can clone a PBSPro variant by swapping in `PBSPROProvider` etc.

### 3.3 Parsl apps wrapping your CLIs

```python
# flows/parsl_llc.py
import os
from pathlib import Path

import parsl
from parsl import bash_app, python_app
from omegaconf import OmegaConf

from lambda_hat.id_utils import target_id_for, run_id_for

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "lambda_hat" / "conf"
STORE = ROOT / "runs"

@bash_app
def build_target_app(cfg_yaml, target_id, target_dir, stdout=None, stderr=None):
    cmd = (
        f"uv run lambda-hat-build-target "
        f"--config-yaml {cfg_yaml} "
        f"--target-id {target_id} "
        f"--target-dir {target_dir}"
    )
    return cmd

@bash_app
def run_sampler_app(cfg_yaml, target_id, run_dir, stdout=None, stderr=None):
    cmd = (
        f"uv run lambda-hat-sample "
        f"--config-yaml {cfg_yaml} "
        f"--target-id {target_id} "
        f"--run-dir {run_dir}"
    )
    return cmd
```

### 3.4 Building the grid (reuse your config composition)

We can literally reuse the logic from your Snakefile (`compose_build_cfg` and `compose_sample_cfg`), extracted into a Python module (e.g. `lambda_hat/workflow_config.py`). For now I’ll inline something equivalent:

```python
from lambda_hat import omegaconf_support  # registers resolvers
from lambda_hat.id_utils import target_id_for, run_id_for

def compose_build_cfg(t, store_root, jax_enable_x64=True):
    cfg = OmegaConf.load(CONF / "workflow.yaml")
    cfg = OmegaConf.merge(
        cfg,
        {"model": OmegaConf.load(CONF / "model" / f"{t['model']}.yaml")},
        {"data": OmegaConf.load(CONF / "data" / f"{t['data']}.yaml")},
        {"teacher": OmegaConf.load(CONF / "teacher" / f"{t.get('teacher', '_null')}.yaml")},
        {
            "target": {"seed": t["seed"]},
            "jax": {"enable_x64": jax_enable_x64},
            "store": {"root": store_root},
        },
    )
    if "overrides" in t:
        cfg = OmegaConf.merge(cfg, t["overrides"])
    return cfg

def compose_sample_cfg(tid, s, store_root, jax_enable_x64=True):
    base = OmegaConf.load(CONF / "sample" / "base.yaml")
    smpl = OmegaConf.load(CONF / "sample" / "sampler" / f"{s['name']}.yaml")
    cfg = OmegaConf.merge(
        base,
        {"sampler": smpl},
        {
            "target_id": tid,
            "jax": {"enable_x64": jax_enable_x64},
            "store": {"root": store_root},
            "runtime": {"seed": s.get("seed", 12345)},
        },
    )
    if "overrides" in s:
        cfg = OmegaConf.merge(cfg, {"sampler": {s["name"]: s["overrides"]}})
    return cfg
```

That’s essentially the Snakefile version but in plain Python.

### 3.5 Driver: build N targets, then N×M runs

```python
import json
import pandas as pd

def main(experiments_yaml="config/experiments.yaml", parsl_config="parsl_config_slurm.py"):
    # 1. Load Parsl
    from importlib import import_module
    cfg_mod = import_module(parsl_config.replace(".py", ""))
    parsl.load(cfg_mod.config)

    exp = OmegaConf.load(experiments_yaml)
    store_root = exp.get("store_root", "runs")
    jax_x64 = bool(exp.get("jax_enable_x64", True))

    targets_conf = list(exp["targets"])
    samplers_conf = list(exp["samplers"])

    # 2. Build configs & IDs
    target_futures = {}
    target_cfg_paths = {}

    temp_cfg_dir = ROOT / "temp_parsl_cfg"
    temp_cfg_dir.mkdir(exist_ok=True, parents=True)

    for t in targets_conf:
        build_cfg = compose_build_cfg(t, store_root, jax_x64)
        tid = target_id_for(build_cfg)

        cfg_yaml_path = temp_cfg_dir / f"build_{tid}.yaml"
        cfg_yaml_path.write_text(OmegaConf.to_yaml(build_cfg))
        target_cfg_paths[tid] = cfg_yaml_path

        target_dir = Path(store_root) / "targets" / tid
        target_dir.mkdir(parents=True, exist_ok=True)

        # Submit build_target_app once per target
        f = build_target_app(
            cfg_yaml=str(cfg_yaml_path),
            target_id=tid,
            target_dir=str(target_dir),
            stdout=str(target_dir / "build.log"),
            stderr=str(target_dir / "build.err"),
        )
        target_futures[tid] = f

    # 3. For each (target, sampler) pair, submit a sampling job depending on the target future
    run_futures = []
    records = []

    for tid, t in zip(target_futures.keys(), targets_conf):
        for s in samplers_conf:
            sample_cfg = compose_sample_cfg(tid, s, store_root, jax_x64)
            rid = run_id_for(sample_cfg)

            cfg_yaml_path = temp_cfg_dir / f"sample_{tid}_{s['name']}_{rid}.yaml"
            cfg_yaml_path.write_text(OmegaConf.to_yaml(sample_cfg))

            run_dir = Path(store_root) / "targets" / tid / f"run_{s['name']}_{rid}"
            run_dir.mkdir(parents=True, exist_ok=True)

            f = run_sampler_app(
                cfg_yaml=str(cfg_yaml_path),
                target_id=tid,
                run_dir=str(run_dir),
                stdout=str(run_dir / "sample.log"),
                stderr=str(run_dir / "sample.err"),
                inputs=[target_futures[tid]],  # makes it depend on target build
            )
            run_futures.append(f)

            records.append(
                {
                    "target_id": tid,
                    "sampler": s["name"],
                    "run_id": rid,
                    "run_dir": str(run_dir),
                    "cfg_yaml": str(cfg_yaml_path),
                }
            )

    # 4. Wait for all runs to finish
    [f.result() for f in run_futures]

    # 5. Aggregate analysis.json into a DataFrame
    rows = []
    for rec in records:
        analysis_path = Path(rec["run_dir"]) / "analysis.json"
        if analysis_path.exists():
            metrics = json.loads(analysis_path.read_text())
            rows.append({**rec, **metrics})

    df = pd.DataFrame(rows)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    df.to_parquet(results_dir / "llc_runs.parquet", index=False)
    print(f"Wrote {len(df)} rows to {results_dir/'llc_runs.parquet'}")
```

Usage:

```bash
# On your laptop (test quickly, using Local provider in Parsl config)
python flows/parsl_llc.py

# On SLURM
python flows/parsl_llc.py  # Parsl uses parsl_config_slurm.py and submits jobs
```

This is the **entire** replacement for the Snakemake orchestration.

---

## 4. How this maps onto SLURM / PBS

Parsl’s job model:

* You run `python flows/parsl_llc.py` **once** on a login/head node.
* Parsl:

  * Starts a small “manager” process.
  * Submits `sbatch` jobs (or PBS `qsub`) to get worker nodes.
  * Each worker runs a “pilot” that executes your `@bash_app` calls.
* Dependencies (`inputs=[...]`) are just future chaining; Parsl ensures that:

  * `run_sampler_app` for a given target doesn’t start until the `build_target_app` future has completed successfully.
* You get logs per app (in `stdout`/`stderr` paths you chose).

If you want different resource requirements for Stage A vs Stage B, you can define two executors:

* A “big RAM” executor for `build_target_app` (training).
* A “cheaper” executor for `run_sampler_app` (sampling).

and then pass `apps` a `executors=['name']` argument. Not hard, but don’t bother until you need it.

---

## 5. Maintenance trade-offs for *you*

Pros vs your current Snakemake setup:

* All orchestration logic is **Python**; no Snakefile DSL, no profiles.
* Parameter sweeps are just loops + Python data structures; you can generate grids, random search, etc without yaml gymnastics.
* IDs, manifests, aggregation: all in your own code, typed, testable.

Cons:

* You must maintain one more *infra* dependency: Parsl and a small `parsl_config_*.py`.
* You lose Snakemake’s nice `snakemake -n`, rule-specific rerun, and profile ecosystem.
* Debugging failing jobs is now “look at app logs + SLURM logs”, not Snakemake’s summary.

Given your context:

* Single dev, no back-compat, HPC cluster with SLURM/PBS, and a strong preference for **code you can reason about in Python** – I think Parsl is a reasonable replacement *if* you’re prepared to own a small amount of orchestration code.

If you like this direction, next step is:

* Decide: SLURM vs PBS first? (I’d implement SLURM config even if your real cluster is PBS, then port the provider once you’re happy with the flow.)
* I can then rewrite the sketch above into a **fully repo-consistent** `flows/parsl_llc.py` plus `lambda_hat/id_utils.py` and `parsl_config_slurm.py`, matching your exact path conventions and CLI expectations.
