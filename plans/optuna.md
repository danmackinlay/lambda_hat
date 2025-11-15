
---

## Objective (simplified and univariate)

Define the reference and trial estimates as real numbers for a given problem (p):

* ( \widehat{\text{LLC}}^{\mathrm{HMC}}_p ) : HMC reference LLC (a long‑run HMC‑based estimator you trust).
* ( \widehat{\text{LLC}}^{\mathrm{meth}}_{p,\theta} ) : the method’s LLC estimate under hyperparameters ( \theta ).

**Primary objective to minimize**:

[
J(p,\theta) ;=; \big|;\widehat{\text{LLC}}^{\mathrm{meth}}_{p,\theta} - \widehat{\text{LLC}}^{\mathrm{HMC}}_p;\big|
]

That’s it. If you want a slightly more robust objective in case reference noise is non‑negligible, use Huber:

[
J_\delta(x) =
\begin{cases}
\frac{x^2}{2\delta} & |x|\le \delta\
|x| - \frac{\delta}{2} & |x| > \delta
\end{cases}
\quad \text{with } x = \widehat{\text{LLC}}^{\mathrm{meth}}_{p,\theta} - \widehat{\text{LLC}}^{\mathrm{HMC}}_p.
]

Pick (\delta) ~ 1–2× the Monte Carlo SE of the HMC estimator if you have it. If not, set (\delta) to a small constant and move on.

You will still **record** supporting metrics (runtime, SE/CI if available, diagnostics), but they won’t enter the objective unless you decide to multi‑objective later.

---

## Files and layout (minimal, explicit)

```
workflows/
  parsl_config_slurm.py
  parsl_config_pbs.py
  parsl_optuna.py           # the orchestrator (this replaces the Snakefile + BO wrapper)
  apps.py                   # Parsl apps (HMC reference, method run)
lambda_hat/
  id_utils.py               # stable IDs (shared with existing code)
  runners/
    hmc_reference.py        # returns {llc_ref, se_ref?, diagnostics}
    run_method.py           # returns {llc_hat, se_hat?, runtime, ...}
results/
  runs.parquet              # appended/overwritten dataframe of all finished trials
  studies/<study_name>/
    state.pkl               # pickle of Optuna study (optional if using journal)
    journal.log             # (optional) Optuna JournalStorage file
artifacts/
  problems/<pid>/
    spec.yaml
    ref.json                # {llc_ref, se_ref?, ...}
  runs/<pid>/<method>/<trial_id>/
    manifest.json
    metrics.json            # {objective, llc_hat, ...}
    stdout.log / stderr.log (if you use bash_app)
```

IDs (`pid`, `trial_id`) are content‑addressed hashes of normalized configs (problem spec, method + hyperparams, seed, code version). That keeps runs idempotent with trivial dedup.

---

## Parsl config (SLURM & PBS)

**SLURM** (`workflows/parsl_config_slurm.py`):

```python
# workflows/parsl_config_slurm.py
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.addresses import address_by_hostname

config = Config(
    executors=[
        HighThroughputExecutor(
            label="htex",
            address=address_by_hostname(),
            max_workers=1,                 # 1 Python worker per node for predictability
            provider=SlurmProvider(
                partition="normal",
                nodes_per_block=1,
                init_blocks=0,
                min_blocks=0,
                max_blocks=200,           # cluster capacity cap
                walltime="02:00:00",
                scheduler_options="",     # extra #SBATCH lines if needed
                worker_init=(
                    "source ~/.bashrc; "
                    "module load python/3.10 || true; "
                    "source .venv/bin/activate || conda activate llc || true"
                ),
            ),
        )
    ],
    retries=1,
)
```

**PBS/PBSPro** (`workflows/parsl_config_pbs.py`): identical structure, swap `SlurmProvider` → `PBSProProvider` (or `TorqueProvider`) and adjust resource strings.

---

## Parsl apps

Keep them **thin** and call your existing library functions. Use `@python_app` to get structured return values. If you want shell‑captured logs, wrap them with `@bash_app` and call `python -m lambda_hat.runners.*`—but it’s unnecessary if your runners write `metrics.json` themselves.

```python
# workflows/apps.py
from parsl import python_app

@python_app
def compute_hmc_reference(problem_spec: dict, out_ref_json: str, budget_sec: int = 36000):
    from lambda_hat.runners.hmc_reference import run_hmc_reference
    # returns dict: {"llc_ref": float, "se_ref": float|None, "diagnostics": {...}}
    return run_hmc_reference(problem_spec, out_ref_json, budget_sec)

@python_app
def run_method_trial(problem_spec: dict, method_cfg: dict, ref_llc: float,
                     out_metrics_json: str, budget_sec: int = 6000):
    from lambda_hat.runners.run_method import run_trial
    # returns dict: {"llc_hat": float, "se_hat": float|None, "runtime_sec": float, ...}
    return run_trial(problem_spec, method_cfg, ref_llc, out_metrics_json, budget_sec)
```

**Notes**

* Put *all* time‑budget enforcement inside `run_hmc_reference` / `run_trial` so trials exit cleanly and still write metrics before the scheduler walltime.
* If you need different resource shapes, define additional executors (e.g., `hmc_exec` with more memory) and pass `executors=['hmc_exec']` to the app decorator.

---

## The orchestrator (Optuna + Parsl futures), end‑to‑end

Key ideas:

* Build or load **N problems** (each with `spec.yaml`).
* **Launch HMC** references for all N in parallel (or limit concurrency if you must).
* As each reference finishes, **start Optuna** for that problem & method, propose *k* trials, submit them as Parsl tasks, and “tell” Optuna as futures finish.
* Objective is `abs(llc_hat - llc_ref)` (or Huber).
* Write every finished trial to a **single DataFrame** (results/runs.parquet).

```python
# workflows/parsl_optuna.py
import os, json, time, pickle
from pathlib import Path
import optuna
import parsl
import pandas as pd
from importlib import import_module

from omegaconf import OmegaConf
from lambda_hat.id_utils import stable_hash   # implement as blake2/sha1 of normalized dict
from workflows.apps import compute_hmc_reference, run_method_trial

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RES = ROOT / "results"
RES.mkdir(exist_ok=True, parents=True)

def load_parsl(cfg="workflows.parsl_config_slurm"):
    m = import_module(cfg)
    return parsl.load(m.config)

def huber(x, delta=0.1):
    ax = abs(x)
    return 0.5 * x*x/delta if ax <= delta else ax - 0.5*delta

def objective_from_metrics(llc_hat, llc_ref, huber_delta=None):
    diff = llc_hat - llc_ref
    return huber(diff, huber_delta) if huber_delta else abs(diff)

def suggest_method_params(trial, method_name):
    if method_name == "sgld":
        return {
            "eta0": trial.suggest_float("eta0", 1e-6, 1e-1, log=True),
            "gamma": trial.suggest_float("gamma", 0.3, 1.0),
            "batch": trial.suggest_categorical("batch", [32,64,128,256]),
        }
    if method_name == "vi":
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-2, log=True),
            "family": trial.suggest_categorical("family", ["meanfield","fullrank"]),
            "mc_samples": trial.suggest_categorical("mc_samples", [1,4,8,16]),
        }
    if method_name == "mclmc":  # e.g., (U)LMC/MALA
        return {
            "stepsize": trial.suggest_float("stepsize", 1e-5, 1e-1, log=True),
            "target_accept": trial.suggest_float("target_accept", 0.7, 0.95),
        }
    raise ValueError(method_name)

def main():
    # --- Parsl backend ---
    load_parsl("workflows.parsl_config_slurm")  # or pbs

    # --- Problems list (OmegaConf -> dicts) ---
    exp = OmegaConf.load("config/experiments.yaml")  # your existing config
    problems = list(exp["targets"])                  # reusing your notion of "target problem"
    methods  = [s["name"] for s in exp["samplers"] if s["name"] in ("sgld","vi","mclmc")]

    # --- Launch HMC references in parallel ---
    ref_futs = {}
    ref_meta = {}  # pid -> {llc_ref, se_ref?, ...}
    for p in problems:
        # Resolve full problem spec here (model+data+teacher), then freeze to dict
        problem_spec = OmegaConf.to_container(p, resolve=True)
        pid = "p_" + stable_hash(problem_spec)[:12]
        out_ref = ART / "problems" / pid / "ref.json"
        out_ref.parent.mkdir(parents=True, exist_ok=True)

        if out_ref.exists():
            ref_meta[pid] = json.loads(out_ref.read_text())
        else:
            ref_futs[pid] = compute_hmc_reference(problem_spec, str(out_ref), budget_sec=8*3600)

    # Wait for any missing references
    for pid, fut in ref_futs.items():
        ref_meta[pid] = fut.result()  # also written to out_ref.json by the app

    # --- Optuna storage (optional: journal for crash-proof resume) ---
    study_dir = RES / "studies" / "llc_vs_hmc"
    study_dir.mkdir(parents=True, exist_ok=True)
    # For simplicity, single-process in-memory + periodic pickles:
    def save_study(study, path=study_dir/"state.pkl"):
        path.write_bytes(pickle.dumps(study))

    # --- For each problem & method, run a batched ask-&-tell loop with Parsl tasks ---
    rows = []

    for p in problems:
        problem_spec = OmegaConf.to_container(p, resolve=True)
        pid = "p_" + stable_hash(problem_spec)[:12]
        llc_ref = float(ref_meta[pid]["llc_ref"])

        for method_name in methods:
            study = optuna.create_study(direction="minimize", study_name=f"{pid}:{method_name}")
            inflight = {}
            BATCH = 32        # concurrent trials for this (pid, method)
            MAX_TRIALS = 200  # stop criterion per (pid, method)

            def submit_one():
                t = study.ask()
                hp = suggest_method_params(t, method_name)
                manifest = {
                    "pid": pid, "method": method_name, "hyperparams": hp,
                    "seed": int(t.number), "budget_sec": 6000,
                }
                trial_id = "r_" + stable_hash(manifest)[:12]
                run_dir  = ART / "runs" / pid / method_name / trial_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

                fut = run_method_trial(problem_spec, {"name": method_name, **hp},
                                       llc_ref, str(run_dir/"metrics.json"),
                                       budget_sec=manifest["budget_sec"])
                inflight[fut] = (t, trial_id, run_dir)
                return fut

            # prime the pump
            submitted = 0
            while submitted < min(BATCH, MAX_TRIALS):
                submit_one(); submitted += 1

            while study.trials_dataframe().shape[0] < MAX_TRIALS:
                # wait for any future to complete
                done = [f for f in list(inflight.keys()) if f.done()]
                if not done:
                    time.sleep(1); continue

                for f in done:
                    t, trial_id, run_dir = inflight.pop(f)
                    try:
                        result = f.result()  # dict: {"llc_hat": ..., "runtime_sec": ...}
                    except Exception as e:
                        # Penalize crashed/failed trials with a large objective
                        study.tell(t, float("inf"))
                        continue

                    llc_hat = float(result["llc_hat"])
                    obj = objective_from_metrics(llc_hat, llc_ref, huber_delta=None)

                    # persist metrics for audit + aggregation
                    metrics = {**result, "objective": obj, "llc_ref": llc_ref,
                               "pid": pid, "method": method_name, "trial_id": trial_id}
                    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

                    # tell Optuna + row for final table
                    study.tell(t, obj)
                    rows.append(metrics)

                    # keep the window full
                    if submitted < MAX_TRIALS:
                        submit_one(); submitted += 1

                save_study(study)

    # --- Aggregate all trials into a single dataframe ---
    df = pd.DataFrame(rows)
    df.to_parquet(RES / "runs.parquet", index=False)
    print(f"wrote {len(df)} rows to {RES/'runs.parquet'}")
```

**Behavior you get:**

* HMC references across problems run in parallel.
* For each (problem, method), you keep *BATCH* trials in flight; as Parsl futures finish you “tell” Optuna and immediately submit new ones—no idle time.
* Objective is the absolute LLC error (or Huber).
* All metrics land in one Parquet.

---

## Runner contracts (you keep control, minimal surface)

**`lambda_hat/runners/hmc_reference.py`**

```python
def run_hmc_reference(problem_spec: dict, out_ref_json: str, budget_sec: int) -> dict:
    """
    Do whatever your current HMC LLC routine does; but:
      - obey budget_sec (extend samples until time is nearly up)
      - compute llc_ref (float) and optional se_ref
      - write out_ref_json (idempotent) and return the same dict
    """
    # ... your code ...
    ref = {"llc_ref": llc_ref, "se_ref": se_ref, "diagnostics": diag}
    Path(out_ref_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ref_json).write_text(json.dumps(ref, indent=2))
    return ref
```

**`lambda_hat/runners/run_method.py`**

```python
def run_trial(problem_spec: dict, method_cfg: dict,
              llc_ref: float, out_metrics_json: str, budget_sec: int) -> dict:
    """
    Train/run the approximation method under method_cfg until budget_sec.
    Must return llc_hat (float) and can include se_hat etc.
    Writes metrics.json (excluding 'objective') and returns the same dict.
    """
    # ... your code ...
    res = {"llc_hat": llc_hat, "se_hat": se_hat, "runtime_sec": elapsed, "extra": {...}}
    Path(out_metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_metrics_json).write_text(json.dumps(res, indent=2))
    return res
```

The orchestrator computes the objective and tells Optuna.

---

## Concurrency, resources, and fairness

* **Concurrency**: tune it with the Parsl executor (`max_blocks`, `max_workers`). Per‑(problem, method) concurrency is `BATCH`.
* **Resources**: if HMC needs fatter nodes, define a second executor and add `executors=['hmc_exec']` on `compute_hmc_reference`.
* **Budget fairness**: every trial gets `budget_sec`, enforced *inside* the runner; Parsl walltime is a backstop.
* **Retries**: Parsl `retries=1` in the config re‑executes failed apps once. You can also catch exceptions and immediately “tell” a large objective.

---

## What you’re *not* doing anymore

* No Snakefile, no cluster profiles.
* No distributional distance metrics unless you want them for diagnostics.
* No Hydra/Metaflow configs. Single config stack (OmegaConf) for problem/method definitions; Parsl config is isolated to cluster submission only.

---

## Minimal commands you’ll actually run

```bash
# 1) Configure Parsl backend (edit workflows/parsl_config_slurm.py or parsl_config_pbs.py)
# 2) Kick off orchestration (single process on a login node)
python -m workflows.parsl_optuna
# This launches HMC refs and then continuously fills the cluster with trials.
```

---

## Sanity check list

* [ ] HMC runner returns a single **float** `llc_ref` and obeys `budget_sec`.
* [ ] Method runner returns a single **float** `llc_hat` and obeys `budget_sec`.
* [ ] Absolute (or Huber) error is the **only** objective.
* [ ] All runs are content‑addressed (`pid`, `trial_id`) for natural dedup.
* [ ] One Parquet file with all finished trials; nothing else is required to analyze results.

If you want me to tighten the code skeletons to your exact module names and current entrypoints (so you can drop them straight in), say so and I’ll produce them in that shape.
