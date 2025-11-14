Great brief. Here’s a design that gives you a *single, resumable, file‑backed* exploration loop that (a) generates multiple teacher/target problems, (b) computes a long‑run HMC reference per problem, and (c) *optimizes* SGLD, VI, and MCLMC hyperparameters to match that reference under a fixed wall‑time budget per run (100 minutes). It runs locally (laptop) and scales to SLURM/PBSPro with the same commands. No database daemons, minimal moving parts, and clean on‑disk bookkeeping.

---

## TL;DR architecture

**One orchestrator process + Snakemake for execution.**

* **Orchestrator (Python, long‑running, resumable)**
  Uses an *ask‑and‑tell* Bayesian optimizer to propose batches of hyperparameters; materializes them as run directories; calls Snakemake to realize those runs; ingests finished metrics from disk; “tells” the optimizer; repeats until the budget is spent per problem/method.
  (Optuna’s ask‑and‑tell fits perfectly and doesn’t require a DB; if you ever want multiprocess/multinode optimizers, Optuna’s *JournalStorage* is a file‑based NFS‑safe store, so still no DB process. ([optuna.readthedocs.io][1]))

* **Executors (Snakemake rules)**
  One generic rule per algorithm (HMC, SGLD, VI, MCLMC). Snakemake only sees *concrete targets* created by the orchestrator (e.g., `runs/<run_id>/metrics.json`). You can run the exact same Snakefile:

  * Locally: `snakemake -j 8`.
  * On clusters: `snakemake --profile slurm …` or `--profile pbs` (profiles encapsulate the `sbatch/qsub` boilerplate, retries, status scripts, etc.). ([Snakemake][2])

This keeps Snakemake as a *pure executor* (great at parallel job submission and resumption), while all BO logic lives in one small, testable Python module.

---

## On‑disk layout (single shared filesystem)

```
experiments/
  problems/
    <pid>/                         # one directory per teacher/target problem
      spec.yaml                    # data gen + log_prob config
      ground_truth/
        hmc_config.yaml
        posterior_samples.h5
        summary.json               # ESS, R-hat, μ_ref, Σ_ref, diagnostics
  runs/
    <run_id>/
      manifest.json                # method, pid, seed, hyperparams, code_version, budget
      stdout.log / stderr.log
      artifacts/
        samples.h5 | vi_params.pt | …  # method-specific outputs
      metrics.json                 # scalar objective + components
  studies/
    <study_name>/
      study_state.pkl              # opt state (ask-and-tell) for exact resumption
      trials.csv                   # append-only ledger of proposed/told trials
      pending.csv                  # runs proposed but not yet told
      settings.yaml                # knobs: batch_size, max_concurrent, etc.
```

All IDs (`pid`, `run_id`) are *content‑addressed* hashes of normalized configs (problem spec + method + hyperparams + seed + code version). That makes runs idempotent and naturally de‑duplicates work.

---

## What the orchestrator actually does

### 1) Generate diverse teacher/target problems

Create a `ProblemSpec` registry with a few high‑contrast “feels”:

* Conjugate & well‑behaved: Gaussian mean with known variance.
* Correlated GLM: logistic regression with strong feature correlation.
* Multimodal: well‑separated Gaussian mixture.
* Pathology: Neal’s funnel / banana distribution.
* Hierarchical: 2‑level random effects.
* (Optional) Likelihood with stiff geometry: e.g., Bayesian probit w/ Cauchy priors.

Each `spec.yaml` includes: data seed, dimensionality, log‑pdf implementation hook, and any model‑specific transforms. The orchestrator writes these to `experiments/problems/<pid>/spec.yaml`.

### 2) HMC “ground truth” per problem

A dedicated Snakemake rule runs long‑run HMC (e.g., NUTS) until target ESS/diagnostics pass. Output: `posterior_samples.h5` and `summary.json` (μ_ref, Σ_ref, ESS, R̂, divergences). If HMC is still running, the optimizer can temporarily use the latest checkpointed summary and *refresh the objective later* when HMC extends—your metrics get re‑computed without re‑running candidates.

> **Note on caps:** If a single HMC job might exceed 24h preference, let Snakemake submit it with `--resources time=1440` and give the profile a retry policy (`--restart-times` / `--retries`) so long jobs automatically resume from checkpoints if preempted. ([Snakemake][3])

### 3) Optimize SGLD, VI, MCLMC against HMC

**Objective** (minimize): a robust *distance to HMC* computed from samples/params within **100 minutes wall‑time per trial** (enforced inside the algorithm runners).

A good single‑number score that works across methods:

* **MMD** (RBF kernel with median heuristic) between `est_samples` and `ref_samples`, plus
* **Moment error**: ‖μ_est−μ_ref‖₂ normalized by diag(Σ_ref) + Frobenius norm of covariance difference normalized by ‖Σ_ref‖_F.
* (Optional) Predictive calibration if the problem defines a likelihood and test set.

You record the components *and* the scalar in `metrics.json` so you can inspect tradeoffs post‑hoc.

**Budgeting:** Every runner receives `--budget-sec 6000`. Internally, they checkpoint progress and exit cleanly at budget (write partial samples/ELBO trajectory and metrics so the trial is usable even if truncated).

### 4) BO loop design (Optuna ask‑and‑tell)

* Use **ask‑and‑tell** so the optimizer lives inside your orchestrator and Snakemake stays a pure executor. This is the most Snakemake‑friendly way to integrate BO. ([optuna.readthedocs.io][1])
* Run in **batches** (e.g., 8–64 trials at a time, depending on cluster capacity). “Ask” *k* trials, materialize their run dirs, then ask Snakemake to build those targets. When they finish, parse `metrics.json` and “tell” the results, then ask the next *k*.
* If you ever need multi‑process/multi‑node optimizers without a DB, Optuna’s **JournalStorage** is file‑based (NFS OK) and designed for distributed runs without RDB/Redis. Keep it optional; you don’t need it for the single‑orchestrator pattern. ([optuna.readthedocs.io][4])

---

## Snakemake glue (simple, stable)

**Snakefile (sketch):**

```python
# workflows/Snakefile
import json, time, pathlib

# Inputs are concrete targets orchestrator asks for.
# HMC reference ---------------------------------------------------------------
rule hmc_reference:
    input:
        "experiments/problems/{pid}/spec.yaml"
    output:
        "experiments/problems/{pid}/ground_truth/posterior_samples.h5",
        "experiments/problems/{pid}/ground_truth/summary.json"
    resources:
        time=1440, mem_mb=16000, threads=4
    shell:
        """
        python pipelines/hmc_reference.py \
          --spec {input} \
          --samples {output[0]} \
          --summary {output[1]}
        """

# Method run (SGLD/VI/MCLMC) -------------------------------------------------
rule method_run:
    input:
        spec  = "experiments/problems/{pid}/spec.yaml",
        ref   = "experiments/problems/{pid}/ground_truth/summary.json",
        mfest = "experiments/runs/{rid}/manifest.json"
    output:
        metrics = "experiments/runs/{rid}/metrics.json"
    resources:
        time=120, mem_mb=8000, threads=1
    shell:
        """
        python pipelines/run_method.py \
          --manifest {input.mfest} \
          --spec {input.spec} \
          --ref-summary {input.ref} \
          --metrics {output.metrics}
        """
```

> **Why this works with BO:** the orchestrator *generates* `manifest.json` and asks Snakemake to build the exact `metrics.json` files for those run IDs. Snakemake handles parallelization, retries, and cluster submission. Profiles make the same workflow run on SLURM or PBS without changing your code/commands. ([Snakemake][2])

**Cluster portability with profiles**
Put a `profiles/slurm/` and `profiles/pbs/` in your repo (or use user‑level ones). Then:

```bash
# Local laptop
snakemake -s workflows/Snakefile -j 8

# SLURM
snakemake -s workflows/Snakefile --profile slurm

# PBS / PBSPro (Torque profile works similarly)
snakemake -s workflows/Snakefile --profile pbs
```

Profiles encapsulate submission commands, default resources, cluster status scripts, and retries. (See Snakemake profiles docs and examples; cluster execution covers SLURM and generic PBS explicitly.) ([Snakemake][2])

**Retries & canceled jobs**
Use `--retries N` (formerly `--restart-times`) to automatically resubmit failed/canceled jobs from the same target, and include a cluster status script in the profile so Snakemake detects cancelation properly on your scheduler. ([Snakemake][3])

---

## Orchestrator (sketch)

```python
# orchestrator/study.py
import json, time, subprocess, hashlib, pickle, signal, os
from pathlib import Path
import optuna

STUDY_DIR = Path("experiments/studies/bo_v1")
RUNS_DIR  = Path("experiments/runs")
PROB_DIR  = Path("experiments/problems")

def stable_id(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()[:12]

def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))

def ensure_hmc(pid):
    # Ask Snakemake for the ref artifacts (it will no-op if present)
    targets = [
      f"experiments/problems/{pid}/ground_truth/posterior_samples.h5",
      f"experiments/problems/{pid}/ground_truth/summary.json",
    ]
    run_snakemake(targets)

def run_snakemake(targets, profile=None, jobs=None):
    cmd = ["snakemake", "-s", "workflows/Snakefile", "--rerun-incomplete"]
    if profile: cmd += ["--profile", profile]
    if jobs: cmd += ["-j", str(jobs)]
    cmd += targets
    subprocess.run(cmd, check=True)

def suggest_batch(study, pid, method, batch_k):
    trials = []
    for _ in range(batch_k):
        t = study.ask()
        # Suggest hyperparams *per method*
        if method == "sgld":
            eta0 = t.suggest_float("eta0", 1e-6, 1e-1, log=True)
            gamma = t.suggest_float("gamma", 0.3, 1.0)
            mb    = t.suggest_categorical("batch", [16,32,64,128,256])
            t.set_user_attr("method", "sgld")
            hp = {"eta0": eta0, "gamma": gamma, "batch": mb}
        elif method == "vi":
            lr  = t.suggest_float("lr", 1e-5, 5e-2, log=True)
            k   = t.suggest_categorical("mc_samples", [1,4,8,16])
            fam = t.suggest_categorical("family", ["meanfield","fullrank"])
            hp  = {"lr": lr, "mc_samples": k, "family": fam}
            t.set_user_attr("method", "vi")
        else: # mclmc (e.g., MALA/ULMC)
            eps = t.suggest_float("stepsize", 1e-5, 1e-1, log=True)
            acc = t.suggest_float("target_accept", 0.7, 0.95)
            hp  = {"stepsize": eps, "target_accept": acc}
            t.set_user_attr("method", "mclmc")

        manifest = {
            "pid": pid, "method": t.user_attrs["method"],
            "hyperparams": hp, "seed": t.number,
            "budget_sec": 6000, "code_version": os.getenv("GIT_COMMIT","dirty")
        }
        rid = stable_id(manifest)
        run_dir = RUNS_DIR / rid
        write_json(run_dir / "manifest.json", manifest)
        trials.append((t, rid))
    return trials

def tell_finished(study, pending):
    done, still = [], []
    for t, rid in pending:
        metrics_path = RUNS_DIR / rid / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            study.tell(t, m["objective"])  # lower is better
            done.append(rid)
        else:
            still.append((t, rid))
    return done, still

def main():
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    storage = STUDY_DIR / "study_state.pkl"
    if storage.exists():
        study = pickle.loads(storage.read_bytes())
    else:
        study = optuna.create_study(direction="minimize")  # single-objective
    profile = os.getenv("SNAKEMAKE_PROFILE", None)

    problems = [...]   # materialize N pids + write spec.yaml for each
    methods  = ["sgld","vi","mclmc"]

    for pid in problems:
        ensure_hmc(pid)
        for method in methods:
            pending = suggest_batch(study, pid, method, batch_k=16)
            targets = [f"experiments/runs/{rid}/metrics.json" for _, rid in pending]
            run_snakemake(targets, profile=profile)
            # Collect all finished
            while pending:
                done, pending = tell_finished(study, pending)
                time.sleep(5)
            # checkpoint optimizer state
            STUDY_DIR.joinpath("trials.csv").write_text(study.trials_dataframe().to_csv(index=False))
            STUDY_DIR.joinpath("study_state.pkl").write_bytes(pickle.dumps(study))

if __name__ == "__main__":
    main()
```

* The orchestrator is **fully resumable**: it pickles the study, writes a CSV ledger, and every run is keyed by a stable `run_id`. On restart, it simply re‑reads `metrics.json` for any runs that finished while it was down and “tells” the optimizer before continuing.
* You can *optionally* switch to Optuna **JournalStorage** later to allow multiple orchestrators across nodes using a single journal file on NFS (no DB). ([optuna.readthedocs.io][4])

---

## Algorithm runners (one binary each; simple CLIs)

Each of these obeys the `--budget-sec` param and writes `metrics.json` no matter what:

* `pipelines/hmc_reference.py --spec ... --samples ... --summary ...`
* `pipelines/run_method.py --manifest ... --spec ... --ref-summary ... --metrics ...`

Suggested **hyperparameters** to expose (and their typical search ranges):

* **SGLD**: `eta0` (1e‑6..1e‑1, log), `gamma` anneal exponent (0.3..1.0), `batch` (16..256), optional preconditioner on diag(Hessian) toggled.
* **VI (reparam, ELBO)**: `lr` (1e‑5..5e‑2, log), `family` (meanfield/fullrank), `mc_samples` (1,4,8,16), `kl_anneal` schedule (off/sigmoid/cosine), `num_iters` auto‑determined inside by wall‑time.
* **MCLMC (e.g., MALA or underdamped LMC)**: `stepsize` (1e‑5..1e‑1, log), `target_accept` (0.70..0.95), optional `mass_diag` preconditioning.

Each runner *streams checkpoints* (e.g., every 60s): partial samples (or VI params), running diagnostics, and a provisional `metrics.json`. That ensures you get a valid score even if the scheduler evicts the job.

---

## Objective and metrics (`metrics.json` schema)

```json
{
  "objective": 0.1234,           // scalar minimized by BO
  "mmd": 0.081,
  "mean_error_l2": 0.032,
  "cov_error_fro": 0.010,
  "runtime_sec": 5980,
  "num_samples": 20000,
  "diagnostics": { "ess_bulk": ..., "rhat": ... }
}
```

The objective can be a weighted sum, e.g. `mmd + 0.5*mean_error_l2 + 0.5*cov_error_fro`. Keep the components so you can audit winners later.

---

## Parallelism and scaling

* **Local testing:** `snakemake -s workflows/Snakefile -j 8` while the orchestrator proposes modest batches (e.g., 4–8).
* **SLURM:** `snakemake --profile slurm` (profiles hide `sbatch` flags, memory, time, status scripts; modern Snakemake also has `--slurm` executor). ([Snakemake][2])
* **PBS/PBSPro:** Use a PBS profile (Torque profile works similarly) or the generic cluster executor with `qsub`. ([Snakemake][5])

**Retries & robustness**

* Add `--retries 3` in the profile to automatically resubmit failed/evicted jobs (older docs call this `--restart-times`). Include a cluster status script so canceled jobs are detected. ([Snakemake][3])
* Snakemake’s on‑disk DAG plus your per‑run directories make resumption natural—rerunning the same targets is a fast no‑op if outputs exist. ([Snakemake][6])

**Why not a DB or a separate scheduler (Ray/Dask/MLflow)?**
Because with *ask‑and‑tell + Snakemake*, you already have robust parallelism, retries, and cluster portability without services to keep alive. If you *later* want multiple optimizers, Optuna JournalStorage on NFS is designed for that without Redis/Postgres. ([optuna.readthedocs.io][4])

---

## Will Snakemake work “inside” a BO loop?

**Yes—when you make BO propose concrete files and let Snakemake just build them.**
Avoid trying to generate dynamic rules at runtime; instead, the orchestrator creates *run manifests* and then invokes Snakemake with those *targets*. This pattern plays nicely with Snakemake’s scheduler and cluster backends (SLURM/PBS). Profiles keep the orchestration commands identical across your laptop and the cluster. ([Snakemake][2])

---

## What you’ll implement (checklist)

1. **Problem registry & generators**

   * `problems/registry.py` (spec -> data + log_prob factory).
   * `cli: python -m problems.make --k 6` writes `spec.yaml` for each pid.

2. **Pipelines**

   * `pipelines/hmc_reference.py` (long‑run, checkpointed; writes `summary.json`).
   * `pipelines/run_method.py` (loads manifest, enforces `--budget-sec`, writes `metrics.json` often).

3. **Snakemake**

   * `workflows/Snakefile` using the two rules shown.
   * `profiles/slurm` and/or `profiles/pbs` (cookiecutter + add a cluster status script). ([Snakemake][3])

4. **Orchestrator**

   * `orchestrator/study.py` as above, with batch size and max concurrency knobs.
   * Checkpoint study state every loop; on start, *rehydrate* and “tell” any completed runs that were pending.

5. **Docs (1 page)**

   * “Dynamic exploration loop” quickstart with 3 commands:

     ```bash
     # 1) Make 6 problems
     python -m problems.make --k 6

     # 2) Start the orchestrator (local)
     python -m orchestrator.study

     # 3) Or on SLURM
     SNAKEMAKE_PROFILE=slurm python -m orchestrator.study
     ```
   * A table listing exposed hyperparameters per method and how to override their *ranges* (for BO) and *values* (for manual runs).

---

## A few pragmatic notes

* **100‑minute fairness:** pass the same `--budget-sec` to every runner; each runner tracks time and exits cleanly with final metrics.
* **Ground truth not ready yet:** allow BO to start; as soon as HMC extends, the orchestrator re‑computes distances and (optionally) *re‑tells* top trials using the improved reference.
* **Serverless “lambda”** isn’t a good fit for >15‑minute jobs. If you want cloud scale without containers, spin up ephemeral VMs and run the same Snakemake profile over NFS/object store; but given your HPC access and container constraints, SLURM/PBS are simpler.
* **PBSPro quirks:** Use the PBS profile (or generic executor) and a status script; set memory/time mapping per rule in the profile (common when porting from SLURM). ([Snakemake][5])

---

## What if you decide not to use Optuna?

* **BoTorch/Ax** give you qEI/qNEI etc., but you’ll still end up writing an ask‑and‑tell shell that looks like the above. Ax’s storage is less turnkey for file‑only persistence; Optuna’s ask‑and‑tell + file persistence is simpler for our constraints. (If you later need GP‑based batched BO, you can *call BoTorch* from inside the orchestrator while preserving the same file protocol.)

---

### References for the key claims above

* Optuna ask‑and‑tell API and examples. ([optuna.readthedocs.io][1])
* Optuna **JournalStorage** for distributed/file‑backed optimization on NFS—no DB process. ([optuna.readthedocs.io][4])
* Snakemake cluster execution & profiles for SLURM/PBS, and CLI options for retries/restarts. ([Snakemake][5])

---

## Bottom line

* **Feasible?** Yes. The **ask‑and‑tell + Snakemake‑as‑executor** pattern gives you robust, resumable optimization without running a database and with trivial portability from laptop to SLURM/PBS.
* **Maintenance?** Small, modular Python files; Snakemake stays declarative; all state on disk.
* **Next step** (I can do it now if you want): add the two pipeline scripts and the small orchestrator skeleton above, plus a 1‑page “Dynamic Exploration Loop” doc with the three commands and the hyperparameter tables.

[1]: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html?utm_source=chatgpt.com "Ask-and-Tell Interface — Optuna 4.5.0 documentation"
[2]: https://snakemake.readthedocs.io/en/stable/executing/cli.html?highlight=profiles&utm_source=chatgpt.com "Command line interface | Snakemake 9.13.6 documentation"
[3]: https://snakemake.readthedocs.io/en/stable/executing/cli.html?utm_source=chatgpt.com "Command line interface | Snakemake 9.13.7 documentation"
[4]: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/011_journal_storage.html?utm_source=chatgpt.com "(File-based) Journal Storage — Optuna 4.5.0 documentation"
[5]: https://snakemake.readthedocs.io/en/v7.28.3/executing/cluster.html?utm_source=chatgpt.com "Cluster Execution — Snakemake 7.28.3 documentation"
[6]: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html?utm_source=chatgpt.com "Snakefiles and Rules | Snakemake 9.13.7 documentation"
