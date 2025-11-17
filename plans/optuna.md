Here’s a tight, repo‑aligned **update of the Optuna modernization plan** that **adapts to the new executor model** (dtype‑aware executors, Parsl cards, and the artifact `RunContext`) while keeping the YAML‑first posture from the earlier brief.

---

## 0) What drifted (and what we must honor now)

* **Executor model changed**: workflows now route tasks to specific **executor labels** (e.g., `"htex64"` vs `"htex32"`) based on precision or method, and pass that label at **app call time** (`executor=...`) — see dtype→executor helpers and per‑call executor selection in the LLC workflow, and the same pattern already copy/pasted into the Optuna implementation. We need to integrate that cleanly rather than hard‑coding labels in the workflow.
* **Parsl cards** are now the way to configure backends. Local cards define **multiple HTEX executors** (dual x64/x32) via an `executors` list, while SLURM cards usually define a **single** HTEX executor (`label: htex_slurm`). Our plan must work when the label set is {`htex64`,`htex32`} **or** just one label.
* **Optuna CLI still exposes redundant flags** (`--max-trials`, `--batch-size`, `--hmc-budget`, `--method-budget`, etc.) we wanted to delete in favor of YAML/dot‑list overrides. Let’s finish that migration.
* **Current Optuna code** calls apps with `executor="htex64"/"htex32"` and still hard‑codes search spaces in `suggest_method_params()` — both misalign with the YAML‑first goal.

---

## 1) Top‑line design (unchanged principles, executor‑aware)

* **YAML is the source of truth** (OmegaConf). CLI = file pointer + `--set` overrides.
* **One place per concern**:

  * Study behavior & search spaces → **optuna config (YAML)**.
  * Execution resources & routing → **Parsl card** (+ a tiny mapping section in the optuna YAML).
* **Executors are declarative**:

  * Choose executors via a small **`executor_map`** in YAML; default to dtype‑based **auto** mapping when possible; gracefully **fallback** to “the only executor we have”.
* **Ids & artifacts**:

  * Use the same content‑addressed ids and persist a **resolved config snapshot**.
  * Switch the Optuna workflow to the shared **`Paths`/`RunContext`** used by the main workflow so run dirs, logs and `parsl_runinfo` are consistent.

---

## 2) Updated Optuna YAML schema (executor‑aware)

```yaml
# config/optuna/default.yaml
problems:
  - model: small
    data: base
    teacher: _null
    seed: 42
    overrides:
      data.noise_scale: 0.05

methods: [sgld, vi, mclmc]

optuna:
  objective: {type: abs}                  # or {type: huber, delta: 0.1}
  max_trials_per_method: 100
  concurrency:
    batch_size: 16                        # total inflight per (problem, method)
    per_executor:                         # optional caps; omit to disable
      htex64: 8
      htex32: 16
  sampler:
    type: tpe
    seed: 42
    n_startup_trials: 20
  pruner:
    type: median
    n_startup_trials: 10
    n_warmup_steps: 1

search_space:                              # moved from Python into YAML
  sgld:
    eta0:  {dist: float, low: 1e-6, high: 1e-1, log: true}
    gamma: {dist: float, low: 0.3, high: 1.0}
    batch: {dist: categorical, choices: [32, 64, 128, 256]}
    precond_type: {dist: categorical, choices: [none, rmsprop, adam]}
    steps: {dist: int, low: 5_000, high: 20_000, step: 1_000}
  vi:
    lr: {dist: float, low: 1e-5, high: 5e-2, log: true}
    M:  {dist: categorical, choices: [4, 8, 16]}
    r:  {dist: categorical, choices: [1, 2, 4]}
    whitening_mode: {dist: categorical, choices: [none, rmsprop, adam]}
    steps: {dist: int, low: 2_000, high: 20_000, step: 1_000}
    batch_size: {dist: categorical, choices: [32, 64, 128, 256]}
  mclmc:
    step_size:    {dist: float, low: 1e-4, high: 1e-1, log: true}
    target_accept:{dist: float, low: 0.6, high: 0.95}
    L:            {dist: float, low: 0.5, high: 3.0}
    steps:        {dist: int, low: 200, high: 2000, step: 50}

execution:
  parsl_card: config/parsl/local.yaml     # or slurm/gpu-a100.yaml, etc.
  budget:
    hmc_sec: 7200                         # replaces --hmc-budget
    trial_sec: 600                        # replaces --method-budget
  executor_map:
    # Optional. If omitted, use AUTO rules below.
    # You can set per-role or per-method labels:
    hmc: htex64
    methods:
      mclmc: htex64
      sgld:  htex32
      vi:    htex32

store:
  # Uses Paths/RunContext; keep root here for consistency
  root: runs
  layout_version: v1
  namespace: optuna
  ttl_days: 30
```

**AUTO executor rules (when `executor_map` is omitted):**

1. If **both** `htex64` and `htex32` exist in the loaded Parsl config, route: `hmc→htex64`, `mclmc→htex64`, `sgld/vi→htex32`. 2) If only **one** executor exists, route **all** tasks to that label (warn once). We’ll validate labels against the loaded Parsl config. This mirrors how LLC already selects executors by dtype and calls apps with `executor=...`.

---

## 3) CLI policy (finish the clean‑up)

Replace the current, flag‑heavy Optuna command with a YAML‑first CLI:

```
lambda-hat workflow optuna \
  --config config/optuna/default.yaml \
  [--local | --parsl-card config/parsl/slurm/gpu-a100.yaml] \
  [--set key=val ...] [--resume] [--dry-run]
```

**Delete** (in `cli.py`): `--max-trials`, `--batch-size`, `--hmc-budget`, `--method-budget`, `--artifacts-dir`, `--results-dir`, `--study-name`, `--storage`. These are declared in YAML or inferred by `RunContext`. The current CLI shows these flags are still present; this removes the duplication.

Keep: `--config`, `--local`/`--parsl-card` + `--set` (passes dot‑list overrides to the **card**, just like LLC), `--resume`, `--dry-run`. The LLC wiring already demonstrates card loading and early `RunContext` setup; we reuse that for Optuna.

---

## 4) Code changes (executor‑aware + YAML spaces + artifacts unification)

### 4.1 `lambda_hat/cli.py` (optuna command)

* Collapse options to YAML + `--set`. Route Parsl config resolution the same way LLC does (`load_parsl_config_from_card` and `RunContext` to carry `run_dir` override into the card), then `parsl.load(...)`. This mirrors the LLC path that writes the resolved card to `parsl_runinfo/selected_parsl_card.yaml`.
* Add `--dry-run` to dump the resolved Optuna YAML (merged with dot‑list overrides).

### 4.2 `lambda_hat/workflows/parsl_optuna.py`

* **Accept a single `cfg: DictConfig`** (resolved from the YAML), not a forest of parameters.

* **Executor routing**:

  ```python
  def _available_labels() -> set[str]:
      # Read from the loaded Parsl config (parsl.dfk().config.executors)
      return {ex.label for ex in parsl.dfk().config.executors}

  def _executor_for(role: str, method: str, cfg) -> str:
      # 1) explicit map wins
      m = cfg.execution.get("executor_map", {})
      if role == "hmc" and "hmc" in m: return m["hmc"]
      if role == "method" and "methods" in m and method in m["methods"]:
          return m["methods"][method]
      # 2) AUTO rules
      labels = _available_labels()
      both = {"htex64","htex32"}.issubset(labels)
      if role == "hmc" and ("htex64" in labels): return "htex64"
      if role == "method":
          if both and method in ("mclmc",): return "htex64"
          if both: return "htex32"
      # 3) Single executor fallback
      if len(labels) == 1:
          return next(iter(labels))
      raise RuntimeError(f"No suitable executor for role={role}, method={method}.")
  ```

  *This matches the new pattern of passing `executor=...` on each app call.*

* **Budgets & concurrency**:

  * Pull from `cfg.execution.budget.{hmc_sec, trial_sec}` and `cfg.optuna.concurrency.*`.
  * If the Parsl **card walltime** is known (e.g., SLURM cards), assert `trial_sec ≤ walltime_sec`; fail early with a helpful message. (Card walltime lives in the card YAML; resolved card is saved to `parsl_runinfo/selected_parsl_card.yaml` by `load_parsl_config_from_card`.)
  * Honor optional `per_executor` caps by tracking inflight futures **per label**, in addition to the global `batch_size`.

* **Search‑space interpreter (delete hard‑coded ranges)**:

  ```python
  def _suggest(trial, name, spec):
      if spec["dist"] == "float":
          return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
      if spec["dist"] == "int":
          return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step"))
      if spec["dist"] == "categorical":
          return trial.suggest_categorical(name, list(spec["choices"]))
      raise ValueError(f"Unknown dist: {spec['dist']}")

  def suggest_params_from_yaml(trial, space_dict):
      return {k: _suggest(trial, k, dict(v)) for k, v in space_dict.items()}
  ```

  Replace calls to the current `suggest_method_params()` (documented as the source of truth for ranges) with the YAML interpreter.

* **HMC stage**: submit `compute_hmc_reference(..., executor=_executor_for('hmc', '', cfg))` instead of the current hard‑coded `"htex64"` string. This removes the mismatch with SLURM cards that label their single executor `htex_slurm`.

* **Method trials**: compute the label via `_executor_for('method', method_name, cfg)`; submit the `python_app` with `executor=label`, as LLC does.

* **Artifacts**: switch to the shared `Paths`/`RunContext` to compute:

  ```
  runs/optuna/v1/
    problems/p_<hash>/ref.json
    trials/p_<hash>/<method>/r_<hash>/{manifest.json, metrics.json}
    studies/<pid>:<method>.pkl
    tables/optuna_trials.parquet
    meta/resolved_config.yaml
  ```

  (This retains current ids and study pickles, just relocates under a single root with the same rules used elsewhere.) The LLC workflow already constructs `RunContext` up front and injects its `run_dir` into the card; we mirror that.

### 4.3 `lambda_hat/parsl_cards.py` & cards

* **No code change required**. Just ensure **local** cards define **two executors** with `label: htex64` and `label: htex32` so AUTO routing can use them; SLURM cards can keep a single `label: htex_slurm`. The builder already supports multi‑executor local configs and single‑executor SLURM configs.

---

## 5) Example card (local, dual executors)

```yaml
# config/parsl/local.yaml
type: local
run_dir: parsl_runinfo
retries: 1
executors:
  - label: htex64
    max_workers: 2
    cores_per_worker: 1
    worker_init: |
      export MPLBACKEND=agg
      export JAX_ENABLE_X64=true
  - label: htex32
    max_workers: 6
    cores_per_worker: 1
    worker_init: |
      export MPLBACKEND=agg
      export JAX_ENABLE_X64=false
```

AUTO mapping will now route HMC/MCLMC to `htex64`, SGLD/VI to `htex32`.

---

## 6) Migration plan (minimal, one‑way)

1. **Schema + loader**

   * Add `lambda_hat/config_optuna.py` with `load_cfg(path, dotlist) -> DictConfig`:

     * loads YAML, merges `--set` overrides, fills defaults, validates invariants (budgets > 0, `batch_size ≥ 1`, executor labels exist), and returns a **resolved** DictConfig.
     * On load, log detected executor labels from the active Parsl config and the effective mapping (AUTO or explicit).

2. **Workflow rewrite (parsl_optuna.py)**

   * Thread a single `cfg` everywhere; drop CLI‑injected numbers; compute executor routing via `_executor_for(...)` as above.
   * Replace `suggest_method_params()` with the YAML interpreter.
   * Write `meta/resolved_config.yaml` and aggregate to `tables/optuna_trials.parquet` under the `RunContext` root.

3. **CLI simplification**

   * In `lambda_hat/cli.py`, delete redundant flags for Optuna (`--max-trials`, `--batch-size`, budgets, dirs…). Keep `--config`, `--set`, `--local`/`--parsl-card`, `--resume`, `--dry-run`. The code for LLC already shows the card selection + run_dir override.

4. **Cards + docs**

   * Ensure `config/parsl/local.yaml` includes `htex64`/`htex32` labels as above.
   * In docs, stop recommending CLI budget overrides; show YAML + `--set` instead. Current docs still advertise `--max-trials/--batch-size/--hmc-budget/--method-budget`; update those pages.

5. **Safety rails**

   * Validate: if `trial_sec` exceeds parsed walltime in the **resolved card**, error with a suggestion to increase card `walltime` or reduce `trial_sec`. (The resolved card is persisted automatically by `load_parsl_config_from_card`.)

---

## 7) End‑to‑end usage (post‑migration)

**Local test:**

```bash
uv run lambda-hat workflow optuna \
  --config config/optuna/default.yaml \
  --local \
  --dry-run  # prints resolved config & inferred executor map
```

**Quick override without editing files:**

```bash
uv run lambda-hat workflow optuna \
  --config config/optuna/default.yaml \
  --set optuna.max_trials_per_method=24 \
  --set optuna.concurrency.batch_size=6 \
  --set execution.budget.trial_sec=300
```

**SLURM (single‑executor card):**

```bash
uv run lambda-hat workflow optuna \
  --config config/optuna/default.yaml \
  --parsl-card config/parsl/slurm/gpu-a100.yaml \
  --set walltime=04:00:00  # override the card if needed
```

The AUTO routing will detect only `htex_slurm` and route everything there (with a one‑time warning).

---

## 8) Notes on robustness (what you get)

* **Resume**: keep pickled studies (`studies/<pid>:<method>.pkl`) and content‑addressed trial ids; re‑runs skip completed trials. (Documented today and preserved.)
* **Precision correctness**: you can still enforce dtype in sampler presets; AUTO routing follows the intended precision split by default. (HMC/MCLMC presets default to float64; SGLD defaults to float32.)
* **Ask/tell concurrency**: global `batch_size` plus optional `per_executor` caps prevent starving the x64 pool when a mixed workload runs.

---

## 9) Minimal diffs (summary)

* **Delete** redundant Optuna flags in `cli.py`.
* **Add** YAML search‑space interpreter; **delete** `suggest_method_params()` usage.
* **Replace** hard‑coded executor strings with `_executor_for(...)`.
* **Adopt** `Paths`/`RunContext` in Optuna for storage/run dirs to match LLC.
* **Document** the local dual‑executor card and SLURM single‑executor fallback.

---

### Why this fixes the drift

It keeps the original “YAML is the source of truth” promise while aligning with the **executor‑label model** you use elsewhere: you still choose resources via Cards, the workflow selects the right **label per task**, and there’s no more duplication between YAML, code, and CLI. The plan also removes the two‑root (“artifacts/… + results/…”) oddity by adopting the existing artifact context used by the standard workflow, but without changing your content‑addressed ids or Optuna resumability.
