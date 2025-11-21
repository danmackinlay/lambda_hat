Short version: you should treat Optuna as “just another workflow” sitting on top of your *already simplified* execution model (Option D: one executor per card, dtype handled inside tasks). That means:

* **No executor routing in Optuna.**
* **No per‑executor concurrency.**
* **One knob for concurrency (`batch_size`) and one knob for trial budget (`trial_sec`).**
* Slurm + GPU is “just” a different Parsl card.

Below is a replacement plan that assumes that worldview, and tells your agent exactly what to change.

---

## 1. Design decisions (what we’re committing to)

### 1.1 Execution model for Optuna

**Decision:** Optuna uses exactly the same execution model as `workflow llc`:

* One Parsl config (card) ⇒ one HTEX executor.
* JAX precision (float32 vs float64) is controlled **inside the worker** based on the sampler config, *not* by choosing a different executor.

**Rationale:**

* You already went all‑in on Option D for the main workflow to kill the “two executors + routing” complexity.
* Keeping a special multi‑executor model just for Optuna is extra surface area for no real benefit.
* On Slurm+GPU, you’ll be constrained by **one GPU per node** anyway; per‑executor routing doesn’t buy you much.

### 1.2 How “interactive” Optuna loops should feel

**Decision:** Interactive loops look like:

```bash
# Local smoke test
uv run lambda-hat workflow optuna \
  --config config/optuna/default.yaml \
  --local \
  --set optuna.max_trials_per_method=8 \
  --set execution.budget.trial_sec=120

# HPC GPU run
uv run lambda-hat workflow optuna \
  --config config/optuna/default.yaml \
  --parsl-card config/parsl/slurm/gpu-a100.yaml \
  --set optuna.max_trials_per_method=64 \
  --set optuna.concurrency.batch_size=16 \
  --set execution.budget.trial_sec=600
```

All the “interesting” choices (max trials, batch size, budgets) live in YAML; CLI only gives you `--config`, `--local/--parsl-card` and `--set`.

**Rationale:** You don’t want to agonise at the CLI. This is consistent with `config_optuna.py`: YAML is the source of truth; `--set` is just a low‑friction override path.

### 1.3 Slurm + GPU model

**Decision:** Slurm GPU cards define *one* HTEX executor that owns exactly one GPU per block:

```yaml
# config/parsl/slurm/gpu-a100.yaml (intent)
type: slurm
label: htex_slurm
nodes_per_block: 1
gpus_per_node: 1
max_blocks: 64            # upper bound on parallel trials
max_workers: 1            # one worker per GPU
walltime: "00:30:00"      # trials must be <= this
...
```

Optuna’s `batch_size` is used **in addition** to this; effective parallelism is:

```
concurrency = min(batch_size, max_blocks * max_workers)
```

**Rationale:**

* You said you’re penalised for long jobs; the obvious pattern is many short blocks with `nodes_per_block=1` and small walltime, not huge multi‑hour pilots.
* With `gpus_per_node=1` and `max_workers=1`, every trial sees exactly one GPU, no oversubscription games.

---

## 2. What needs changing (high‑level)

Given Option D, the Optuna stack has three pieces that are now over‑engineered:

1. **Config** still talks about multi‑executor routing and per‑executor caps.
   Files: `config/optuna/default.yaml`, `config/optuna_demo.yaml`.

2. **Config loader** builds `_executor_routing` and validates labels against the currently loaded Parsl config.

3. **Workflow** uses `_executor_for(...)` and keeps per‑executor inflight caps.

You want all of that gone. Optuna should know nothing about executor labels; it should just:

* Submit HMC references.
* Submit trials with a global `batch_size` cap.
* Let the *card* decide how that maps to real jobs.

---

## 3. Concrete changes, file by file

### 3.1 `config/optuna/default.yaml` (and `config/optuna_demo.yaml`)

**Goal:** Remove executor routing / per‑executor concurrency from the schema. Keep budgets and a *single* concurrency knob.

**Edits:**

1. Under `optuna.concurrency`, delete `per_executor`:

```yaml
optuna:
  concurrency:
    batch_size: 16    # Total inflight trials per (problem, method)
    # per_executor:   # ← delete this block
    #   htex64: 8
    #   htex32: 16
```

2. Under `execution`, delete `executor_map` and its comment:

```yaml
execution:
  parsl_card: config/parsl/local.yaml

  budget:
    hmc_sec: 7200
    trial_sec: 600

  # executor_map: ...   # ← delete entire executor_map block & AUTO rules comment
```

3. In the demo config, do the same (the demo currently comments out `executor_map`; just ensure there is no mention of per‑executor concurrency).

4. If you want to bake in “interactive defaults” for dev, you can make the demo config more aggressive:

```yaml
optuna:
  max_trials_per_method: 16
  concurrency:
    batch_size: 4
execution:
  budget:
    hmc_sec: 1800
    trial_sec: 300
```

### 3.2 `lambda_hat/config_optuna.py`

**Goal:** Strip out all executor‑aware logic; make the loader purely about YAML + basic validation.

Right now you have:

* `_available_executor_labels()`
* `_resolve_executor_map(...)`
* `_validate_executor_labels(...)`
* `load_cfg(..., validate_executors=True)` that pokes at `parsl.dfk().config`.

**Edits:**

1. **Delete** `_available_executor_labels`, `_resolve_executor_map`, `_validate_executor_labels`. They are dead in the new model.

2. In `_validate_budgets`, drop the per‑executor section:

```python
def _validate_budgets(cfg: DictConfig):
    hmc_sec = cfg.execution.budget.hmc_sec
    trial_sec = cfg.execution.budget.trial_sec
    batch_size = cfg.optuna.concurrency.batch_size

    if hmc_sec <= 0: ...
    if trial_sec <= 0: ...
    if batch_size < 1: ...

    # Remove per_executor validation entirely
```

3. In `load_cfg(...)`:

* Drop the `validate_executors` argument altogether; it’s meaningless now.
* Remove the block that calls `_available_executor_labels`, `_resolve_executor_map`, `_validate_executor_labels` and stuffs `_executor_routing` into `cfg`.

Leaving something like:

```python
def load_cfg(config_path, dotlist_overrides=None) -> DictConfig:
    cfg = OmegaConf.load(...)
    # apply overrides; set defaults for optuna.concurrency.batch_size,
    # execution.budget.hmc_sec / trial_sec
    _validate_budgets(cfg)
    _validate_search_spaces(cfg)
    return cfg
```

4. `lambda_hat/cli.py` currently passes `validate_executors=True` when calling `load_cfg` from `workflow_optuna`. Remove that argument.

Result: the Optuna config loader doesn’t care how many executors exist or what they’re called.

### 3.3 `lambda_hat/workflows/parsl_optuna.py`

**Goal:** Make the Optuna workflow oblivious to executor labels and routing; concurrency solely via `batch_size`.

You currently have:

* `_available_labels()`
* `_executor_for(role, method, cfg)` that consults `cfg._executor_routing` and falls back to querying the DFK again.
* A log line that prints “Per-executor caps”.
* HMC stage: `executor = _executor_for("hmc"...); compute_hmc_reference(..., executor=executor)`.
* Trial loop: tracks `per_executor_caps` and inflight counts per executor.

**Edits:**

1. **Delete** `_available_labels()` and `_executor_for()`. They have no role in a single‑executor world.

2. In `run_optuna_workflow`:

   * Drop all references to `per_executor_caps` and `cfg._executor_routing`. Simplify the initial logging:

   ```python
   log.info("=== Optuna Workflow Configuration ===")
   log.info("Problems: %d", len(problems))
   log.info("Methods: %s", methods)
   log.info("Max trials per (problem, method): %d", max_trials_per_method)
   log.info("Batch size (concurrent trials): %d", batch_size)
   log.info("HMC budget: %ds (%.1fh)", hmc_budget_sec, hmc_budget_sec / 3600)
   log.info("Method budget: %ds (%.1fmin)", trial_budget_sec, trial_budget_sec / 60)
   ```

   (No per‑executor caps, no routing log.)

3. **Stage 1 – HMC references:**

   Replace:

   ```python
   executor = _executor_for("hmc", "", cfg)
   log.info("    Submitting HMC reference computation → %s", executor)
   ref_futs[pid] = compute_hmc_reference(
       problem_spec, str(out_ref), budget_sec=hmc_budget_sec, executor=executor
   )
   ```

   with:

   ```python
   log.info("    Submitting HMC reference computation")
   ref_futs[pid] = compute_hmc_reference(
       problem_spec, str(out_ref), budget_sec=hmc_budget_sec
   )
   ```

   (Let Parsl dispatch to the single configured executor.)

4. **Stage 2 – Optuna ask/tell loop:**

   The structure is roughly:

   * Keep `inflight: dict[trial_number, Future]`.
   * While `len(inflight) < batch_size` and `completed < max_trials_per_method`: `ask` a new trial, generate hyperparams, submit `run_method_trial(...)` with `executor=_executor_for(...)` and bookkeep.

   You want to:

   * Drop per‑executor accounting entirely.
   * Drop `executor=_executor_for("method", method_name, cfg)` from the `run_method_trial` call.

   So submission becomes:

   ```python
   fut = run_method_trial(
       problem_spec=problem_spec,
       method_name=method_name,
       method_params=method_params,
       ref_llc=llc_ref,
       out_dir=str(trial_dir),
       budget_sec=trial_budget_sec,
   )
   inflight[trial.number] = fut
   ```

   And the refill logic is simply:

   ```python
   while len(inflight) < batch_size and n_submitted < max_trials_per_method:
       trial = study.ask()
       params = suggest_params_from_yaml(trial, search_space[method_name])
       # submit as above
   ```

   No `per_executor_caps`, no label bookkeeping.

5. **Optional:** Log which executor label is actually being used, purely for debugging:

   ```python
   import parsl
   labels = [ex.label for ex in parsl.dfk().config.executors]
   log.info("Parsl executors: %s", labels)
   ```

   But don’t use those labels for routing.

---

## 4. Slurm + GPU card sanity

Your `config/parsl/slurm/gpu-a100.yaml` already looks roughly like this:

```yaml
type: slurm
label: htex_slurm
partition: gpu
nodes_per_block: 1
init_blocks: 0
min_blocks: 0
max_blocks: 50
walltime: "01:59:00"
cores_per_node: 16
mem_per_node: 64
gpus_per_node: 1
...
max_workers: 1
worker_init: |
  module load python || true
  ...
  export JAX_DEFAULT_PRNG_IMPL=threefry2x32
```

For GPU Optuna runs this is fine; just make sure:

* `max_workers: 1` (one worker per GPU).
* Walltime isn’t smaller than `execution.budget.trial_sec` in Optuna configs.
* If you want more parallelism, bump `max_blocks` and `optuna.concurrency.batch_size` together; concurrency is capped by both.

If you later want CPU optuna runs, define `config/parsl/slurm/cpu.yaml` similarly (no GPU, maybe `max_workers>1`).

---

## 5. CLI & UX tweaks

The `workflow optuna` command in `lambda_hat/cli.py` already looks close to what you want: it loads a Parsl card, then calls `load_cfg`, then `run_optuna_workflow`.

Given the simplifications above:

1. **Keep**:

   * `--config`
   * `--local` / `--parsl-card`
   * `--set`
   * `--dry-run`
   * `--resume` (if present)

2. **Do not re‑introduce** any “max‑trials / batch‑size / budget” CLI flags. Those belong in YAML and `--set`.

3. **Dry run:** Already prints the resolved cfg and executor routing; after you remove routing, adjust it to just print the resolved cfg (and maybe the loaded executor labels for sanity):

   ```python
   if dry_run:
       click.echo(OmegaConf.to_yaml(cfg))
       labels = [ex.label for ex in parsl_cfg.executors]
       click.echo(f"Executors: {labels}")
       ...
   ```

---

## 6. How this enables interactive loops in practice

Once you apply the changes above, your interactive Optuna workflow degenerates to:

* **On laptop:**

  ```bash
  uv run lambda-hat workflow optuna \
    --config config/optuna_demo.yaml \
    --local \
    --set optuna.max_trials_per_method=8 \
    --set optuna.concurrency.batch_size=2 \
    --set execution.budget.trial_sec=120 \
    --dry-run  # sanity check
  ```

  If the dry run prints the config you expect, drop `--dry-run` and run for real. You can iterate by nudging search spaces and budgets in YAML.

* **On Slurm+GPU:**

  ```bash
  uv run lambda-hat workflow optuna \
    --config config/optuna/default.yaml \
    --parsl-card config/parsl/slurm/gpu-a100.yaml \
    --set optuna.max_trials_per_method=64 \
    --set optuna.concurrency.batch_size=16 \
    --set execution.budget.trial_sec=600
  ```

  The cluster sees a stream of short jobs (blocks) each using one GPU; Optuna keeps `<=batch_size` trials inflight; you can stop and restart with the same config to resume from the study pickle.

There is no mental overhead about “which executor label is my SGLD going to?” and no code that needs to know those labels.

---

## 7. Sanity/consistency checks after refactor

Have the agent add / update tests:

1. **Config loader tests**:

   * `load_cfg` no longer touches Parsl.
   * `_executor_routing` field no longer exists on cfg.
   * `per_executor` in YAML is ignored or rejected (I’d prefer `KeyError` if someone adds it).

2. **Integration test**: `tests/test_optuna_workflow.py` still passes with:

   * Single executor local card.
   * Single executor Slurm CPU/GPU cards (if you can exercise them in CI or a local cluster).

3. **Docs**: Update `docs/optuna_workflow.md` to remove references to `executor_map`, per‑executor caps and the old `suggest_method_params()`; point to YAML search space instead. Some of this doc text is stale and still references CLI flags you’ve since deleted.

---

If you stick to this plan, Optuna becomes “just another user” of your simplified execution stack: it doesn’t know about executors, it only knows “I can submit a bounded number of trials at a time and each one has a time budget”. That’s exactly what you want for interactive tuning on both laptop and Slurm.
