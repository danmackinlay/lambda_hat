Short answer: **yes**—moving the local workflow from a **thread** pool to a **process** pool is a clean way to fence off the two things that keep biting you (Matplotlib’s global state and JAX’s global flags). And you’re right: the way your Parsl apps are written, you’re **already loading the big stuff from disk inside the task**, so switching to multi‑process won’t force you to pickle large objects.

Below I show (1) why this helps in *your* repo, (2) the smallest code/config changes to add a local multi‑process profile, and (3) how to run the smoke test with it.

---

## Why this will help here

* **Local mode is threads today.** Your CLI/Docs wire `--local` to a Parsl config that uses `ThreadPoolExecutor` (up to 8 workers). That’s explicit both in the CLI help and in the workflow docs.

* **Threaded workers share process-global state.** That’s exactly where you saw intermittent:

  * Matplotlib/ArviZ mathtext parsing failures (global rcParams/backends).
  * JAX `jax_enable_x64` toggling/race across threads.

  With **separate processes**, each worker gets an isolated interpreter and its own JAX/Matplotlib state.

* **You’re not passing big Python objects between workers anyway.** Your Parsl apps (`build_target_app`, `run_sampler_app`) take *paths and IDs*, set JAX precision **inside** the app, and then call the command entrypoints which load artifacts from disk. That means no large models/datasets are pickled across the executor boundary—just small strings. See the app signatures and their internal `jax.config.update(...)` calls; the heavy I/O happens in the command modules they invoke.
  (And the artifact layout/IO pattern is designed for this: params/data are written once and read by each run. )

* **Your Parsl “cards” already use HTEX for SLURM.** The SLURM profiles use Parsl’s `HighThroughputExecutor` (HTEX), which launches worker **processes**. We can mirror that locally with a `LocalProvider` variant so you get the same isolation without a cluster. The current `parsl_cards.py` shows how the SLURM card builds an HTEX; we’ll add a “local htex” branch.

---

## Minimal changes: add a local multi‑process card

### 1) Create a local HTEX Parsl card

Add `config/parsl/local-htex.yaml`:

```yaml
# Local multi-process execution via HighThroughputExecutor
type: local_htex
label: htex_local
run_dir: parsl_runinfo
retries: 1

# Concurrency: how many concurrent sampling/build jobs
max_workers: 4          # tune to cores; 4 is a safe default on laptops

# Optional: extra worker initialization (env for plotting, etc.)
worker_init: |
  export MPLBACKEND=Agg
  export JAX_DEFAULT_PRNG_IMPL=threefry2x32
```

> Rationale
> *`max_workers`* controls how many separate **processes** run in parallel (not per-chain; your chains are vmapped inside each job). `MPLBACKEND=Agg` here is belt‑and‑suspenders to ensure non‑GUI plotting inside each process.

### 2) Teach `parsl_cards.py` how to build a local HTEX config

Patch `lambda_hat/parsl_cards.py` to add a `local_htex` branch (mirrors the existing SLURM HTEX branch but with a `LocalProvider`):

```diff
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
-from parsl.providers import SlurmProvider
+from parsl.providers import SlurmProvider, LocalProvider
@@
 def build_parsl_config_from_card(card: DictConfig) -> Config:
@@
     if typ == "local":
         ...
         return Config(
             executors=[
                 ThreadPoolExecutor(
                     label=card.get("label", "local_threads"), max_threads=int(max_threads)
                 )
             ],
             retries=int(card.get("retries", 1)),
             run_dir=run_dir,
         )
+    if typ in ("local_htex", "htex_local"):
+        # Local multi-process via HighThroughputExecutor
+        provider = LocalProvider(
+            init_blocks=1,
+            min_blocks=1,
+            max_blocks=1,
+            nodes_per_block=1,
+            worker_init=(card.get("worker_init") or "").strip(),
+        )
+        htex = HighThroughputExecutor(
+            label=card.get("label", "htex_local"),
+            address=address_by_hostname(),
+            max_workers=int(card.get("max_workers", os.cpu_count() or 4)),
+            provider=provider,
+        )
+        return Config(
+            executors=[htex],
+            retries=int(card.get("retries", 1)),
+            run_dir=run_dir,
+        )
@@
     if typ == "slurm":
         ...
```

This follows the existing HTEX pattern used for SLURM in the same file (that one uses `SlurmProvider`; we use `LocalProvider` and expose `max_workers` the same way).

> Why not `ProcessPoolExecutor`?
> Parsl doesn’t ship a `ProcessPoolExecutor`; the standard way to get multi‑process is HTEX. You already use it on SLURM; this just makes a local flavor.

---

## How to run it

Use the card instead of `--local`:

```bash
uv run lambda-hat workflow llc \
  --config tests/test_smoke_workflow_config.yaml \
  --parsl-card config/parsl/local-htex.yaml \
  --set max_workers=4
```

In the CLI and docs, `--local` is hard‑wired to the **ThreadPool** profile; `--parsl-card` lets you pick the new **HTEX** profile. That linkage is visible in both the CLI and docs (“Using Parsl mode: local (ThreadPool)”).

---

## Will we still serialize big objects?

No—the apps pass only small strings and flags and then do all IO internally:

* `run_sampler_app(cfg_yaml, target_id, experiment, jax_x64, ...)` sets `jax_enable_x64` **inside** the worker process and calls `sample_entry`, which loads the target (data + params) from disk via your artifact layer.
* Targets are stored on disk as `data.npz` and Equinox `params.eqx` (and re‑loaded per run), designed to avoid shipping large pytrees through the executor.

So moving to process workers won’t change data movement patterns; it just gives each job its own interpreter and global state.

---

## Extra hardening (keep or drop as you see fit)

Even with multi‑process, these are good, low‑risk guards:

1. **Keep LaTeX off entirely in analysis** (before any ArviZ plotting):

   ```python
   import matplotlib as mpl
   mpl.rcParams['text.usetex'] = False
   mpl.rcParams['mathtext.default'] = 'regular'
   ```

   (Thread safety issues largely go away with processes, but this eliminates the mathtext parser path that raised the `ParseException`.)

2. **Unique Matplotlib cache per run** (avoids any cross‑process cache contention on macOS):

   ```python
   os.environ.setdefault("MPLCONFIGDIR", str(run_dir / "mplconfig"))
   ```

   where `run_dir` is the sampler’s run directory.

3. **Leave your dtype guards in place.** The `_deserialize_model()` x64 shim and the “belt‑and‑suspenders” casts in SGLD/VI stay useful, but with process isolation the odds of cross‑job x64 drift drop substantially. (Your SLURM HTEX branch keeps `max_workers` and `worker_init` already—mirroring that locally is consistent.)

---

## Summary

* **Yes**—switching to a multi‑process executor will likely remove the intermittent Matplotlib/JAX global‑state collisions you’re seeing.
* You **already** load big artifacts from disk inside each app, so there’s **no large-object pickling** penalty to moving to processes.
* **Minimal change path**: add a `local_htex` pars l card + tiny `parsl_cards.py` patch, then run with `--parsl-card config/parsl/local-htex.yaml`.
