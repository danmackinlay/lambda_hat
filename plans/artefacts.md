Got it. Below is a concrete, HPC‑friendly plan (with drop‑in code you can paste) that:

* **Unifies** all artefacts under one predictable root.
* **Keeps content‑addressable objects** (targets, fitted models, diagnostics).
* **Makes runs and experiments easy to navigate** for both human and Parsl.
* **Adds first‑class TensorBoard logging** (per run, aggregated per experiment).
* **Enables simple, safe cleanup (GC)** of unreachable objects and stale run/scratch files.
* **Uses `direnv`** for sane defaults, but still lets CLI overrides do their thing.
* **Avoids any `llc` naming**; uses `lambda_hat` consistently.

---

## 0) The mental model (3 layers)

**Layer A — Object Store (immutable, deduped)**
Everything that can be reused (synthetic targets, trained artefacts, diagnostics bundles, etc.) is stored once, content‑addressed:

```
$LAMBDA_HAT_STORE/
  objects/
    sha256/<2>/<2>/<64…>/
      payload/            # your files (directory or single file copied here)
      meta.json           # type, schema, params, provenance
```

**Layer B — Experiments (grouping, manifests, TB aggregation)**
Experiments group many runs and expose a simple index and a TensorBoard aggregation point:

```
$LAMBDA_HAT_EXPERIMENTS/
  <experiment_name>/
    manifest.jsonl        # one JSON per run
    targets/              # links to target objects used in this experiment
      <target_id> -> ../../../store/objects/sha256/..../
    runs/
      <run_id>/
        manifest.json
        config.yaml
        inputs/
          target -> ../../targets/<target_id>
        artifacts/        # links to object store outputs from this run
        tb/               # per-run TensorBoard logs
        logs/             # stdout/stderr + python logs
        parsl/            # Parsl run_dir for this run
        scratch/          # temp working dir for tasks (safe to delete)
    tb/                   # links to runs/*/tb for easy multi-run TB
    optuna/               # ask-tell files if applicable
```

**Layer C — Scratch/Cache (ephemeral, safe to wipe)**

```
$LAMBDA_HAT_SCRATCH/      # defaults under artefacts/scratch or $JOBSCRATCH if set
```

This separation keeps “what to keep” (store), “how to view/query” (experiments), and “what can be deleted” (scratch) clean and explicit.

---

## 1) `direnv` defaults (copy into `.envrc`)

```bash
# Root for all lambda_hat artefacts inside your repo by default.
export LAMBDA_HAT_HOME="${LAMBDA_HAT_HOME:-$PWD/artefacts}"

# Core roots (can be redirected to fast/capacity filesystems independently).
export LAMBDA_HAT_STORE="${LAMBDA_HAT_STORE:-$LAMBDA_HAT_HOME/store}"
export LAMBDA_HAT_EXPERIMENTS="${LAMBDA_HAT_EXPERIMENTS:-$LAMBDA_HAT_HOME/experiments}"

# Scratch can prefer job-local NVMe if available.
export LAMBDA_HAT_SCRATCH="${LAMBDA_HAT_SCRATCH:-${JOBSCRATCH:-$LAMBDA_HAT_HOME/scratch}}"

# Nice-to-have defaults:
export LAMBDA_HAT_DEFAULT_EXPERIMENT="${LAMBDA_HAT_DEFAULT_EXPERIMENT:-dev}"
export LAMBDA_HAT_TTL_DAYS="${LAMBDA_HAT_TTL_DAYS:-30}"
export LAMBDA_HAT_TB_MAX_RUNS="${LAMBDA_HAT_TB_MAX_RUNS:-400}"  # guard TB overload

# Optional: pin a Python if you use direnv's 'use python'
# use python
```

Add to `.gitignore`:

```
artefacts/
```

---

## 2) Minimal plumbing you can vendor in (`lambda_hat/artifacts.py`)

> Paste this as a new internal module (no external deps beyond stdlib); your entrypoints import it.

```python
# lambda_hat/artifacts.py
from __future__ import annotations
import os, json, time, uuid, socket, shutil, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, Optional

SCHEMA_VERSION = "1"

def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _short_id(n=6) -> str:
    return uuid.uuid4().hex[:n]

def write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(p)

def atomic_append_jsonl(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    line = json.dumps(obj, sort_keys=True) + "\n"
    with tmp.open("w") as f:
        f.write(line)
    # Append atomically where possible; fall back to rename
    with open(p, "a") as dest:
        with tmp.open("r") as src:
            shutil.copyfileobj(src, dest)
    tmp.unlink(missing_ok=True)

def dir_sha256(path: Path) -> str:
    """Stable hash over relative paths + content; ignores meta.json if present."""
    h = hashlib.sha256()
    base = path
    for root, dirs, files in os.walk(base):
        dirs.sort()
        files.sort()
        for fname in files:
            if fname == "meta.json":
                continue
            fp = Path(root) / fname
            rel = fp.relative_to(base).as_posix().encode()
            h.update(b"FILENAME:")
            h.update(rel)
            h.update(b"\0")
            with open(fp, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk: break
                    h.update(chunk)
    return h.hexdigest()

@dataclass
class Paths:
    home: Path
    store: Path
    experiments: Path
    scratch: Path

    @staticmethod
    def from_env() -> "Paths":
        home = Path(os.environ.get("LAMBDA_HAT_HOME", Path.cwd() / "artefacts")).resolve()
        return Paths(
            home=home,
            store=Path(os.environ.get("LAMBDA_HAT_STORE", home / "store")).resolve(),
            experiments=Path(os.environ.get("LAMBDA_HAT_EXPERIMENTS", home / "experiments")).resolve(),
            scratch=Path(os.environ.get("LAMBDA_HAT_SCRATCH", home / "scratch")).resolve(),
        )

    def ensure(self) -> None:
        for p in [self.store, self.experiments, self.scratch]:
            p.mkdir(parents=True, exist_ok=True)

class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root
        (self.root / "objects" / "sha256").mkdir(parents=True, exist_ok=True)

    def _dest_for_hash(self, h: str) -> Path:
        return self.root / "objects" / "sha256" / h[:2] / h[2:4] / h

    def put_dir(self, src_dir: Path, a_type: str, meta: Dict[str, Any]) -> str:
        """Copy a directory to store, content-addressed. Returns URN."""
        tmp = self.root / "tmp" / uuid.uuid4().hex
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src_dir, tmp)
        h = dir_sha256(tmp)
        dest = self._dest_for_hash(h)
        payload = dest / "payload"
        meta_p = dest / "meta.json"

        if dest.exists():
            shutil.rmtree(tmp)
        else:
            payload.parent.mkdir(parents=True, exist_ok=True)
            tmp.rename(payload)
        # Write/ensure meta.json (first write wins)
        if not meta_p.exists():
            meta = {
                "schema": SCHEMA_VERSION,
                "type": a_type,
                "hash": {"algo": "sha256", "hex": h},
                "created": _now_utc_iso(),
                **meta,
            }
            write_json(meta_p, meta)
        urn = f"urn:lh:{a_type}:sha256:{h}"
        return urn

    def put_file(self, src_file: Path, a_type: str, meta: Dict[str, Any]) -> str:
        tmp_dir = self.root / "tmp" / uuid.uuid4().hex
        payload = tmp_dir / "payload"
        payload.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, payload)
        return self.put_dir(tmp_dir, a_type, meta)

    def path_for(self, urn: str) -> Path:
        # urn format: urn:lh:<type>:sha256:<hash>
        parts = urn.split(":")
        assert len(parts) >= 5 and parts[0] == "urn" and parts[1] == "lh" and parts[3] == "sha256"
        h = parts[4]
        return self._dest_for_hash(h)

def safe_symlink(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
    except FileNotFoundError:
        pass
    os.symlink(src, dest, target_is_directory=src.is_dir())

@dataclass
class RunContext:
    experiment: str
    algo: str
    run_id: str
    run_dir: Path
    tb_dir: Path
    logs_dir: Path
    parsl_dir: Path
    artifacts_dir: Path
    inputs_dir: Path
    scratch_dir: Path

    @staticmethod
    def create(experiment: Optional[str], algo: str, paths: Paths, tag: Optional[str]=None) -> "RunContext":
        experiment = experiment or os.environ.get("LAMBDA_HAT_DEFAULT_EXPERIMENT", "dev")
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        rid = f"{ts}-{algo}{('-' + tag) if tag else ''}-{_short_id()}"
        base = paths.experiments / experiment / "runs" / rid
        ctx = RunContext(
            experiment=experiment,
            algo=algo,
            run_id=rid,
            run_dir=base,
            tb_dir=base / "tb",
            logs_dir=base / "logs",
            parsl_dir=base / "parsl",
            artifacts_dir=base / "artifacts",
            inputs_dir=base / "inputs",
            scratch_dir=base / "scratch",
        )
        for d in [ctx.run_dir, ctx.tb_dir, ctx.logs_dir, ctx.parsl_dir, ctx.artifacts_dir, ctx.inputs_dir, ctx.scratch_dir]:
            d.mkdir(parents=True, exist_ok=True)
        # keep a friendly symlink for TB aggregation
        tb_agg = paths.experiments / experiment / "tb" / rid
        safe_symlink(ctx.tb_dir, tb_agg)
        return ctx

    def write_run_manifest(self, payload: Dict[str, Any]) -> None:
        payload = {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "algo": self.algo,
            "host": socket.gethostname(),
            "created": _now_utc_iso(),
            **payload,
        }
        write_json(self.run_dir / "manifest.json", payload)
        # Also append to experiment-level index
        atomic_append_jsonl(self.run_dir.parent.parent / "manifest.jsonl", payload)
```

---

## 3) How your entrypoints change (light-touch)

### `entrypoints/build_target.py` (pseudo-diff)

```python
# from lambda_hat.artifacts import Paths, ArtifactStore, RunContext, safe_symlink
paths = Paths.from_env(); paths.ensure()
ctx = RunContext.create(experiment=args.experiment, algo="build_target", paths=paths, tag=args.tag)
store = ArtifactStore(paths.store)

# ... generate target into a temp dir 'tmp_target_dir' (files: model.pt, data.npz, etc.)
# IMPORTANT: write into ctx.scratch_dir first, then commit to store for atomicity.
target_dir = ctx.scratch_dir / "target_payload"
target_dir.mkdir(parents=True, exist_ok=True)
# [write files into target_dir here]

urn = store.put_dir(
    target_dir,
    a_type="target",
    meta={"params": vars(args)}  # or whatever describes the target
)

# expose it inside the experiment for humans
target_id = urn.split(":")[-1][:12]
safe_symlink(store.path_for(urn), paths.experiments / ctx.experiment / "targets" / target_id)
safe_symlink(paths.experiments / ctx.experiment / "targets" / target_id, ctx.inputs_dir / "target")

ctx.write_run_manifest({
    "phase": "build_target",
    "outputs": [{"urn": urn, "role": "target"}],
})

print(urn)  # stdout for chaining
```

### `entrypoints/sample.py` (pseudo-diff; VI or samplers)

```python
# from lambda_hat.artifacts import Paths, ArtifactStore, RunContext, safe_symlink
paths = Paths.from_env(); paths.ensure()
ctx = RunContext.create(experiment=args.experiment, algo=args.algo, paths=paths, tag=args.tag)
store = ArtifactStore(paths.store)

# Resolve target (URN or short id)
def resolve_target(t: str) -> str:
    if t.startswith("urn:lh:target:sha256:"):
        return t
    # allow short id present under experiments/<exp>/targets
    p = paths.experiments / ctx.experiment / "targets" / t
    if p.exists():
        meta = json.loads((store.path_for(f"urn:lh:target:sha256:{p.resolve().parts[-1]}") / "meta.json").read_text())
    # Simpler: read meta.json at symlink target
    meta = json.loads((p / "meta.json").read_text())
    return f"urn:lh:target:sha256:{meta['hash']['hex']}"

target_urn = resolve_target(args.target)
safe_symlink(store.path_for(target_urn), ctx.inputs_dir / "target")

# ---- TensorBoard writer (PyTorch or tensorboardX) ----
tb_dir = ctx.tb_dir.as_posix()
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    from tensorboardX import SummaryWriter  # fallback if you prefer
tb = SummaryWriter(tb_dir)
tb.add_text("meta/target_urn", target_urn)
tb.add_text("meta/algo", args.algo)

# ---- run VI / sampler; log key metrics ----
# Example VI metrics:
# for step in range(steps):
#     tb.add_scalar("vi/elbo", elbo, step)
#     tb.add_scalar("vi/kl_q_p", kl, step)
#     tb.add_scalar("vi/lr", lr, step)
#     tb.add_scalar("grad/norm", gnorm, step)

# ---- persist outputs as content-addressed ----
results_dir = ctx.scratch_dir / "results_payload"
results_dir.mkdir(parents=True, exist_ok=True)
# [write result files here: params.pt, traces.npz, metrics.json, etc.]

out_urn = store.put_dir(
    results_dir,
    a_type="fit",
    meta={
        "algo": args.algo,
        "target": target_urn,
        "params": vars(args),
    },
)

# link into run for humans and DAG readability
safe_symlink(store.path_for(out_urn), ctx.artifacts_dir / "fit")

ctx.write_run_manifest({
    "phase": "fit",
    "inputs": [{"urn": target_urn, "role": "target"}],
    "outputs": [{"urn": out_urn, "role": "fit"}],
    "tb_dir": str(ctx.tb_dir),
})

print(out_urn)
```

> **Why this helps Parsl:** set `run_dir=ctx.parsl_dir` in your Parsl `Config` so all Parsl internals go under the run, not scattered globally. Every task writes to `ctx.scratch_dir/<task_id>/...` and the parent driver commits final artefacts to the store.

---

## 4) Parsl: keep it neat inside each run

Where you currently create a Parsl `Config`, do:

```python
from parsl.config import Config
# ... your executors
config = Config(
    executors=[...],
    run_dir=str(ctx.parsl_dir),           # <- key change
    strategy=None,                        # as you like
    checkpoint_mode="task_exit",          # optional
)
```

All of Parsl’s `runinfo`, monitoring, etc., now lives under:

```
experiments/<exp>/runs/<run_id>/parsl/
```

No more top-level `parsl_runinfo` or `temp_parsl_config`.

---

## 5) TensorBoard: per‑run & per‑experiment out of the box

* Each run logs to `runs/<run_id>/tb`.
* The experiment maintains a symlink farm: `experiments/<exp>/tb/<run_id> -> ../runs/<run_id>/tb`.
* You can view everything with:

  ```bash
  tensorboard --logdir "$LAMBDA_HAT_EXPERIMENTS/<exp>/tb"
  ```

**Recommended VI tags**

* `vi/elbo` (scalar)
* `vi/elbo_smoothed` (optional)
* `vi/kl_q_p`, `vi/recon_loss`
* `grad/norm`, `optim/lr`
* `time/step_sec`, `system/gpu_mem_gb` (if relevant)
* For samplers: `sampler/acceptance`, `sampler/ess`, `sampler/rhat`

**Log hyperparams once** using `add_hparams` if you like. Because logs live per run, TensorBoard naturally separates trials and sweeps.

---

## 6) A tiny utility CLI (`tools/lh.py`) for GC, listing and TB

> Keep it simple, no servers/DBs. This CLI scans manifests to compute reachability.

```python
# tools/lh.py
import json, os, shutil, time
from pathlib import Path
import argparse
from lambda_hat.artifacts import Paths, ArtifactStore

def collect_reachable(paths: Paths) -> set[str]:
    reachable = set()
    for exp in (paths.experiments.glob("*/manifest.jsonl")):
        with exp.open() as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                for k in ("inputs", "outputs"):
                    for it in rec.get(k, []):
                        urn = it.get("urn")
                        if urn: reachable.add(urn)
    return reachable

def cmd_gc(args):
    paths = Paths.from_env(); paths.ensure()
    store = ArtifactStore(paths.store)
    ttl_days = int(os.environ.get("LAMBDA_HAT_TTL_DAYS", "30"))
    cutoff = time.time() - ttl_days * 86400

    # 1) prune run scratch/logs older than TTL (but keep artifacts + manifests)
    for run_dir in paths.experiments.glob("*/runs/*"):
        mtime = run_dir.stat().st_mtime
        if mtime < cutoff and run_dir.is_dir():
            # remove scratch; keep manifest/artifacts/tb
            for sub in ("scratch", "logs", "parsl"):
                p = run_dir / sub
                if p.exists(): shutil.rmtree(p, ignore_errors=True)

    # 2) prune unreachable objects in store older than TTL
    reachable = collect_reachable(paths)
    base = paths.store / "objects" / "sha256"
    removed = 0
    for d2 in base.glob("*/*/*"):
        meta_p = d2 / "meta.json"
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        urn = f"urn:lh:{meta.get('type','unknown')}:sha256:{meta['hash']['hex']}"
        if urn in reachable:
            continue
        # age gate
        if meta_p.stat().st_mtime < cutoff:
            shutil.rmtree(d2, ignore_errors=True)
            removed += 1
    print(f"GC removed {removed} unreachable objects (older than {ttl_days}d).")

def cmd_ls(args):
    paths = Paths.from_env(); paths.ensure()
    for exp in sorted(d.name for d in paths.experiments.glob("*") if d.is_dir()):
        print(f"[{exp}]")
        m = paths.experiments / exp / "manifest.jsonl"
        if not m.exists():
            print("  (no runs)");
            continue
        with m.open() as f:
            for line in f:
                rec = json.loads(line)
                print(f"  {rec['run_id']}  {rec.get('algo','?')}  phase={rec.get('phase','?')}")

def cmd_tb(args):
    paths = Paths.from_env(); paths.ensure()
    tb_root = paths.experiments / args.experiment / "tb"
    print(f"TensorBoard logdir: {tb_root}")

def main():
    ap = argparse.ArgumentParser("lh")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gc")
    sub.add_parser("ls")
    tbp = sub.add_parser("tb"); tbp.add_argument("experiment")
    args = ap.parse_args()
    if args.cmd == "gc": cmd_gc(args)
    elif args.cmd == "ls": cmd_ls(args)
    elif args.cmd == "tb": cmd_tb(args)

if __name__ == "__main__":
    main()
```

Usage:

```bash
python tools/lh.py ls
python tools/lh.py gc
tensorboard --logdir "$(python tools/lh.py tb my_experiment | awk '{print $NF}')"
```

---

## 7) How this fixes your current sprawl

* `logs/` → now per‑run: `experiments/<exp>/runs/<run_id>/logs/`
* `parsl_runinfo/` → now `.../runs/<run_id>/parsl/` (via `Config.run_dir`)
* `results/` → replaced by content‑addressed objects linked under `.../runs/<run_id>/artifacts/`
* `runs/` (top‑level) → folded into `experiments/<exp>/runs/`
* `temp_parsl_config/` → unnecessary; Parsl uses `run_dir`, config is in code

Nothing writes top‑level junk anymore; **one** home (`artefacts/` by default) contains everything, split cleanly into `store/`, `experiments/`, and `scratch/`.

---

## 8) Inter‑artefact dependencies (1 → N)

* **Targets** (synthetic NN + data) are stored once in the **store** (`type="target"`).
* Every **fit/sampler/VI run** writes `type="fit"` objects that *reference* the target’s URN in their `meta.target`.
* The **run manifest** captures both the input URN (target) and output URN(s).
* To find “all fits for a target”, scan `manifest.jsonl` in the experiment (or all experiments) for `inputs[].urn == <target_urn>`. (You can add a helper, but it’s one grep away.)

This makes the dependency DAG obvious to both your scripts and you, without a DB server.

---

## 9) Single‑run CLI vs Parsl orchestrations

* **Manual single run**
  Users keep their normal CLI calls. The only visible change: results land under `experiments/<exp>/runs/<run_id>/` and TB is ready at `experiments/<exp>/tb/`.

* **Parsl multi‑run**
  The orchestrator creates a **RunContext per trial** (so each trial has its own `run_id`, `parsl/`, `tb/`, `artifacts/`). Because **all commit to the same `store/`**, deduplication comes for free and the GC later can safely prune unreachable objects.

* **Optuna ask‑tell on disk**
  Keep your existing files under `experiments/<exp>/optuna/` so the trial directories and TB sit next to it. No servers.

---

## 10) Practical conventions (recommended)

* **Run id format:** `YYYYmmddTHHMMSSZ-<algo>[-<tag>]-<rand6>`
  Easy to sort by time; readable; unique enough for HPC concurrency.

* **URIs:** `urn:lh:<type>:sha256:<hash>`
  Short human hint: display first 12 hex chars in UIs; keep full in manifests.

* **Metrics naming:**

  * VI: `vi/elbo`, `vi/elbo_best`, `vi/kl_q_p`, `vi/recon`, `grad/norm`, `optim/lr`, `time/step_sec`
  * Sampling: `sampler/acceptance`, `sampler/ess`, `sampler/rhat`, `sampler/wall_sec`

* **Atomicity on network filesystems:**
  Write payloads to `ctx.scratch_dir`, then `put_dir()` copies → hashes → atomically moves into the store. This avoids partial objects.

---

## 11) Migration note

Back‑compat is not a priority. If you want, add a one‑off script that creates an experiment named `legacy/` and moves your old `results/` into `experiments/legacy/runs/...` while committing them to the store. But you don’t need to; you can start fresh today.

---

## 12) Quick start checklist

1. Add `.envrc` above and `direnv allow`.
2. Vendor `lambda_hat/artifacts.py` and `tools/lh.py`.
3. Modify `build_target.py` and `sample.py` as in §3.
4. In Parsl configs, set `run_dir=ctx.parsl_dir`.
5. Run a target build:

   ```bash
   python entrypoints/build_target.py --experiment=toy --tag=mlp_w50_seed42 > /tmp/target_urn.txt
   ```
6. Fit with VI:

   ```bash
   python entrypoints/sample.py --experiment=toy --algo=mfvi --target $(cat /tmp/target_urn.txt)
   ```
7. Open TB:

   ```bash
   tensorboard --logdir "$LAMBDA_HAT_EXPERIMENTS/toy/tb"
   ```
8. Clean up after exploration:

   ```bash
   python tools/lh.py gc
   ```

---

### Why this hits your goals

* **Cleaner & comprehensible**: one root, predictable per‑experiment/per‑run layout, Parsl confined to run dir.
* **Content‑addressed reuse**: keep your current good idea, but stop scattering it.
* **Parsl‑friendly**: every task has a scratch and a commit point; no global mutable state.
* **Disk‑only coordination**: JSONL manifests, atomic renames; no servers.
* **TensorBoard integration**: first‑class, per run and per experiment, easy to nuke later.
* **Easy cleanup**: `lh gc` prunes scratch and unreachable objects by TTL; no mystery “where did that go?”.

If you want, I can also sketch a tiny `query.py` to list “all runs that used target X” or “best ELBO in experiment Y,” but the structure above already makes those trivial with a 20‑line script.
