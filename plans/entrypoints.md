You’re right: `lambda_hat/entrypoints` mixes *parsing*, *routing*, and *business logic*, and even the naming varies (`lh`, `lambda-hat-promote`, `parsl-llc`, etc.). For example:

* `build_target.py` and `sample.py` both parse CLI **and** run the whole workflow inside the same file (heavy business logic in entrypoints).
* `parsl_llc.py` shells out to the other entrypoints via `bash_app` invocations of `python -m lambda_hat.entrypoints.*`, coupling orchestration to the CLI surface.
* `lh.py` defines a separate `lh` program with its own parser and subcommands (`gc/ls/tb`), diverging from the other names.
* `lambda-hat-promote` is yet another name, with its own parser and subcommands (`single/gallery`).

Meanwhile, in the *other* package in this repo (`llc`) you already have a clean “thin CLI / fat commands” pattern using Click: the CLI group does routing only; command modules hold the work. Let’s copy that architecture.

---

## Goal (single‑dev, lowest maintenance)

1. **One CLI**: `lambda-hat …`
2. **Thin CLI, fat commands**: parsing + routing only in `lambda_hat/cli`, all work in `lambda_hat/commands`.
3. **No back-compat shims**: delete/stop installing the old `entrypoints` binaries.
4. **Orchestration calls Python, not shell**: Parsl apps call command functions directly (as `@python_app`), not `python -m …`.

---

## Target layout

```text
lambda_hat/
  cli/
    __init__.py                 # Click group: only routing, no work
  commands/
    build_cmd.py                # build_entry(args: dict) -> None / dict
    sample_cmd.py               # sample_entry(args: dict) -> None / dict
    promote_cmd.py              # promote_entry / promote_gallery_entry
    artifacts_cmd.py            # gc_entry / ls_entry / tb_entry
    parsl_cmd.py                # orchestration entry (optional Click subcommands)
  # existing core modules stay where they are (artifacts, targets, promote/core, …)

# remove:
lambda_hat/entrypoints/         # delete the folder after migrating
```

This mirrors the pattern already used by `llc` (`llc/cli` + `llc/commands`), which keeps the CLI thin and business logic testable.

---

## Naming & UX rules

* **Single binary**: `lambda-hat`.
* **Verb-first subcommands** (simple, memorable):

  * `lambda-hat build` (Stage A)
  * `lambda-hat sample` (Stage B)
  * `lambda-hat promote [single|gallery]`
  * `lambda-hat artifacts [gc|ls|tb]`
  * `lambda-hat parsl [run|…]` (optional)
* **Consistent args**: `--config-yaml`, `--target-id`, `--experiment`, `--runs-root`, etc. (these already exist in today’s entrypoints).
* **Env keys**: keep `LAMBDA_HAT_*` (already used by `lh.py`’s GC paths).

---

## Mechanical refactor (surgical, fast)

### 1) Create the thin Click CLI

```python
# lambda_hat/cli/__init__.py
from __future__ import annotations
import click

@click.group()
def cli():
    "Lambda-Hat command-line interface (thin router)."
    # Put global env toggles here if you need them.
    # (Keep minimal; business logic belongs in commands.)

@cli.command()
@click.option("--config-yaml", required=True, type=click.Path(exists=True))
@click.option("--target-id", required=True)
@click.option("--experiment", required=False)
def build(**kwargs):
    "Stage A: build a target artifact."
    from lambda_hat.commands.build_cmd import build_entry
    return build_entry(kwargs)

@cli.command()
@click.option("--config-yaml", required=True, type=click.Path(exists=True))
@click.option("--target-id", required=True)
@click.option("--experiment", required=False)
def sample(**kwargs):
    "Stage B: run a sampler on a target."
    from lambda_hat.commands.sample_cmd import sample_entry
    return sample_entry(kwargs)

@cli.group()
def artifacts():  # lambda-hat artifacts <sub>
    "GC, list, and TB helpers for artifacts."

@artifacts.command("gc")
def artifacts_gc():
    from lambda_hat.commands.artifacts_cmd import gc_entry
    return gc_entry()

@artifacts.command("ls")
def artifacts_ls():
    from lambda_hat.commands.artifacts_cmd import ls_entry
    return ls_entry()

@artifacts.command("tb")
@click.argument("experiment")
def artifacts_tb(experiment):
    from lambda_hat.commands.artifacts_cmd import tb_entry
    return tb_entry(experiment)

@cli.group()
def promote():    # lambda-hat promote <sub>
    "Copy plots into galleries / stable locations."

@promote.command("single")
@click.option("--runs-root", required=True, type=click.Path(exists=True))
@click.option("--samplers", required=True)
@click.option("--outdir", required=True, type=click.Path())
@click.option("--plot-name", default="trace.png", show_default=True)
def promote_single(**kwargs):
    from lambda_hat.commands.promote_cmd import promote_entry
    return promote_entry(kwargs)

@promote.command("gallery")
@click.option("--runs-root", required=True, type=click.Path(exists=True))
@click.option("--samplers", required=True)
@click.option("--outdir", default="runs/promotion", show_default=True, type=click.Path())
@click.option("--plot-name", default="trace.png", show_default=True)
@click.option("--snippet-out", default=None, type=click.Path())
def promote_gallery(**kwargs):
    from lambda_hat.commands.promote_cmd import promote_gallery_entry
    return promote_gallery_entry(kwargs)
```

This is structurally identical to how `llc/cli/__init__.py` routes to `llc.commands.*`, which is a pattern you already use successfully.

### 2) Move “business logic” out of entrypoints into commands

* **From** `lambda_hat/entrypoints/build_target.py` → **to** `lambda_hat/commands/build_cmd.py`
  Keep the core workflow (OmegaConf load, Paths/ArtifactStore setup, seed handling, call into `targets.build_target`, store/save). Today all of this lives inside the entrypoint, tightly coupled to `argparse`.

* **From** `lambda_hat/entrypoints/sample.py` → **to** `lambda_hat/commands/sample_cmd.py`
  Move the sampler run, diagnostics, ArviZ conversion, and output writing. Again, it’s currently a big `main()` behind `argparse`.

* **From** `lambda_hat/entrypoints/promote.py` → **to** `lambda_hat/commands/promote_cmd.py`
  You already have pure helpers in `lambda_hat/promote/core.py`. The command module should be a thin wrapper around those.

* **From** `lambda_hat/entrypoints/lh.py` → **to** `lambda_hat/commands/artifacts_cmd.py`
  Keep `gc/ls/tb` as three functions (`gc_entry/ls_entry/tb_entry`) that use `Paths` and `ArtifactStore`. The logic is already clean; it’s just living in an entrypoint.

Here is a concrete sketch for *artifacts*:

```python
# lambda_hat/commands/artifacts_cmd.py
from __future__ import annotations
import json, os, shutil, time
from lambda_hat.artifacts import ArtifactStore, Paths

def _collect_reachable(paths: Paths) -> set[str]:
    reachable = set()
    for exp in paths.experiments.glob("*/manifest.jsonl"):
        for line in exp.read_text().splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            for k in ("inputs", "outputs"):
                for it in rec.get(k, []):
                    urn = it.get("urn")
                    if urn: reachable.add(urn)
    return reachable

def gc_entry(ttl_days_env="LAMBDA_HAT_TTL_DAYS") -> None:
    paths = Paths.from_env(); paths.ensure()
    store = ArtifactStore(paths.store)
    ttl_days = int(os.environ.get(ttl_days_env, "30"))
    cutoff = time.time() - ttl_days * 86400

    # prune run scratch/logs older than TTL
    for run_dir in paths.experiments.glob("*/runs/*"):
        try: mtime = run_dir.stat().st_mtime
        except FileNotFoundError: continue
        if mtime < cutoff and run_dir.is_dir():
            for sub in ("scratch", "logs", "parsl"):
                p = run_dir / sub
                if p.exists(): shutil.rmtree(p, ignore_errors=True)

    # prune unreachable objects in store older than TTL
    reachable = _collect_reachable(paths)
    base = paths.store / "objects" / "sha256"
    removed = 0
    for d2 in base.glob("*/*/*"):
        meta_p = d2 / "meta.json"
        try: meta = json.loads(meta_p.read_text())
        except Exception: continue
        urn = f"urn:lh:{meta.get('type','unknown')}:sha256:{meta['hash']['hex']}"
        if urn in reachable: continue
        if meta_p.stat().st_mtime < cutoff:
            shutil.rmtree(d2, ignore_errors=True)
            removed += 1
    print(f"GC removed {removed} unreachable objects (older than {ttl_days}d).")

def ls_entry() -> None:
    paths = Paths.from_env(); paths.ensure()
    for exp in sorted(d.name for d in paths.experiments.glob("*") if d.is_dir()):
        print(f"[{exp}]")
        m = paths.experiments / exp / "manifest.jsonl"
        if not m.exists(): print("  (no runs)"); continue
        for line in m.read_text().splitlines():
            rec = json.loads(line)
            print(f"  {rec['run_id']}  {rec.get('algo','?')}  phase={rec.get('phase','?')}")

def tb_entry(experiment: str) -> None:
    paths = Paths.from_env(); paths.ensure()
    print(f"TensorBoard logdir: {paths.experiments / experiment / 'tb'}")
```

This is 1:1 the logic in `lh.py`, just moved to a command module.

### 3) Make Parsl apps call Python, not CLI

In `parsl_llc.py`, your `bash_app`s spawn `python -m lambda_hat.entrypoints.*`. Replace the shell strings with `@python_app` functions that call the command functions directly. This removes the dependency on entrypoint names and makes refactors invisible to orchestration. Today’s version: `bash_app → "python -m lambda_hat.entrypoints.build_target …"` / `"…sample …"`.

Sketch:

```python
# lambda_hat/commands/parsl_cmd.py
from parsl import python_app

@python_app
def build_target_app(cfg_yaml: str, target_id: str, experiment: str|None):
    from lambda_hat.commands.build_cmd import build_entry
    return build_entry({"config_yaml": cfg_yaml, "target_id": target_id, "experiment": experiment})

@python_app
def run_sampler_app(cfg_yaml: str, target_id: str, experiment: str|None):
    from lambda_hat.commands.sample_cmd import sample_entry
    return sample_entry({"config_yaml": cfg_yaml, "target_id": target_id, "experiment": experiment})
```

If you still need SLURM shell wrappers for environment reasons, keep them minimal and only around *launch*, not business logic.

### 4) Package a single script

Update *one* `pyproject.toml` to expose the new CLI and include the package:

```toml
[tool.setuptools.packages.find]
include = ["llc*", "lambda_hat*"]  # ensure we ship both packages

[project.scripts]
llc = "llc.cli:cli"                # already present
lambda-hat = "lambda_hat.cli:cli"  # new: one binary
```


---

## What this fixes (concretely)

* **No more “logic in entrypoints”**: `build`, `sample`, `promote`, `artifacts` all live in `lambda_hat/commands/*`. (Before: `build_target.py`, `sample.py`, `promote.py`, `lh.py` mixed parsers and logic.)
* **One naming scheme**: `lambda-hat` + verb subcommands. (Before: `lh`, `lambda-hat-promote`, `parsl-llc`, etc.)
* **Parsl decoupled from CLI**: orchestration no longer depends on exact module paths or `python -m …` strings. (Before: hardcoded `bash_app` command strings.)
* **Testability**: command functions can be unit tested by passing dicts (no CLI parsing in the way).
* **Fewer moving parts**: one script to install and remember.

---

## Minimal migration steps (do-now checklist)

1. **Create** `lambda_hat/cli/__init__.py` and `lambda_hat/commands/*` as above.
2. **Cut & paste** logic from:

   * `entrypoints/build_target.py → commands/build_cmd.py`,
   * `entrypoints/sample.py → commands/sample_cmd.py`,
   * `entrypoints/promote.py → commands/promote_cmd.py`,
   * `entrypoints/lh.py → commands/artifacts_cmd.py`.
     Keep function names `*_entry` so signatures are uniform.
3. **Refactor Parsl orchestration** to use `@python_app` calling those functions (remove `bash_app` shell calls).
4. **Edit `pyproject.toml`** to include `lambda_hat*` and add the `lambda-hat` console script.
5. **Delete** `lambda_hat/entrypoints/` and any installers that referenced them.
6. **Smoke-test**:

   * `uv run lambda-hat build --config-yaml … --target-id …`
   * `uv run lambda-hat sample --config-yaml … --target-id …`
   * `uv run lambda-hat promote gallery --runs-root runs --samplers sgld,hmc,mclmc`
   * `uv run lambda-hat artifacts gc`

---

## Optional niceties (only if they *reduce* future toil)

* **Centralize config reading** (ΩConf load + validation) in a small helper used by `build_entry`/`sample_entry`, so both commands can’t drift.
* **Common logging setup** in the CLI group (mirroring how `llc` does this).
* **Keep `lambda_hat/__main__.py`** that simply calls the Click group, so `python -m lambda_hat …` works without extra scripts.

---


Short answer: it’s **not** obvious you should collapse *everything* into one CLI, and the biggest win isn’t “one binary vs many”, it’s “one set of command functions vs many copies of logic”.

Let me break it down in terms that actually affect your maintenance burden.

---

## What you have today

Top-level scripts (from `pyproject.toml`):

* `lambda-hat-build-target` → `entrypoints/build_target.py`
* `lambda-hat-sample` → `entrypoints/sample.py`
* `lambda-hat-promote` → `entrypoints/promote.py`
* `lh` → `entrypoints/lh.py`
* `parsl-llc` → `entrypoints/parsl_llc.py`
* `parsl-optuna` → `entrypoints/parsl_optuna.py`

These split into:

1. **Primitive ops (Stage-level)**

   * Build target
   * Sample
   * Promote
   * Artifact GC/ls/tb

2. **Workflows (orchestration)**

   * Parsl N×M sweep (`parsl-llc`)
   * Parsl+Optuna BO loop (`parsl-optuna`)

Workflows already call into library code (`workflow_utils`, `runners`, `parsl_apps`, etc.), and only `parsl-llc` shells out to the other entrypoints via `bash_app`.

So the *real* problem is: primitive entrypoints mix CLI and business logic; workflow entrypoints partially do the right thing.

---

## Axis 1: Single CLI vs several — ergonomics

### One unified `lambda-hat`:

**Pros**

* One name to remember and document.
* One global set of options (logging, verbosity, config helpers).
* Easier to add common features (e.g. `--dry-run`, `--json`) once.
* Autocomplete / `--help` output shows the whole surface area.

**Cons**

* CLI gets deep and busy:
  e.g. `lambda-hat target build`, `lambda-hat target sample`, `lambda-hat workflow llc`, `lambda-hat workflow optuna`.
* Subcommand trees interact (global options vs per-subcommand options) → more subtle parser behaviour to think about.
* Slightly worse for high-level scripts in docs or Makefiles: `parsl-llc` is immediately clearer than `lambda-hat workflow llc`.

Net: fewer binaries, more CLI complexity.

### Several small CLIs (what you have now, but cleaned)

**Pros**

* Mental model:

  * *“Low-level primitives”* (build, sample, promote, artifacts)
  * *“Big workflows”* (`parsl-llc`, `parsl-optuna`)
* Each binary has a tight help page and signature.
* Explicit boundaries: workflows are “bigger things” that you treat differently, and you *already* think of them that way.

**Cons**

* More names in docs / scripts.
* If you don’t factor common parsing/behaviour out, you duplicate it (which is basically the current mess).

Net: UX is fine as long as logic is centralised. The number of binaries itself is not the maintenance killer.

---

## Axis 2: Maintenance & code structure

This is the one that actually matters given your constraints.

### What you want regardless of CLI count

* **No business logic in entrypoints.**
* **One set of command functions** in something like `lambda_hat.commands.*` that:

  * Take *data* (config objects, paths), not `argparse.Namespace`.
  * Return structured results or throw.
  * Are callable from:

    * CLI wrappers (for humans),
    * Parsl apps (for workflows),
    * Tests.

Once that’s in place, *whether* you have 1 or 4 console scripts is almost irrelevant compared to the existing duplication.

### Single CLI + command modules

Pros:

* You get forced into a coherent subcommand tree, so you’re less likely to grow random one-off binaries.
* Shared parser/Click group does the routing.

Cons:

* You still need separate *conceptual* grouping (subcommands for workflows vs primitives). The wiring is more complex than a few tiny wrappers.

### Multiple CLIs + command modules

Pros:

* Each console script is literally a 10–20 line wrapper calling a single command function:

  * `lambda-hat-build-target` calls `lambda_hat.commands.build_cmd.build_entry_from_argv(argv)`.
  * `lambda-hat-sample` calls `lambda_hat.commands.sample_cmd.sample_entry_from_argv(argv)`.
* `parsl-llc` and `parsl-optuna` stay as dedicated, easy-to-spot workflow entrypoints that call Python APIs, not other CLIs.
* You can drop/rename primitives later without touching workflows, because Parsl should be calling the *Python* functions, not `python -m ...`.

Cons:

* Slightly more boilerplate (5 wrappers instead of 1). But the wrappers do nothing interesting.

Net: centralising logic into `commands/` gives you 95% of the maintenance win. The remaining 5% is bikeshedding CLI layout.

---

## Axis 3: Orchestration & HPC

You absolutely **do not** want Parsl to depend on fragile CLI names. Right now `parsl_llc.py` uses `bash_app` calling `python -m lambda_hat.entrypoints.build_target` / `sample`. That’s brittle and is the worst coupling in the system.

Independently of “one vs many CLIs”, you should:

* Switch Parsl to `@python_app` calling shared command functions (`build_target_cmd`, `sample_cmd`) directly.
* Make all workflow entrypoints (`parsl-llc`, `parsl-optuna`) operate purely via Python APIs.

Once you do that, workflow entrypoints really become *just another thin CLI*, and it doesn’t matter if they’re separate binaries or subcommands — they’re not in the hot path for code changes.

---

## Axis 4: How often you actually use these

Realistically:

* You’ll call `parsl-llc` and `parsl-optuna` in automation / long runs.
* You’ll occasionally call `lambda-hat-build-target`, `lambda-hat-sample`, `lambda-hat-promote` for debugging/one-offs.
* `lh` is basically a GC/ls tool you use when poking at the store.

You are not building a tool for a team; you don’t need a “professional” monolithic CLI layout. You need **something you can remember and that doesn’t make future refactors painful**.

---

## Concrete recommendation

Given:

* You are the only user and dev.
* Backwards compat is a non-goal.
* Maintenance burden is your only real constraint.

I’d do this:

### 1. Introduce command modules (this is non-negotiable)

Create:

* `lambda_hat/commands/build_cmd.py`
* `lambda_hat/commands/sample_cmd.py`
* `lambda_hat/commands/promote_cmd.py`
* `lambda_hat/commands/artifacts_cmd.py`

Move *all* business logic out of `entrypoints/*.py` into those.

Also provide:

* `build_entry_from_args(args: Namespace)` / `build_entry(cfg: Dict)` etc.

And update Parsl to call these via `@python_app`. That removes the worst coupling.

### 2. Keep **multiple small entrypoints**, but make them *thin*

Keep separate console scripts, but each is a trivial wrapper:

* `lambda-hat-build-target` → uses `argparse` or Click, then calls `build_entry_from_args`.
* `lambda-hat-sample` → calls `sample_entry_from_args`.
* `lambda-hat-promote` → calls `promote_entry_from_args`.
* `lh` → calls `gc_entry/ls_entry/tb_entry`.

For workflows:

* Keep `parsl-llc` and `parsl-optuna` as separate binaries — they’re conceptually “run the big pipeline, please”. They already use library functions (`workflow_utils`, `runners`, `parsl_apps`), so you mostly just clean up any remaining shell calls.

This gives you:

* Clear distinction:

  * **Low-level tools**: `lambda-hat-build-target`, `lambda-hat-sample`, `lambda-hat-promote`, `lh`
  * **High-level workflows**: `parsl-llc`, `parsl-optuna`
* No duplicated logic across entrypoints.
* Minimal surface you actually need to think about when changing core behaviour: it lives in `commands/` and underlying libs.

### 3. Optionally add a unified “nice” CLI later

If you ever find yourself annoyed by the proliferation of small commands, you can *layer* a `lambda-hat` umbrella CLI on top that just delegates to the same command functions:

* `lambda-hat target build` → `build_cmd.build_entry()`
* `lambda-hat target sample`
* `lambda-hat artifacts gc`
* `lambda-hat promote gallery` / `single`

But you don’t need to commit to that now. It’s sugar on top of an already-clean command layer.

---

## Direct answer to your question

> Do we for sure wish to consolidate all the entry points into one?

No. That’s not the primary win, and forcing everything into a single binary will give you more CLI plumbing to maintain without much benefit.

What you **do** want, on solid grounds:

* **Single source of truth for “what a build/sample/promote/artifact command does”** (command modules).
* **Parsl and Optuna workflows talking to those command functions**, not shelling out.
* **Entry points (one or many) that are all 10–20 line wrappers with zero bespoke logic.**

If you follow that, the choice between “one CLI with subcommands” and “a few small binaries” is mostly aesthetic. Given your situation, I’d keep several small, focused entrypoints and prioritise cleaning up the command layer and the Parsl coupling first.
