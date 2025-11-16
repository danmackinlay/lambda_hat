Below is a concrete, opinionated plan to bring your docs back under control—**short, single‑source, and mechanically kept in sync with the code**. I’ve anchored the plan to the current repo layout and a few places where drift has already crept in.

---

## What’s broken today (quick audit)

* **Two doc trees**: you have both `docs/` *and* an `md/` folder with overlapping topics (e.g., `sgld.md`, `sweeps.md`, `vi_*`). This alone guarantees duplication and drift.
* **Stale internal references**: `docs/blackjax.md` says examples live in `lambda_hat/sampling.py`, but the repo actually ships `lambda_hat/sampling_runner.py`. That’s concrete drift.
* **CLI is richer than the docs show**: the unified CLI exposes `build`, `sample`, `artifacts {gc,ls,tb}`, and `promote …` subcommands, but there’s no single, authoritative CLI reference in your docs.
* **Config is hand‑explained, not generated**: you already own well‑structured Hydra/OmegaConf YAML for data/model/sampler, but the doc pages re-explain options instead of deriving them from source.
* **Pin drift risks**: the project pins `blackjax==1.2.5` in `pyproject.toml` and states the same in `CLAUDE.md`, yet pages like `docs/blackjax.md` carry free‑text API notes—these always rot when upstream changes.
* **Workflow topics are spread around**: Parsl/Optuna and output/parallelism are scattered (e.g., `config/parsl/*`, `optuna_demo.yaml`, and multiple markdown pages). Consolidation will help.

---

## Target state: **10 pages max**, each with an owner and a single source of truth

> Principle: **Reference > Tutorial > Explanation**, and anything that can be **generated or validated** must be.

### 0) Top‑level

* **README.md** – product overview + 90‑second quickstart. Link out to the 10 pages below. (Keep README high‑signal; no API minutiae.)

### 1) **Install**

* CPU / CUDA / optional FlowJAX extras with the exact `uv` extras already declared. Source of truth: `pyproject.toml` optional deps. Automation: a CI check that the doc’s dependency table is rendered from `pyproject`.

### 2) **Quickstart**

* “Build → Sample → Promote” happy path that mirrors the CLI (copy/pasteable snippets only). Source of truth: `lambda_hat/cli.py` help + smoke‑runnable snippets.

### 3) **CLI Reference** (auto‑generated)

* One page that dumps the `lambda-hat` Click command tree and options. Source of truth: introspecting Click at build time; **no hand‑typed flags**.

### 4) **Configuration Reference** (auto‑generated)

* Single page that renders config groups/fields for `data`, `model`, `sample/sampler/*`, and `teacher/…`, with defaults pulled directly from YAML. Include the posterior block (β/γ) semantics in one callout, referencing the actual implementation.

### 5) **Samplers**

* Short “how to run SGLD/HMC/MCLMC” with pointers into the config reference; **no re‑listing of flags**. Keep BlackJAX details to a “Compatibility” box that only states the pinned version.

### 6) **VI**

* Merge `vi_mfa.md` + `vi_normalizing_flow.md` into **vi.md** that explains how to use the plug‑in registry and when FlowJAX is required. Link to the protocol in `vi/base.py`.

### 7) **Workflows**

* One page that covers Parsl (local, SLURM, A100 profile) and Optuna orchestration, mapped to the shipped YAMLs. Keep it prescriptive with 2–3 copy/paste recipes.

### 8) **Artifacts & Output**

* Where runs go, what files look like, how to GC, list and point TensorBoard at them (`artifacts {gc,ls,tb}`). Mirror the CLI.

### 9) **Targets**

* Single explanation of the target families and data generators; push all knobs to the **Configuration Reference**. (Keep this conceptual, not a catalog.)

### 10) **Compatibility**

* A tiny page that lists the few pinned versions the code promises to work with (JAX, BlackJAX) and how the project enforces them (see “automation” below). Tie this to the repo’s “lowest maintenance burden” philosophy from `CLAUDE.md`.

---

## Filesystem changes (do this first)

1. **Delete** `md/` after merging its content into `docs/` (or vice‑versa). Keep **only** `docs/`.
2. **Rename/replace** pages:

   * `docs/blackjax.md` → **docs/compatibility.md** (no API mirrors; just pins + caveats).
   * `docs/output_management.md`, `parallelism.md`, `sweeps.md` → **docs/workflows.md** (sections).
   * `docs/sgld.md` → **docs/samplers.md** (SGLD + HMC + MCLMC in one place; flags link to config page).
   * `docs/vi_mfa.md` + `docs/vi_normalizing_flow.md` → **docs/vi.md**.
3. **Fix stale paths**: replace all references to `lambda_hat/sampling.py` with either the CLI command or the actual module (`sampling_runner.py`) when a code path is necessary.

---

## Make it “always current”: build‑time automation

Add a tiny `docs/` build script (invoked by CI) that **generates** the two reference pages and **fails** if pins or paths drift.

1. **CLI reference generator** (reads Click app):

   ````python
   # docs/_gen_cli.py
   from click.testing import CliRunner
   from lambda_hat.cli import cli

   def generate():
       r = CliRunner().invoke(cli, ["--help"])
       (Path("docs")/"cli.md").write_text("```text\n"+r.output+"\n```")

   if __name__ == "__main__":
       generate()
   ````

   Source of truth: `lambda_hat/cli.py`. This prevents hand‑edited flag tables.

2. **Config reference generator** (walks YAML):

   ````python
   # docs/_gen_config.py
   import yaml, textwrap, pathlib
   root = pathlib.Path("lambda_hat/conf")
   out = ["# Configuration Reference\n"]
   for p in sorted(root.rglob("*.yaml")):
       out += [f"## `{p.relative_to(root)}`", "```yaml", p.read_text(), "```", ""]
   pathlib.Path("docs/config.md").write_text("\n".join(out))
   ````

   Source of truth: `lambda_hat/conf/**`.

3. **Pin & path assertions**:

   * Assert that the BlackJAX pin in `pyproject.toml` equals the value displayed in `docs/compatibility.md`; fail CI otherwise.
   * Grep the docs for forbidden stale paths (e.g., `lambda_hat/sampling.py`); fail CI on match.

4. **Doctest/smoke run**:

   * Run a single minimal “quickstart” invocation in CI: `lambda-hat --help` and a dry‑run that doesn’t touch GPUs (or run a tiny CPU build with `small.yaml`). This validates the snippets without long jobs.

5. **Link doc metadata to commits**:

   * You already register OmegaConf resolvers for `git_sha`/`hostname`. Surface these in generated config docs or “About this build” boxes to help users match docs to code snapshots.

> Wire these into a single `uv run python docs/_build.py` invoked by CI; never hand‑edit `docs/cli.md` or `docs/config.md`.

---

## Writing style (so pages stay *fast and focused*)

* **Top‑box “At a glance”**: 5 bullets with the essentials; everything else can be collapsed.
* **One task per page**: “How to run MCLMC”, not “Everything about MCLMC”.
* **No option catalogs in narrative pages**: link to **config.md** or **cli.md** instead.
* **Copy/paste first**: every code block must be runnable as‑is.
* **Hard limits**: 300–700 words per page; if you exceed, split.

---

## Examples of the new pages

### A) `docs/compatibility.md` (replacement for `blackjax.md`)

> **Pins**
>
> * JAX `~=0.7.2` (CPU or CUDA12 extras)
> * BlackJAX `==1.2.5`
> * FlowJAX (optional; required for flow‑based VI)
>   These are enforced in `pyproject.toml`. If a PR bumps a pin, CI will fail unless this page gets regenerated with the new values.

> **Notes**
>
> * HMC/MCLMC usage follows BlackJAX 1.2.5. Avoid copying examples from `main` docs.
> * If you see API mismatches, regenerate docs and re‑run the quickstart smoke test.

*(No API signatures here—those belong to upstream.)*

### B) `docs/samplers.md`

* **What you choose**: `sampler: {sgld|hmc|mclmc}` in YAML.
* **Where to configure**: `lambda_hat/conf/sample/sampler/*.yaml` (see **Configuration Reference**).
* **Run it**:

  ```bash
  lambda-hat sample --config-yaml path/to/composed.yaml --target-id tgt_xxx
  ```

  Options are in **CLI Reference**.

### C) `docs/workflows.md`

* **Local/threads** profile and **SLURM A100** profile with copy/paste configs from `config/parsl/local.yaml` and `config/parsl/slurm/gpu-a100.yaml`, plus a 3‑step Optuna example derived from `config/optuna_demo.yaml`.

---

## Migration map (from old → new)

| Old page                                                             | Action                                                    |
| -------------------------------------------------------------------- | --------------------------------------------------------- |
| `docs/blackjax.md`                                                   | Replace with `docs/compatibility.md` (no API mirrors).    |
| `docs/sgld.md`                                                       | Merge into `docs/samplers.md`.                            |
| `docs/vi_mfa.md`, `docs/vi_normalizing_flow.md`                      | Merge into `docs/vi.md`.                                  |
| `docs/parallelism.md`, `docs/sweeps.md`, `docs/output_management.md` | Merge into `docs/workflows.md`.                           |
| `md/*`                                                               | Delete after content merge; keep single tree in `docs/`.  |

---

## “Drift prevention” checklist (add as `docs/CONTRIBUTING.md`)

* [ ] If you **rename/move** a module, run `uv run python docs/_checks.py` (CI will too). This catches stale paths like `lambda_hat/sampling.py`.
* [ ] If you **change CLI flags**, the **CLI Reference** must be regenerated (script does this automatically).
* [ ] If you **change defaults in YAML**, regenerate **Configuration Reference** (script).
* [ ] If you **bump pins**, the **Compatibility** page must be regenerated (script checks `pyproject.toml`).

---

## Small but high‑impact fixes to apply immediately

1. Replace all mentions of `lambda_hat/sampling.py` → refer to CLI or `sampling_runner.py` where a file path is truly needed.
2. Delete `md/` after merging into `docs/`.
3. Collapse BlackJAX API notes into a **Compatibility** callout (no signatures), grounded by the BlackJAX pin in `pyproject.toml`.
4. Add `lambda-hat` CLI reference generation + YAML rendering scripts under `docs/` and wire them into CI.
5. Put your “**reduce maintenance burden**” policy in the docs front‑matter so future edits don’t balloon scope.

---

## Why this will stay fast, focused, current

* **Fast**: 10 pages max, each < ~700 words, all action‑oriented.
* **Focused**: narrative pages explain *what* and *why*; flags and defaults live in generated references.
* **Current**: the only hand‑edited pages are overview/how‑to; everything else is **generated from running code** or YAML, and CI fails if it drifts.

If you want, I can draft the two generator scripts and a minimal CI job next, using the exact paths in this repo so you can drop them in and turn the crank.
