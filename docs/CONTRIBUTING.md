# Contributing to Documentation

This guide ensures documentation stays **fast, focused, and current**.

---

## Drift Prevention Checklist

Before submitting changes, run:

```bash
uv run python docs/_build.py
```

This will:
1. Regenerate CLI reference (`docs/cli.md`)
2. Regenerate configuration reference (`docs/config.md`)
3. Check for stale path references
4. Verify BlackJAX pin consistency

**CI will fail if generated docs are out of sync.**

---

## Manual Checks

### When you rename/move a module

- [ ] Run `uv run python docs/_checks.py` to catch stale paths
- [ ] Update references in narrative docs if needed

### When you change CLI flags

- [ ] The **CLI Reference** will be regenerated automatically
- [ ] Do **not** hand-edit `docs/cli.md` — it's auto-generated

### When you change YAML defaults

- [ ] The **Configuration Reference** will be regenerated automatically
- [ ] Do **not** hand-edit `docs/config.md` — it's auto-generated

### When you bump dependency pins

- [ ] Update `docs/compatibility.md` with new version numbers
- [ ] The checks will verify `pyproject.toml` matches the docs

---

## Documentation Structure

Lambda-Hat uses a **10-page-max** documentation structure:

1. **README.md** — Product overview + quickstart
2. **cli.md** — Auto-generated CLI reference
3. **config.md** — Auto-generated configuration reference
4. **samplers.md** — How to run SGLD/HMC/MCLMC
5. **vi.md** — Variational inference (MFA + Flow)
6. **workflows.md** — Parsl orchestration, sweeps, artifact management
7. **compatibility.md** — Version pins and API notes
8. **methodology.md** — Conceptual background
9. **optuna_workflow.md** — Hyperparameter optimization
10. **output_management.md** — Detailed artifact system architecture

---

## Writing Guidelines

### Keep pages fast and focused

* **Top-box "At a glance"**: 5 bullets with essentials
* **One task per page**: "How to run MCLMC", not "Everything about MCLMC"
* **No option catalogs**: Link to `config.md` or `cli.md` instead
* **Copy/paste first**: Every code block must be runnable as-is
* **Hard limits**: 300–700 words per page; if you exceed, split

### Reference > Tutorial > Explanation

* **Reference docs** (CLI, config) are auto-generated
* **Tutorial docs** (samplers, workflows) show copy/paste recipes
* **Explanation docs** (methodology) provide conceptual background

### No duplication

* **Single source of truth** for all information
* If information exists in code (YAML, CLI), **generate** the docs
* If information exists in narrative, **link** to it

---

## File Organization

### Auto-generated (do not edit)

* `docs/cli.md` — Generated from `lambda_hat/cli.py`
* `docs/config.md` — Generated from `lambda_hat/conf/**/*.yaml`

### Narrative (edit freely)

* `docs/samplers.md`
* `docs/vi.md`
* `docs/workflows.md`
* `docs/compatibility.md`
* `docs/methodology.md`
* `docs/optuna_workflow.md`

### Internal (automation)

* `docs/_gen_cli.py` — CLI generator
* `docs/_gen_config.py` — Config generator
* `docs/_checks.py` — Drift checks
* `docs/_build.py` — Build orchestrator

---

## Committing Changes

### Before committing

1. Run `uv run python docs/_build.py`
2. Commit **both** narrative changes **and** generated files
3. CI will verify they're in sync

### Example workflow

```bash
# Make changes to docs/samplers.md
vim docs/samplers.md

# Regenerate auto docs
uv run python docs/_build.py

# Commit everything
git add docs/
git commit -m "docs: update samplers page with MCLMC example"
```

---

## Philosophy

Lambda-Hat prioritizes **reducing maintenance burden** over other concerns:

* **Break things to reduce burden** — no sacred cows
* **Generate, don't write** — automate documentation where possible
* **Delete, don't archive** — stale docs are worse than no docs
* **Fail fast** — CI catches drift immediately

If a doc page is rarely used or hard to maintain, **delete it** and add a one-liner to a higher-level page instead.

---

## Questions?

* **Bug reports**: [GitHub Issues](https://github.com/anthropics/claude-code/issues)
* **Documentation issues**: Same place
* **Unclear docs**: File an issue with the specific confusion
