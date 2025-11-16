# Documentation Overview

Lambda-Hat uses a streamlined documentation structure with **auto-generated references** and **consolidated narrative pages**.

---

## Quick Navigation

| Page | Type | Purpose |
|------|------|---------|
| [CLI Reference](./cli.md) | Auto-generated | All command-line options |
| [Configuration Reference](./config.md) | Auto-generated | YAML schema and defaults |
| [Experiments Guide](./experiments.md) | Guide | Composing experiments with overrides |
| [Samplers](./samplers.md) | Guide | HMC, MCLMC, SGLD, VI usage |
| [Variational Inference](./vi.md) | Guide | VI overview, shared config, usage |
| [MFA VI](./vi_mfa.md) | Guide | Mixture of factor analyzers details |
| [Flow VI](./vi_flow.md) | Technical | Normalizing flows, JAX/vmap notes |
| [Workflows](./workflows.md) | Guide | Parsl, sweeps, artifacts |
| [Compatibility](./compatibility.md) | Reference | Version pins and API notes |
| [CONTRIBUTING](./CONTRIBUTING.md) | Meta | Documentation maintenance |

---

## Maintenance Philosophy

**Auto-generate where possible, write where necessary.**

### Auto-Generated Docs

These files are **never edited by hand**:

* `cli.md` — Generated from `lambda_hat/cli.py`
* `config.md` — Generated from `lambda_hat/conf/**/*.yaml`

To regenerate:
```bash
uv run python docs/_build.py
```

**CI will fail if they're out of sync.**

### Narrative Docs

These files are **manually written**:

* `samplers.md` — Sampler usage guide
* `vi.md` — VI overview and shared concepts
* `vi_mfa.md` — MFA VI algorithm details
* `workflows.md` — Workflow orchestration
* `compatibility.md` — Version compatibility notes
* `experiments.md` — Experiments and config composition guide
* `methodology.md` — Conceptual background
* `optuna_workflow.md` — Optuna guide
* `output_management.md` — Artifact system details

**Guidelines**:
- Keep pages 300-700 words
- Copy/paste first: all code blocks must be runnable
- Link to references (cli.md, config.md) instead of duplicating

---

## For Contributors

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the drift prevention checklist.

**Before committing**:
```bash
uv run python docs/_build.py  # Regenerate + check
git add docs/
git commit -m "docs: your changes"
```

**CI will verify**:
1. Generated docs are in sync
2. No stale path references
3. Version pins match pyproject.toml

---

## Architecture

```
docs/
├── _build.py              # Orchestrator (runs generators + checks)
├── _gen_cli.py            # CLI reference generator
├── _gen_config.py         # Config reference generator
├── _checks.py             # Drift detection (stale paths, pins)
│
├── cli.md                 # [Auto] CLI reference
├── config.md              # [Auto] Config reference
│
├── samplers.md            # [Manual] Sampler guide
├── vi.md                  # [Manual] VI overview
├── vi_mfa.md              # [Manual] MFA algorithm details
├── vi_flow.md             # [Technical] Flow algorithm details
├── workflows.md           # [Manual] Workflow guide
├── compatibility.md       # [Manual] Compatibility notes
├── experiments.md         # [Manual] Experiments guide
├── methodology.md         # [Manual] Conceptual background
├── optuna_workflow.md     # [Manual] Optuna guide
├── output_management.md   # [Manual] Artifact system
│
├── CONTRIBUTING.md        # [Meta] Maintenance guide
└── index.md               # [Meta] This file
```

---

## Design Principles

1. **Single source of truth** — No duplication between docs and code
2. **Auto-generate references** — CLI and config docs are never hand-written
3. **Consolidate narratives** — One page per topic (samplers, VI, workflows)
4. **Fail fast** — CI catches drift immediately
5. **Delete over archive** — Stale docs are worse than no docs
