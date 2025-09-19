# Caching Behavior

Each run is keyed by a hash of (normalized config, code fingerprint).

## How Cache Works

- **Code fingerprint:** If `LLC_CODE_VERSION` is set in environment → use it; otherwise → hash of all source files (`llc/**/*.py` + `pyproject.toml`)
- **Default behavior:** Skip if a run with the same key already exists
- **Cache location:** Works everywhere (local, Modal, SLURM) with the same logic

## Commands

```bash
# Normal run (uses cache if available)
uv run python -m llc run --preset=quick

# Force re-run even if cached
uv run python -m llc run --preset=quick --no-skip

# Override code version (for CI/Modal deployments)
LLC_CODE_VERSION=deploy-123 uv run python -m llc run
```

## Tips

- Editing any `llc/*.py` file changes the fingerprint automatically
- On Modal/CI, set `LLC_CODE_VERSION` to a build ID if you want explicit control
- The cache works everywhere (local, Modal, SLURM) with the same logic