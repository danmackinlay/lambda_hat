# Caching Behavior

Each run is keyed by a hash of (normalized config, code fingerprint).

## How Cache Works

- **Code fingerprint:** If `LLC_CODE_VERSION` is set in environment → use it; otherwise → hash of all source files (`llc/**/*.py` + `pyproject.toml`)
- **Default behavior:** Skip if a run with the same key already exists
- **Cache location:** Works everywhere (local, Modal, SLURM) with the same logic
- **Schema validation:** Configs include a `config_schema` hash that is checked between client and worker (Modal/SLURM). If schema mismatches, the job fails early with a clear error to prevent silent client/worker version skew.

## Commands

```bash
# Normal run (uses cache if available)
uv run python -m lambda_hat run --preset=quick

# Force re-run even if cached
uv run python -m lambda_hat run --preset=quick --no-skip

# Override code version (for CI/Modal deployments)
LLC_CODE_VERSION=deploy-123 uv run python -m lambda_hat run
```

## Sampler-scoped hashing

When computing `run_id(cfg)` we:
1. Normalize the config (drop volatile fields).
2. **Remove all sampler-specific fields for samplers other than the selected one.**
3. Hash the normalized config + code version.

Effect: changing `hmc_*` fields does not change the hash for an `sgld` run (and vice-versa).

## Family IDs

`run_family_id(cfg)` hashes the normalized config **after** dropping:
- `samplers`
- *all* sampler-specific fields (for any sampler)

It intentionally **excludes code version**, making family comparisons stable across upgrades.

## Tips

- Editing any `llc/*.py` file changes the fingerprint automatically
- On Modal/CI, set `LLC_CODE_VERSION` to a build ID if you want explicit control
- The cache works everywhere (local, Modal, SLURM) with the same logic