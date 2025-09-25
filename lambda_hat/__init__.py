"""Lambda-Hat — LLC estimation via SGLD/HMC/MCLMC with ArviZ diagnostics."""

# Ensure resolvers are registered for any entrypoint importing lambda_hat.*
try:
    from .hydra_support import register_resolvers
    register_resolvers()
except Exception:
    # Keep imports cheap even if optional deps are missing
    pass
