"""Lambda-Hat — LLC estimation via SGLD/HMC/MCLMC with ArviZ diagnostics."""

# Configure JAX to use threefry2x32 PRNG implementation globally
# This must happen before any jax.random calls to ensure FlowJAX compatibility
# and consistent behavior across Parsl workers. Threefry is preferred over RBG
# because RBG has unusual vmap batching semantics.
# See: https://docs.jax.dev/en/latest/jax.random.html
import jax

jax.config.update("jax_default_prng_impl", "threefry2x32")

# Ensure resolvers are registered for any entrypoint importing lambda_hat.*
try:
    from .omegaconf_support import register_resolvers

    register_resolvers()
except Exception:
    # Keep imports cheap even if optional deps are missing
    pass
