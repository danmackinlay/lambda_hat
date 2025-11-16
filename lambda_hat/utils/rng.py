# lambda_hat/utils/rng.py
"""PRNG key normalization utilities for JAX typed keys."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def ensure_typed_key(key):
    """Normalize ints and legacy uint32[2] keys to typed threefry keys.

    This function must be called on the host (outside jit/vmap/pmap) as it
    performs shape-changing conversions that cannot be traced.

    Args:
        key: One of:
            - Python int or numpy integer (seed)
            - Legacy JAX key: uint32[2] array
            - Batch of legacy keys: uint32[..., 2] array
            - Already-typed KeyArray (returned as-is)

    Returns:
        Typed KeyArray using threefry2x32 implementation

    Examples:
        >>> key = ensure_typed_key(42)  # From int
        >>> key = ensure_typed_key(jax.random.PRNGKey(0))  # From legacy
        >>> keys = ensure_typed_key(jax.random.split(jax.random.PRNGKey(0), 4))

    References:
        - JAX typed keys: https://docs.jax.dev/en/latest/jax.random.html
        - FlowJAX typed key usage: https://danielward27.github.io/flowjax/
    """
    # From plain python ints
    if isinstance(key, (int, np.integer)):
        return jr.key(int(key))  # Respects jax_default_prng_impl

    # Legacy: uint32[2] or batch of them
    if isinstance(key, jax.Array) and key.dtype == jnp.uint32:
        if key.ndim == 1 and key.shape[-1] == 2:
            # Single legacy key -> typed key for threefry
            return jr.wrap_key_data(key, impl="threefry2x32")
        if key.ndim >= 2 and key.shape[-1] == 2:
            # Vectorized conversion for batches of legacy keys
            return jax.vmap(lambda k: jr.wrap_key_data(k, impl="threefry2x32"))(key)

    # Already typed KeyArray (or unknown format - pass through)
    return key
