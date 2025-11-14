# lambda_hat/vi/base.py
"""Base interfaces for pluggable variational inference algorithms."""

from typing import Any, Callable, Protocol, Tuple

import jax
import jax.numpy as jnp

from lambda_hat.types import SamplerRunResult


class VIAlgorithm(Protocol):
    """Protocol for variational inference algorithms.

    All VI algorithms must implement this interface to be compatible
    with the sampling framework.
    """

    name: str

    def run(
        self,
        rng_key: jax.random.PRNGKey,
        loss_batch_fn: Callable,  # (w_pytree, Xb, Yb) -> scalar
        loss_full_fn: Callable,  # (w_pytree) -> scalar
        wstar_flat: jnp.ndarray,  # (d,) flattened ERM solution
        unravel_fn: Callable[[jnp.ndarray], Any],  # flat -> pytree
        data: Tuple[jnp.ndarray, jnp.ndarray],  # (X, Y)
        n_data: int,
        beta: float,  # inverse temperature
        gamma: float,  # localizer strength
        vi_cfg: Any,  # VIConfig dataclass
    ) -> SamplerRunResult:
        """Run VI and return sampling result.

        Args:
            rng_key: JAX random key
            loss_batch_fn: Minibatch loss function (w_pytree, Xb, Yb) -> scalar
            loss_full_fn: Full dataset loss function (w_pytree) -> scalar
            wstar_flat: Flattened ERM solution (d,)
            unravel_fn: Function to convert flat array to pytree
            data: Tuple of (X, Y) data arrays
            n_data: Number of data points
            beta: Inverse temperature for tempering
            gamma: Localizer strength (Gaussian tether around w*)
            vi_cfg: VI configuration (VIConfig dataclass)

        Returns:
            SamplerRunResult with traces, timings, and work metrics
        """
        ...
