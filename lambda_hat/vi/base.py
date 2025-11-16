# lambda_hat/vi/base.py
"""Base interfaces for pluggable variational inference algorithms - FLAT INTERFACE ONLY."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Protocol, Tuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from lambda_hat.vi.types import FlatObjective


class VIAlgorithm(Protocol):
    """Protocol for variational inference algorithms - FLAT INTERFACE ONLY.

    All VI algorithms must implement this interface to be compatible
    with the sampling framework. All operations are in flat R^D space only,
    no model-space pytree conversions.
    """

    name: str

    def run(
        self,
        rng_key: jax.random.PRNGKey,
        objective: "FlatObjective",  # Flat-space loss and gradient functions
        loss_full_flat: Callable[[jnp.ndarray], jnp.ndarray],  # Full-data loss in flat space
        wstar_flat: jnp.ndarray,  # (d,) flattened ERM solution
        data: Tuple[jnp.ndarray, jnp.ndarray],  # (X, Y)
        n_data: int,
        beta: float,  # inverse temperature
        gamma: float,  # localizer strength
        vi_cfg: Any,  # VIConfig dataclass
        whitener: Any = None,  # Optional whitener
    ) -> Dict[str, Any]:
        """Run VI and return results - FLAT INTERFACE ONLY.

        Args:
            rng_key: JAX random key
            objective: FlatObjective providing loss/grad in flat space
            loss_full_flat: Full-data loss in flat space (for MC estimation)
            wstar_flat: Flattened ERM solution (d,)
            data: Tuple of (X, Y) data arrays
            n_data: Number of data points
            beta: Inverse temperature for tempering
            gamma: Localizer strength (Gaussian tether around w*)
            vi_cfg: VI configuration (VIConfig dataclass)
            whitener: Optional whitener (geometry transformation)

        Returns:
            Dict with keys: lambda_hat, traces, extras, timings, work
        """
        ...
