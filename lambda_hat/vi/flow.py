# lambda_hat/vi/flow.py
"""Normalizing flow variational inference (stub for future implementation)."""

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from lambda_hat.sampling import SamplerRunResult
from lambda_hat.vi import registry


class _FlowAlgorithm:
    """Normalizing flow VI algorithm (not yet implemented)."""

    name = "flow"

    def run(
        self,
        rng_key: jax.random.PRNGKey,
        loss_batch_fn: Callable,
        loss_full_fn: Callable,
        wstar_flat: jnp.ndarray,
        unravel_fn: Callable[[jnp.ndarray], Any],
        data: Tuple[jnp.ndarray, jnp.ndarray],
        n_data: int,
        beta: float,
        gamma: float,
        vi_cfg: Any,
    ) -> SamplerRunResult:
        """Run normalizing flow VI (not yet implemented).

        Raises:
            NotImplementedError: Flow VI is not implemented yet
        """
        raise NotImplementedError(
            "Normalizing flow VI is not implemented yet. "
            "The algorithm is plumbed through the config system (vi.algo=flow) "
            "but the implementation is deferred to future work."
        )


def make_flow_algo() -> _FlowAlgorithm:
    """Factory function for flow VI algorithm."""
    return _FlowAlgorithm()


# Register the algorithm
registry.register("flow", make_flow_algo)
