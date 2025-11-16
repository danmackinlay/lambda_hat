# lambda_hat/vi/types.py
"""Type definitions for VI algorithms - flat-space only"""

from typing import Protocol, Tuple

import jax.numpy as jnp

Array = jnp.ndarray
Batch = Tuple[Array, Array]  # (X, Y) minibatch


class FlatObjective(Protocol):
    """Protocol for flat-space objective functions.

    All VI algorithms consume this interface. No model-space conversions.
    """

    def loss(self, w_flat: Array, batch: Batch) -> Array:
        """Compute scalar loss for flat parameters on a minibatch.

        Args:
            w_flat: Flat parameter vector (R^D)
            batch: Tuple of (X, Y) minibatch arrays

        Returns:
            Scalar loss value
        """
        ...

    def grad(self, w_flat: Array, batch: Batch) -> Array:
        """Compute gradient of loss for flat parameters on a minibatch.

        Args:
            w_flat: Flat parameter vector (R^D)
            batch: Tuple of (X, Y) minibatch arrays

        Returns:
            Gradient vector (R^D)
        """
        ...

    def value_and_grad(self, w_flat: Array, batch: Batch) -> Tuple[Array, Array]:
        """Compute both loss and gradient (optional convenience method).

        Args:
            w_flat: Flat parameter vector (R^D)
            batch: Tuple of (X, Y) minibatch arrays

        Returns:
            Tuple of (scalar loss, gradient vector)
        """
        ...
