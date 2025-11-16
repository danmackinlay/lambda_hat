# lambda_hat/losses.py
"""Loss function utilities for neural network models"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    pass


def as_dtype(x, dtype_str):  # 'float32' or 'float64'
    """Convert array or PyTree to specified dtype (ensures JAX arrays).

    For Equinox modules, only converts array leaves (not static components).
    """
    target_dtype = jnp.float32 if dtype_str == "float32" else jnp.float64

    def convert_leaf(arr):
        # Only convert if it's actually an array
        if isinstance(arr, (jnp.ndarray, jax.Array)):
            return jnp.asarray(arr, dtype=target_dtype)
        return arr

    return jax.tree.map(convert_leaf, x, is_leaf=lambda x: isinstance(x, (jnp.ndarray, jax.Array)))


def make_loss_fns(
    predict_fn: Callable,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    loss_type: str = "mse",
    noise_scale: float = 0.1,  # Required for t_regression
    student_df: float = 4.0,  # Required for t_regression
):
    """Create loss functions for both full data and minibatch

    Args:
        predict_fn: Prediction function with signature (params, X) -> predictions
                   For Equinox models: lambda model, X: model(X)
        X: Input data
        Y: Target data
        loss_type: Type of loss function ("mse" or "t_regression")
        noise_scale: Noise scale for t_regression
        student_df: Student t degrees of freedom for t_regression

    Returns:
        Tuple of (full_loss_fn, minibatch_loss_fn)
    """
    if loss_type == "mse":

        def full(params):
            pred = predict_fn(params, X)
            return jnp.mean((pred - Y) ** 2)

        def minibatch(params, Xb, Yb):
            pred = predict_fn(params, Xb)
            return jnp.mean((pred - Yb) ** 2)

    elif loss_type == "t_regression":
        s2 = noise_scale**2
        nu = student_df

        def neglogt(resid):
            return 0.5 * (nu + 1) * jnp.log1p((resid**2) / (nu * s2))

        def full(params):
            pred = predict_fn(params, X)
            return jnp.mean(neglogt(pred - Y))

        def minibatch(params, Xb, Yb):
            pred = predict_fn(params, Xb)
            return jnp.mean(neglogt(pred - Yb))
    else:
        raise ValueError(f"Unknown loss: {loss_type}")

    return full, minibatch
