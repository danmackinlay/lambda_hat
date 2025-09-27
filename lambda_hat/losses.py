# llc/losses.py
"""Loss function utilities adapted for Haiku models"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    pass


def as_dtype(x, dtype_str):  # 'float32' or 'float64'
    """Convert array or PyTree to specified dtype"""
    target_dtype = jnp.float32 if dtype_str == "float32" else jnp.float64
    return jax.tree.map(lambda arr: arr.astype(target_dtype), x)


def make_loss_fns(
    model_apply: Callable,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    loss_type: str = "mse",
    noise_scale: float = 0.1,  # Required for t_regression
    student_df: float = 4.0,  # Required for t_regression
):
    """Create loss functions for both full data and minibatch

    Args:
        model_apply: Haiku model's apply function
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
            pred = model_apply(params, None, X)
            return jnp.mean((pred - Y) ** 2)

        def minibatch(params, Xb, Yb):
            pred = model_apply(params, None, Xb)
            return jnp.mean((pred - Yb) ** 2)

    elif loss_type == "t_regression":
        s2 = noise_scale**2
        nu = student_df

        def neglogt(resid):
            return 0.5 * (nu + 1) * jnp.log1p((resid**2) / (nu * s2))

        def full(params):
            pred = model_apply(params, None, X)
            return jnp.mean(neglogt(pred - Y))

        def minibatch(params, Xb, Yb):
            pred = model_apply(params, None, Xb)
            return jnp.mean(neglogt(pred - Yb))
    else:
        raise ValueError(f"Unknown loss: {loss_type}")

    return full, minibatch

