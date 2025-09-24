# llc/losses.py
"""Loss function utilities adapted for Haiku models"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
import jax.numpy as jnp

if TYPE_CHECKING:
    from .config import Config


def as_dtype(x, dtype_str):  # 'float32' or 'float64'
    """Convert array to specified dtype"""
    return x.astype(jnp.float32 if dtype_str == "float32" else jnp.float64)


def make_loss_fns(
    model_apply: Callable,
    cfg: Config,
    X: jnp.ndarray,
    Y: jnp.ndarray
):
    """Create loss functions for both full data and minibatch

    Args:
        model_apply: Haiku model's apply function
        cfg: Configuration object
        X: Input data
        Y: Target data

    Returns:
        Tuple of (full_loss_fn, minibatch_loss_fn)
    """
    if cfg.loss == "mse":

        def full(params):
            pred = model_apply(params, None, X)
            return jnp.mean((pred - Y) ** 2)

        def minibatch(params, Xb, Yb):
            pred = model_apply(params, None, Xb)
            return jnp.mean((pred - Yb) ** 2)

    elif cfg.loss == "t_regression":
        s2 = cfg.noise_scale**2
        nu = cfg.student_df

        def neglogt(resid):
            return 0.5 * (nu + 1) * jnp.log1p((resid**2) / (nu * s2))

        def full(params):
            pred = model_apply(params, None, X)
            return jnp.mean(neglogt(pred - Y))

        def minibatch(params, Xb, Yb):
            pred = model_apply(params, None, Xb)
            return jnp.mean(neglogt(pred - Yb))
    else:
        raise ValueError(f"Unknown loss: {cfg.loss}")

    return full, minibatch