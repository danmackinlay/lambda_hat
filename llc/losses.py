# llc/losses.py
"""Loss function utilities"""

from typing import TYPE_CHECKING
import jax.numpy as jnp

if TYPE_CHECKING:
    from .config import Config
else:
    # Runtime import to avoid circular dependency
    Config = "Config"

from .models import mlp_forward


def as_dtype(x, dtype_str):  # 'float32' or 'float64'
    """Convert array to specified dtype"""
    return x.astype(jnp.float32 if dtype_str == "float32" else jnp.float64)


def make_loss_fns(unravel, cfg: Config, X, Y):
    """Create loss functions for both full data and minibatch"""
    if cfg.loss == "mse":

        def full(theta):
            params = unravel(theta)
            pred = mlp_forward(
                params,
                X,
                cfg.activation,
                cfg.skip_connections,
                cfg.residual_period,
                cfg.layernorm,
            )
            return jnp.mean((pred - Y) ** 2)

        def minibatch(theta, Xb, Yb):
            params = unravel(theta)
            pred = mlp_forward(
                params,
                Xb,
                cfg.activation,
                cfg.skip_connections,
                cfg.residual_period,
                cfg.layernorm,
            )
            return jnp.mean((pred - Yb) ** 2)

    elif cfg.loss == "t_regression":
        s2 = cfg.noise_scale**2
        nu = cfg.student_df

        def neglogt(resid):
            return 0.5 * (nu + 1) * jnp.log1p((resid**2) / (nu * s2))

        def full(theta):
            params = unravel(theta)
            pred = mlp_forward(
                params,
                X,
                cfg.activation,
                cfg.skip_connections,
                cfg.residual_period,
                cfg.layernorm,
            )
            return jnp.mean(neglogt(pred - Y))

        def minibatch(theta, Xb, Yb):
            params = unravel(theta)
            pred = mlp_forward(
                params,
                Xb,
                cfg.activation,
                cfg.skip_connections,
                cfg.residual_period,
                cfg.layernorm,
            )
            return jnp.mean(neglogt(pred - Yb))
    else:
        raise ValueError(f"Unknown loss: {cfg.loss}")

    return full, minibatch
