# llc/targets.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .config import Config
from .data import make_dataset
from .models import infer_widths, init_mlp_params
from .losses import make_loss_fns, as_dtype
from .experiments import train_erm

@dataclass
class TargetBundle:
    d: int
    theta0_f32: jnp.ndarray
    theta0_f64: jnp.ndarray
    # loss(theta) -> scalar
    loss_full_f32: Callable[[jnp.ndarray], jnp.ndarray]
    loss_minibatch_f32: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    loss_full_f64: Callable[[jnp.ndarray], jnp.ndarray]
    loss_minibatch_f64: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    # data for minibatching (even if ignored by the target)
    X_f32: jnp.ndarray
    Y_f32: jnp.ndarray
    X_f64: jnp.ndarray
    Y_f64: jnp.ndarray
    L0: float  # L_n at theta0 (f64)

def _identity_unravel(theta: jnp.ndarray):
    # For analytic targets with a flat parameter vector
    return theta

def build_target(key, cfg: Config) -> TargetBundle:
    """Return a self-contained target for the pipeline to consume."""
    if (cfg.target or "mlp") == "mlp":
        # ----- Existing NN path -----
        # Data + teacher
        X, Y, teacher_params, teacher_forward = make_dataset(key, cfg)

        # Init student params & train to ERM (θ⋆), then center prior at θ⋆
        key, subkey = jax.random.split(key)
        widths = cfg.widths or infer_widths(
            cfg.in_dim, cfg.out_dim, cfg.depth, cfg.target_params, fallback_width=cfg.hidden
        )
        w0_pytree = init_mlp_params(
            subkey, cfg.in_dim, widths, cfg.out_dim, cfg.activation, cfg.bias, cfg.init
        )
        theta_star_f64, unravel_star_f64 = train_erm(
            w0_pytree, cfg, X.astype(jnp.float64), Y.astype(jnp.float64)
        )
        params_star_f64 = unravel_star_f64(theta_star_f64)
        params_star_f32 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float32), params_star_f64)
        theta0_f32, unravel_star_f32 = ravel_pytree(params_star_f32)

        # Cast data to both dtypes
        X_f32, Y_f32 = as_dtype(X, cfg.sgld_dtype), as_dtype(Y, cfg.sgld_dtype)
        X_f64, Y_f64 = as_dtype(X, cfg.hmc_dtype), as_dtype(Y, cfg.hmc_dtype)

        # Loss fns for each dtype
        loss_full_f32, loss_minibatch_f32 = make_loss_fns(unravel_star_f32, cfg, X_f32, Y_f32)
        loss_full_f64, loss_minibatch_f64 = make_loss_fns(unravel_star_f64, cfg, X_f64, Y_f64)

        L0 = float(loss_full_f64(theta_star_f64))
        d = int(theta0_f32.size)
        return TargetBundle(
            d=d,
            theta0_f32=theta0_f32,
            theta0_f64=theta_star_f64,
            loss_full_f32=loss_full_f32,
            loss_minibatch_f32=loss_minibatch_f32,
            loss_full_f64=loss_full_f64,
            loss_minibatch_f64=loss_minibatch_f64,
            X_f32=X_f32, Y_f32=Y_f32, X_f64=X_f64, Y_f64=Y_f64,
            L0=L0,
        )

    elif cfg.target == "quadratic":
        # ----- Analytic diagnostic: L_n(θ) = 0.5 ||θ||^2 -----
        d = int(cfg.quad_dim or cfg.target_params or cfg.in_dim)
        theta0_f64 = jnp.zeros((d,), dtype=jnp.float64)
        theta0_f32 = theta0_f64.astype(jnp.float32)

        # loss_full(θ) = 0.5 ||θ||^2 ; minibatch ignores Xb,Yb but keeps signature
        def _lf(theta):      return 0.5 * jnp.sum(theta * theta)
        def _lb(theta, Xb, Yb):  # <— keep (theta, Xb, Yb) to match posterior.py
            return _lf(theta)

        # Provide trivial data so SGLD minibatching works without special cases
        n = int(cfg.n_data)
        X_f32 = jnp.zeros((n, 1), dtype=jnp.float32)
        Y_f32 = jnp.zeros((n, 1), dtype=jnp.float32)
        X_f64 = jnp.zeros((n, 1), dtype=jnp.float64)
        Y_f64 = jnp.zeros((n, 1), dtype=jnp.float64)

        L0 = 0.0  # L_n at θ0=0
        return TargetBundle(
            d=d,
            theta0_f32=theta0_f32,
            theta0_f64=theta0_f64,
            loss_full_f32=lambda th: _lf(th.astype(jnp.float32)).astype(jnp.float32),
            loss_minibatch_f32=lambda th, Xb, Yb: _lb(th.astype(jnp.float32), Xb, Yb).astype(jnp.float32),
            loss_full_f64=lambda th: _lf(th.astype(jnp.float64)).astype(jnp.float64),
            loss_minibatch_f64=lambda th, Xb, Yb: _lb(th.astype(jnp.float64), Xb, Yb).astype(jnp.float64),
            X_f32=X_f32, Y_f32=Y_f32, X_f64=X_f64, Y_f64=Y_f64,
            L0=L0,
        )
    else:
        raise ValueError(f"Unknown target: {cfg.target}")