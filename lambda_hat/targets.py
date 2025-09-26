# llc/targets.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any
import jax
import jax.numpy as jnp

from .config import Config
from .data import make_dataset
from .models import infer_widths, build_mlp_forward_fn, count_params
from .losses import make_loss_fns
from .training import train_erm


def make_loss_full(loss_fn: Callable) -> Callable:
    """Create a loss function that returns scalar per chain for auxiliary recording.

    Args:
        loss_fn: Function that takes position and returns scalar loss

    Returns:
        Callable that accepts batched position (C, ...) or single (...) and returns (C,) or ()
    """

    @jax.jit
    def loss_from_position(position):
        # Returns average negative log-likelihood or your Ln definition
        Ln = loss_fn(position)  # scalar per chain
        return Ln

    return loss_from_position


@dataclass
class TargetBundle:
    d: int
    params0_f32: Dict[str, Any]  # Haiku params
    params0_f64: Dict[str, Any]  # Haiku params
    # loss(params) -> scalar
    loss_full_f32: Callable[[Dict[str, Any]], jnp.ndarray]
    loss_minibatch_f32: Callable[
        [Dict[str, Any], jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]
    loss_full_f64: Callable[[Dict[str, Any]], jnp.ndarray]
    loss_minibatch_f64: Callable[
        [Dict[str, Any], jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]
    # data for minibatching (stored once in f64)
    X: jnp.ndarray
    Y: jnp.ndarray
    L0: float  # L_n at params0 (f64)
    # Haiku model for forward passes
    model: Any


def build_target(key, cfg: Config) -> TargetBundle:
    """Return a self-contained target for the pipeline to consume."""
    m_cfg = cfg.model

    if (cfg.target or "mlp") == "mlp":
        # ----- MLP path with Haiku -----
        # Generate data
        X, Y, teacher_params, teacher_forward = make_dataset(key, cfg)

        # Build Haiku model
        key, subkey = jax.random.split(key)
        widths = m_cfg.widths or infer_widths(
            m_cfg.in_dim,
            m_cfg.out_dim,
            m_cfg.depth,
            m_cfg.target_params,
            fallback_width=m_cfg.hidden,
        )

        model = build_mlp_forward_fn(
            in_dim=m_cfg.in_dim,
            widths=widths,
            out_dim=m_cfg.out_dim,
            activation=m_cfg.activation,
            bias=m_cfg.bias,
            init=m_cfg.init,
            layernorm=m_cfg.layernorm,
        )

        # Initialize parameters
        key, subkey = jax.random.split(key)
        params_init = model.init(subkey, X[:1])  # Use first data point for init

        # Cast data to both dtypes using native JAX operations
        X_f32, Y_f32 = X.astype(jnp.float32), Y.astype(jnp.float32)
        X_f64, Y_f64 = X.astype(jnp.float64), Y.astype(jnp.float64)

        # Create loss functions for f64 (for ERM training)
        loss_full_f64, loss_minibatch_f64 = make_loss_fns(
            model.apply, cfg, X_f64, Y_f64
        )

        # Train to ERM (θ⋆) in f64 precision
        params_star_f64, metrics = train_erm(loss_full_f64, params_init, cfg, key)

        # Convert to f32
        params_star_f32 = jax.tree.map(
            lambda a: a.astype(jnp.float32), params_star_f64
        )

        # Create loss functions for f32
        loss_full_f32, loss_minibatch_f32 = make_loss_fns(
            model.apply, cfg, X_f32, Y_f32
        )

        L0 = float(loss_full_f64(params_star_f64))
        d = count_params(params_star_f64)

        return TargetBundle(
            d=d,
            params0_f32=params_star_f32,
            params0_f64=params_star_f64,
            loss_full_f32=loss_full_f32,
            loss_minibatch_f32=loss_minibatch_f32,
            loss_full_f64=loss_full_f64,
            loss_minibatch_f64=loss_minibatch_f64,
            X=X_f64,
            Y=Y_f64,
            L0=L0,
            model=model,
        )

    elif cfg.target == "quadratic":
        # ----- Analytic diagnostic: L_n(θ) = 0.5 ||θ||^2 -----
        # For quadratic target, we'll use a dummy Haiku model structure
        d = int(cfg.quad_dim or m_cfg.target_params or m_cfg.in_dim)

        # Create dummy params structure (single layer with d parameters)
        params0_f64 = {"quadratic": {"w": jnp.zeros((d,), dtype=jnp.float64)}}
        params0_f32 = {"quadratic": {"w": jnp.zeros((d,), dtype=jnp.float32)}}

        # loss_full(params) = 0.5 ||params||^2
        def _lf(params):
            # Flatten all params and compute quadratic loss
            leaves = jax.tree_util.tree_leaves(params)
            theta = jnp.concatenate([leaf.flatten() for leaf in leaves])
            return 0.5 * jnp.sum(theta * theta)

        def _lb(params, Xb, Yb):  # Keep (params, Xb, Yb) signature
            return _lf(params)

        # Provide trivial data so SGLD minibatching works
        n = int(cfg.data.n_data)
        X_f32 = jnp.zeros((n, 1), dtype=jnp.float32)
        Y_f32 = jnp.zeros((n, 1), dtype=jnp.float32)
        X_f64 = jnp.zeros((n, 1), dtype=jnp.float64)
        Y_f64 = jnp.zeros((n, 1), dtype=jnp.float64)

        L0 = 0.0  # L_n at params0=0

        # Dummy model (not used for quadratic target)
        model = None

        return TargetBundle(
            d=d,
            params0_f32=params0_f32,
            params0_f64=params0_f64,
            loss_full_f32=lambda p: _lf(p).astype(jnp.float32),
            loss_minibatch_f32=lambda p, Xb, Yb: _lb(p, Xb, Yb).astype(jnp.float32),
            loss_full_f64=lambda p: _lf(p).astype(jnp.float64),
            loss_minibatch_f64=lambda p, Xb, Yb: _lb(p, Xb, Yb).astype(jnp.float64),
            X=X_f64,
            Y=Y_f64,
            L0=L0,
            model=model,
        )
    else:
        raise ValueError(f"Unknown target: {cfg.target}")
