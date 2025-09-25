# llc/targets.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any
import jax
import jax.numpy as jnp

from .config import Config
from .data import make_dataset
from .models import infer_widths, build_mlp_forward_fn, count_params
from .losses import make_loss_fns, as_dtype
from .training import train_erm


@dataclass
class TargetBundle:
    """Simplified TargetBundle - single precision version (memory bloat removed)"""

    d: int
    params0: Dict[str, Any]  # Haiku params (single precision)
    # loss(params) -> scalar
    loss_full: Callable[[Dict[str, Any]], jnp.ndarray]
    loss_minibatch: Callable[[Dict[str, Any], jnp.ndarray, jnp.ndarray], jnp.ndarray]
    # data for minibatching (single precision)
    X: jnp.ndarray
    Y: jnp.ndarray
    L0: float  # L_n at params0
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
            skip=m_cfg.skip_connections,
            residual_period=m_cfg.residual_period,
            layernorm=m_cfg.layernorm,
        )

        # Initialize parameters
        key, subkey = jax.random.split(key)
        params_init = model.init(subkey, X[:1])  # Use first data point for init

        # Data will be cast to required precision dynamically in sample.py

        # Determine loss parameters explicitly (required for make_loss_fns)
        loss_type = cfg.posterior.loss
        noise_scale = cfg.data.noise_scale
        student_df = cfg.data.student_df

        # Train in f64 for precision, then cast to f32 for storage efficiency
        X_f64, Y_f64 = as_dtype(X, "float64"), as_dtype(Y, "float64")
        loss_full_f64, _ = make_loss_fns(
            model.apply,
            X_f64,
            Y_f64,
            loss_type=loss_type,
            noise_scale=noise_scale,
            student_df=student_df,
        )

        # Train to ERM (θ⋆) in f64 precision
        params_star_f64, metrics = train_erm(loss_full_f64, params_init, cfg, key)

        # Store in f32 for memory efficiency (precision determined dynamically in sample.py)
        X_f32, Y_f32 = as_dtype(X, "float32"), as_dtype(Y, "float32")
        params_star_f32 = jax.tree.map(lambda a: a.astype(jnp.float32), params_star_f64)

        # Create f32 loss functions for the bundle
        loss_full_f32, loss_minibatch_f32 = make_loss_fns(
            model.apply,
            X_f32,
            Y_f32,
            loss_type=loss_type,
            noise_scale=noise_scale,
            student_df=student_df,
        )

        L0 = float(loss_full_f64(params_star_f64))
        d = count_params(params_star_f64)

        return TargetBundle(
            d=d,
            params0=params_star_f32,
            loss_full=loss_full_f32,
            loss_minibatch=loss_minibatch_f32,
            X=X_f32,
            Y=Y_f32,
            L0=L0,
            model=model,
        )

    elif cfg.target == "quadratic":
        # ----- Analytic diagnostic: L_n(θ) = 0.5 ||θ||^2 -----
        # For quadratic target, we'll use a dummy Haiku model structure
        d = int(cfg.quad_dim or m_cfg.target_params or m_cfg.in_dim)

        # Create dummy params structure (single layer with d parameters) - f32 for efficiency
        params0 = {"quadratic": {"w": jnp.zeros((d,), dtype=jnp.float32)}}

        # loss_full(params) = 0.5 ||params||^2
        def _lf(params):
            # Flatten all params and compute quadratic loss
            leaves = jax.tree_util.tree_leaves(params)
            theta = jnp.concatenate([leaf.flatten() for leaf in leaves])
            return 0.5 * jnp.sum(theta * theta)

        def _lb(params, Xb, Yb):  # Keep (params, Xb, Yb) signature
            return _lf(params)

        # Provide trivial data so SGLD minibatching works - f32 for efficiency
        n = int(cfg.data.n_data)
        X = jnp.zeros((n, 1), dtype=jnp.float32)
        Y = jnp.zeros((n, 1), dtype=jnp.float32)

        L0 = 0.0  # L_n at params0=0

        # Dummy model (not used for quadratic target)
        model = None

        return TargetBundle(
            d=d,
            params0=params0,
            loss_full=_lf,
            loss_minibatch=_lb,
            X=X,
            Y=Y,
            L0=L0,
            model=model,
        )
    else:
        raise ValueError(f"Unknown target: {cfg.target}")
