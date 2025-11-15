# lambda_hat/targets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from lambda_hat.config import validate_teacher_cfg

from .config import Config
from .data import make_dataset
from .losses import as_dtype, make_loss_fns
from .nn_eqx import build_mlp, count_params, infer_widths
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
    params0: Any  # Equinox model (ERM solution, single precision)
    # loss(params) -> scalar
    loss_full: Callable[[Any], jnp.ndarray]
    loss_minibatch: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    # data for minibatching (single precision)
    X: jnp.ndarray
    Y: jnp.ndarray
    L0: float  # L_n at params0
    # Model is now same as params0 for Equinox (kept for backward compat)
    model: Any
    # VI-specific fields (flattened params and unravel function)
    params0_flat: jnp.ndarray  # Flattened ERM solution
    unravel_fn: Callable[[jnp.ndarray], Any]  # flat -> pytree


def build_target(key, cfg: Config) -> tuple[TargetBundle, list[int], list[int] | None]:
    """Return a self-contained target for the pipeline to consume.

    Returns:
        TargetBundle: The target bundle for training
        list[int]: Resolved model widths
        list[int] | None: Resolved teacher widths (None if no teacher)
    """
    m_cfg = cfg.model
    #  mapping forms for cfg.target
    target_name = cfg.target.get("name", "mlp")

    if target_name == "mlp":
        # ----- MLP path with Haiku -----
        # Generate data
        X, Y, teacher_params, teacher_forward = make_dataset(key, cfg)

        # Build Haiku model - compute and store resolved widths
        key, subkey = jax.random.split(key)
        used_model_widths = m_cfg.widths or infer_widths(
            m_cfg.in_dim,
            m_cfg.out_dim,
            m_cfg.depth,
            m_cfg.target_params,
            fallback_width=m_cfg.hidden,
        )

        # Student dims are the truth for I/O
        in_dim = m_cfg.in_dim
        out_dim = m_cfg.out_dim

        used_teacher_widths = None
        if getattr(cfg, "teacher", None) and cfg.teacher != {}:
            t = cfg.teacher
            validate_teacher_cfg(dict(t))
            if t.widths is not None:
                used_teacher_widths = t.widths
            else:
                # one size driver, or both None -> fallback to model.hidden
                t_TP = t.target_params if t.target_params is not None else m_cfg.target_params
                t_hid = t.hidden if t.hidden is not None else m_cfg.hidden
                used_teacher_widths = infer_widths(
                    in_dim, out_dim, t.depth, t_TP, fallback_width=t_hid
                )
        else:
            used_teacher_widths = None  # no teacher

        # Build Equinox model (model IS the params)
        key, subkey = jax.random.split(key)
        model = build_mlp(
            in_dim=m_cfg.in_dim,
            widths=used_model_widths,
            out_dim=m_cfg.out_dim,
            activation=m_cfg.activation,
            bias=m_cfg.bias,
            layernorm=m_cfg.layernorm,
            key=subkey,
        )
        params_init = model  # For Equinox, model IS the params

        # Determine loss parameters explicitly (required for make_loss_fns)
        loss_type = cfg.posterior.loss
        noise_scale = cfg.data.noise_scale
        student_df = cfg.data.student_df

        # --- REVISED STRATEGY: Train and Store in F32 ---
        # Explicitly cast data and initial parameters to F32 to guarantee F32 training,
        # regardless of global JAX settings (e.g. jax_enable_x64).
        X_f32 = as_dtype(X, "float32")
        Y_f32 = as_dtype(Y, "float32")
        params_init_f32 = as_dtype(params_init, "float32")

        # Create F32 loss functions for training
        # Equinox models are called directly: model(x), not model.apply(params, None, x)
        predict_fn = lambda m, x: m(x)
        loss_full_f32, loss_minibatch_f32 = make_loss_fns(
            predict_fn,
            X_f32,
            Y_f32,
            loss_type=loss_type,
            noise_scale=noise_scale,
            student_df=student_df,
        )

        # Train to ERM (θ⋆) in F32 precision
        params_star_f32, metrics = train_erm(loss_full_f32, params_init_f32, cfg, key)

        # L0 is the final loss value from the F32 training
        # Use the metric if available (computed in train_erm), otherwise recompute.
        L0 = float(metrics.get("final_loss", loss_full_f32(params_star_f32)))
        d = count_params(params_star_f32)

        # Flatten params for VI
        params_star_flat, unravel_fn = jax.flatten_util.ravel_pytree(params_star_f32)

        return (
            TargetBundle(
                d=d,
                params0=params_star_f32,
                loss_full=loss_full_f32,
                loss_minibatch=loss_minibatch_f32,
                X=X_f32,
                Y=Y_f32,
                L0=L0,
                model=model,
                params0_flat=params_star_flat,
                unravel_fn=unravel_fn,
            ),
            used_model_widths,
            used_teacher_widths,
        )

    elif target_name == "quadratic":
        # ----- Analytic diagnostic: L_n(θ) = 0.5 ||θ||^2 -----
        # For quadratic target, we use a dummy PyTree params structure
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

        # Flatten params for VI
        params0_flat, unravel_fn = jax.flatten_util.ravel_pytree(params0)

        return (
            TargetBundle(
                d=d,
                params0=params0,
                loss_full=_lf,
                loss_minibatch=_lb,
                X=X,
                Y=Y,
                L0=L0,
                model=model,
                params0_flat=params0_flat,
                unravel_fn=unravel_fn,
            ),
            [],
            None,
        )  # No meaningful widths for quadratic target
    else:
        raise ValueError(f"Unknown target: {target_name}")
