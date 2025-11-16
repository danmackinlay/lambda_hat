# lambda_hat/data.py
"""Data generation utilities for teacher-student experiments"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import random

if TYPE_CHECKING:
    from .config import Config

from lambda_hat.config import validate_teacher_cfg

from .nn_eqx import build_mlp, infer_widths


def sample_X(key, cfg: "Config", n: int, in_dim: int):
    """Sample inputs according to various distributions"""
    data_cfg = cfg.data if hasattr(cfg, "data") else cfg

    if data_cfg.x_dist == "gauss_iso":
        return random.normal(key, (n, in_dim))
    elif data_cfg.x_dist == "gauss_aniso":
        vals = jnp.array([data_cfg.cov_decay**i for i in range(in_dim)])
        A = jnp.diag(jnp.sqrt(vals))
        Z = random.normal(key, (n, in_dim))
        return Z @ A.T
    elif data_cfg.x_dist == "mixture":
        k1, k2, k3 = random.split(key, 3)
        centers = random.normal(k1, (data_cfg.mixture_k, in_dim))
        centers = (
            data_cfg.mixture_spread
            * centers
            / (1e-6 + jnp.linalg.norm(centers, axis=1, keepdims=True))
        )
        comp = random.randint(k2, (n,), 0, data_cfg.mixture_k)
        eps = random.normal(k3, (n, in_dim))
        return centers[comp] + eps
    elif data_cfg.x_dist == "lowdim_manifold":
        kz, ka = random.split(key)
        Z = random.normal(kz, (n, data_cfg.x_dim_latent))
        A = random.normal(ka, (data_cfg.x_dim_latent, in_dim))
        X_low = jnp.tanh(Z @ A)  # non-linear embedding
        return X_low
    elif data_cfg.x_dist == "heavy_tail":
        # Student-t via scaled normal / sqrt(gamma)
        k1, k2 = random.split(key)
        g = random.gamma(k1, data_cfg.student_df / 2, (n, 1)) / (data_cfg.student_df / 2)
        Z = random.normal(k2, (n, in_dim)) / jnp.sqrt(g)
        return Z
    else:
        raise ValueError(f"Unknown x_dist: {data_cfg.x_dist}")


def build_teacher(key, cfg: Config):
    """Build teacher network (can differ from student)"""
    m_cfg = cfg.model

    # Student dims are the truth for I/O
    t_in = m_cfg.in_dim
    t_out = m_cfg.out_dim

    # Use teacher config if present, otherwise fall back to model config
    if hasattr(cfg, "teacher") and cfg.teacher is not None and cfg.teacher != {}:
        t_cfg = cfg.teacher
        validate_teacher_cfg(dict(t_cfg))

        if t_cfg.widths is not None:
            t_widths = t_cfg.widths
        else:
            # one size driver, or both None -> fallback to model.hidden
            t_TP = t_cfg.target_params if t_cfg.target_params is not None else m_cfg.target_params
            t_hid = t_cfg.hidden if t_cfg.hidden is not None else m_cfg.hidden
            t_widths = infer_widths(t_in, t_out, t_cfg.depth, t_TP, fallback_width=t_hid)
        t_act = t_cfg.activation or m_cfg.activation
    else:
        # No teacher config - use model config directly
        t_widths = m_cfg.widths
        if t_widths is None:
            t_widths = infer_widths(t_in, t_out, m_cfg.depth, m_cfg.target_params, m_cfg.hidden)
        t_act = m_cfg.activation

    # Build Equinox model
    model = build_mlp(
        in_dim=t_in,
        widths=t_widths,
        out_dim=t_out,
        activation=t_act,
        bias=True,
        layernorm=False,
        key=key,
    )

    # For Equinox, model IS the params
    params = model

    def forward(X):
        # Equinox models are called directly
        return model(X)

    return params, forward


def add_noise(key, y_clean, cfg: Config, X):
    """Add noise according to various models"""
    data_cfg = cfg.data

    if data_cfg.noise_model == "gauss":
        return y_clean + data_cfg.noise_scale * random.normal(key, y_clean.shape)
    elif data_cfg.noise_model == "hetero":
        scale = data_cfg.noise_scale * (
            1.0 + data_cfg.hetero_scale * jnp.linalg.norm(X, axis=1, keepdims=True)
        )
        return y_clean + scale * random.normal(key, y_clean.shape)
    elif data_cfg.noise_model == "student_t":
        # Draw t noise via normal / sqrt(gamma)
        k1, k2 = random.split(key)
        g = random.gamma(k1, data_cfg.student_df / 2, y_clean.shape) / (data_cfg.student_df / 2)
        return y_clean + data_cfg.noise_scale * random.normal(k2, y_clean.shape) / jnp.sqrt(g)
    elif data_cfg.noise_model == "outliers":
        k1, k2, k3 = random.split(key, 3)
        base = data_cfg.noise_scale * random.normal(k1, y_clean.shape)
        mask = random.uniform(k2, y_clean.shape) < data_cfg.outlier_frac
        outl = data_cfg.outlier_scale * random.normal(k3, y_clean.shape)
        return y_clean + jnp.where(mask, outl, base)
    else:
        raise ValueError(f"Unknown noise_model: {data_cfg.noise_model}")


def make_dataset(key, cfg: Config):
    """End-to-end data generation"""
    kx, kt, kn = random.split(key, 3)
    # Fix the line causing the error:
    X = sample_X(kx, cfg, cfg.data.n_data, cfg.model.in_dim)
    teacher_params, teacher_forward = build_teacher(kt, cfg)
    y_clean = teacher_forward(X)

    # Apply dropout if teacher config exists and specifies it
    dropout_rate = 0.0
    if hasattr(cfg, "teacher") and cfg.teacher is not None and hasattr(cfg.teacher, "dropout_rate"):
        dropout_rate = cfg.teacher.dropout_rate

    if dropout_rate > 0.0:
        kd = random.split(kt, 1)[0]
        mask = (random.uniform(kd, y_clean.shape) > dropout_rate).astype(y_clean.dtype)
        y_clean = y_clean * mask

    Y = add_noise(kn, y_clean, cfg, X)
    return X, Y, teacher_params, teacher_forward
