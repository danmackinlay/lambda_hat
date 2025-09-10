# llc/data.py
"""Data generation utilities for teacher-student experiments"""

from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from jax import random

if TYPE_CHECKING:
    from .config import Config
else:
    # Runtime import to avoid circular dependency
    Config = "Config"

from .models import infer_widths, init_mlp_params, mlp_forward


def sample_X(key, cfg: Config, n: int, in_dim: int):
    """Sample inputs according to various distributions"""
    if cfg.x_dist == "gauss_iso":
        return random.normal(key, (n, in_dim))
    elif cfg.x_dist == "gauss_aniso":
        vals = jnp.array([cfg.cov_decay**i for i in range(in_dim)])
        A = jnp.diag(jnp.sqrt(vals))
        Z = random.normal(key, (n, in_dim))
        return Z @ A.T
    elif cfg.x_dist == "mixture":
        keys = random.split(key, 2)
        centers = random.normal(keys[0], (cfg.mixture_k, in_dim))
        centers = (
            cfg.mixture_spread
            * centers
            / (1e-6 + jnp.linalg.norm(centers, axis=1, keepdims=True))
        )
        comp = random.randint(keys[1], (n,), 0, cfg.mixture_k)
        eps = random.normal(key, (n, in_dim))
        return centers[comp] + eps
    elif cfg.x_dist == "lowdim_manifold":
        kz, ka = random.split(key)
        Z = random.normal(kz, (n, cfg.x_dim_latent))
        A = random.normal(ka, (cfg.x_dim_latent, in_dim))
        X_low = jnp.tanh(Z @ A)  # non-linear embedding
        return X_low
    elif cfg.x_dist == "heavy_tail":
        # Student-t via scaled normal / sqrt(gamma)
        k1, k2 = random.split(key)
        g = random.gamma(k1, cfg.student_df / 2, (n, 1)) / (cfg.student_df / 2)
        Z = random.normal(k2, (n, in_dim)) / jnp.sqrt(g)
        return Z
    else:
        raise ValueError(f"Unknown x_dist: {cfg.x_dist}")


def build_teacher(key, cfg: Config):
    """Build teacher network (can differ from student)"""
    t_depth = cfg.teacher_depth or cfg.depth
    t_widths = cfg.teacher_widths
    if t_widths is None:
        t_widths = infer_widths(
            cfg.in_dim, cfg.out_dim, t_depth, cfg.target_params, cfg.hidden
        )
    t_act = cfg.teacher_activation or cfg.activation

    params = init_mlp_params(
        key, cfg.in_dim, t_widths, cfg.out_dim, t_act, bias=True, init=cfg.init
    )

    def forward(X):
        Y = mlp_forward(
            params,
            X,
            t_act,
            skip=False,
            residual_period=cfg.residual_period,
            layernorm=False,
        )
        return Y

    return params, forward


def add_noise(key, y_clean, cfg: Config, X):
    """Add noise according to various models"""
    if cfg.noise_model == "gauss":
        return y_clean + cfg.noise_scale * random.normal(key, y_clean.shape)
    elif cfg.noise_model == "hetero":
        scale = cfg.noise_scale * (
            1.0 + cfg.hetero_scale * jnp.linalg.norm(X, axis=1, keepdims=True)
        )
        return y_clean + scale * random.normal(key, y_clean.shape)
    elif cfg.noise_model == "student_t":
        # Draw t noise via normal / sqrt(gamma)
        k1, k2 = random.split(key)
        g = random.gamma(k1, cfg.student_df / 2, y_clean.shape) / (cfg.student_df / 2)
        return y_clean + cfg.noise_scale * random.normal(k2, y_clean.shape) / jnp.sqrt(
            g
        )
    elif cfg.noise_model == "outliers":
        k1, k2, k3 = random.split(key, 3)
        base = cfg.noise_scale * random.normal(k1, y_clean.shape)
        mask = random.uniform(k2, y_clean.shape) < cfg.outlier_frac
        outl = cfg.outlier_scale * random.normal(k3, y_clean.shape)
        return y_clean + jnp.where(mask, outl, base)
    else:
        raise ValueError(f"Unknown noise_model: {cfg.noise_model}")


def make_dataset(key, cfg: Config):
    """End-to-end data generation"""
    kx, kt, kn = random.split(key, 3)
    X = sample_X(kx, cfg, cfg.n_data, cfg.in_dim)
    teacher_params, teacher_forward = build_teacher(kt, cfg)
    y_clean = teacher_forward(X)

    if cfg.teacher_dropout_rate > 0.0:
        kd = random.split(kt, 1)[0]
        mask = (random.uniform(kd, y_clean.shape) > cfg.teacher_dropout_rate).astype(
            y_clean.dtype
        )
        y_clean = y_clean * mask

    Y = add_noise(kn, y_clean, cfg, X)
    return X, Y, teacher_params, teacher_forward