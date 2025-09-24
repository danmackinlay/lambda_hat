# llc/posterior.py
"""Posterior construction utilities for tempered local posteriors"""

from __future__ import annotations

from typing import TYPE_CHECKING
import jax.numpy as jnp
from jax import grad, value_and_grad, jit

if TYPE_CHECKING:
    from .config import Config


def compute_beta_gamma(cfg: Config, d: int) -> tuple[float, float]:
    """Compute beta and gamma from config and dimension"""
    n_eff = max(3, int(cfg.n_data))
    beta = cfg.beta0 / jnp.log(n_eff) if cfg.beta_mode == "1_over_log_n" else cfg.beta0
    gamma = (d / (cfg.prior_radius**2)) if (cfg.prior_radius is not None) else cfg.gamma
    return float(beta), float(gamma)


def make_logpost_and_score(loss_full, loss_minibatch, params0, n, beta, gamma):
    """Create log posterior using flexible loss functions for Haiku params"""
    # Flatten params for prior computation
    leaves, tree_def = jax.tree_util.tree_flatten(params0)
    theta0 = jnp.concatenate([leaf.flatten() for leaf in leaves])

    # Make sure scalars match theta dtype so we don't upcast f32 paths to f64
    beta = jnp.asarray(beta, dtype=theta0.dtype)
    gamma = jnp.asarray(gamma, dtype=theta0.dtype)
    n = jnp.asarray(n, dtype=theta0.dtype)

    def logpost(params):
        # Flatten params for prior computation
        leaves, _ = jax.tree_util.tree_flatten(params)
        theta = jnp.concatenate([leaf.flatten() for leaf in leaves])

        Ln = loss_full(params)
        lp = -0.5 * gamma * jnp.sum((theta - theta0) ** 2)
        return lp - n * beta * Ln

    logpost_and_grad = value_and_grad(logpost)

    @jit
    def grad_logpost_minibatch(params, minibatch):  # <- accepts (Xb, Yb)
        Xb, Yb = minibatch

        # Flatten params for prior computation
        leaves, _ = jax.tree_util.tree_flatten(params)
        theta = jnp.concatenate([leaf.flatten() for leaf in leaves])

        # Compute gradient of loss w.r.t. params
        g_Lb = grad(lambda p: loss_minibatch(p, Xb, Yb))(params)

        # Compute prior gradient w.r.t. flattened params
        g_prior_flat = -gamma * (theta - theta0)

        # Reshape prior gradient back to params structure
        shapes = [leaf.shape for leaf in leaves]
        sizes = [leaf.size for leaf in leaves]
        g_prior_leaves = []
        start = 0
        for shape, size in zip(shapes, sizes):
            g_prior_leaves.append(g_prior_flat[start:start+size].reshape(shape))
            start += size
        g_prior = jax.tree_util.tree_unflatten(tree_def, g_prior_leaves)

        # Combine prior and likelihood gradients
        return jax.tree_map(
            lambda gp, gl: gp - beta * n * gl,
            g_prior, g_Lb
        )

    return logpost_and_grad, grad_logpost_minibatch


def make_logdensity_for_mclmc(loss_full64, params0_f64, n, beta, gamma):
    """Create log-density function for MCLMC sampler (f64 precision)

    MCLMC expects a pure log-density function, not a gradient closure.
    This extracts the log posterior: log π(θ) = -nβ L_n(θ) - γ/2 ||θ-θ₀||²
    """
    # Flatten params for prior computation
    leaves, _ = jax.tree_util.tree_flatten(params0_f64)
    theta0_f64 = jnp.concatenate([leaf.flatten() for leaf in leaves])

    # Ensure scalars match theta dtype for consistency
    beta = jnp.asarray(beta, dtype=theta0_f64.dtype)
    gamma = jnp.asarray(gamma, dtype=theta0_f64.dtype)
    n = jnp.asarray(n, dtype=theta0_f64.dtype)

    @jit
    def logdensity(params64):
        # Flatten params for prior computation
        leaves, _ = jax.tree_util.tree_flatten(params64)
        theta64 = jnp.concatenate([leaf.flatten() for leaf in leaves])

        Ln = loss_full64(params64)  # already f64 X,Y inside loss_full64
        lp = -0.5 * gamma * jnp.sum((theta64 - theta0_f64) ** 2)
        return lp - n * beta * Ln

    return logdensity