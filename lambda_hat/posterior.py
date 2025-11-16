# lambda_hat/posterior.py
"""Posterior construction utilities for tempered local posteriors.

NEW DESIGN: All samplers work in FLAT parameter space only.
Samplers never see model pytrees - they only see R^D vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .equinox_adapter import VectorisedModel

if TYPE_CHECKING:
    from .config import PosteriorConfig

Array = jnp.ndarray


def compute_beta_gamma(cfg: PosteriorConfig, d: int, n_data: int) -> tuple[float, float]:
    """Compute beta and gamma from config and dimension"""
    n_eff = max(3, int(n_data))
    beta = cfg.beta0 / jnp.log(n_eff) if cfg.beta_mode == "1_over_log_n" else cfg.beta0
    if cfg.prior_radius is not None and cfg.prior_radius > 0:
        gamma = d / (cfg.prior_radius**2)
    else:
        gamma = cfg.gamma
    return float(beta), float(gamma)


@dataclass(frozen=True)
class Posterior:
    """Posterior in flat parameter space.

    All functions operate on flat R^D vectors, never on model pytrees.
    This is the ONLY interface samplers should use.
    """

    vm: VectorisedModel
    logpost_flat: Callable[[Array], Array]  # R^D -> scalar log posterior
    grad_logpost_flat: Callable[[Array], Array]  # R^D -> R^D gradient
    flat0: Array  # initial flat parameters


def make_posterior(
    vm: VectorisedModel,
    flat0: Array,
    loss_full: Callable[[Any], Array],  # model -> scalar loss
    n_data: int,
    beta: float,
    gamma: float,
) -> Posterior:
    """Create posterior in flat space.

    Args:
        vm: VectorisedModel for converting flat <-> model
        flat0: Initial flat parameters (for prior center)
        loss_full: Full-data loss function (takes model, returns scalar)
        n_data: Number of data points
        beta: Temperature parameter
        gamma: Prior strength parameter

    Returns:
        Posterior with flat-space log probability and gradient functions
    """
    # Convert scalars to arrays with correct dtype
    beta_arr = jnp.asarray(beta, dtype=vm.dtype)
    gamma_arr = jnp.asarray(gamma, dtype=vm.dtype)
    n_arr = jnp.asarray(n_data, dtype=vm.dtype)
    theta0 = flat0  # Prior center in flat space

    def _logpost_from_flat(flat: Array) -> Array:
        """Log posterior: log P(w) = -n*beta*L_n(w) - gamma/2*||w-w0||^2"""
        model = vm.to_model(flat)
        Ln = loss_full(model)

        # Prior term: -gamma/2 * ||w-w0||^2
        prior = -0.5 * gamma_arr * jnp.sum((flat - theta0) ** 2)

        # Likelihood term: -n * beta * L_n(w)
        return prior - n_arr * beta_arr * Ln

    # Compute gradient using eqx.filter_grad (handles static leaves correctly)
    @jax.jit
    def _grad(flat: Array) -> Array:
        return eqx.filter_grad(_logpost_from_flat)(flat)

    @jax.jit
    def _lp(flat: Array) -> Array:
        return _logpost_from_flat(flat)

    return Posterior(vm=vm, logpost_flat=_lp, grad_logpost_flat=_grad, flat0=flat0)


def make_grad_loss_minibatch_flat(
    vm: VectorisedModel, loss_minibatch: Callable[[Any, Array, Array], Array]
) -> Callable[[Array, Tuple[Array, Array]], Array]:
    """Create gradient of minibatch loss in flat space.

    Args:
        vm: VectorisedModel
        loss_minibatch: Minibatch loss function (model, Xb, Yb) -> scalar

    Returns:
        Function (flat, (Xb, Yb)) -> flat_gradient
    """

    @jax.jit
    def _grad_fn(flat: Array, minibatch: Tuple[Array, Array]) -> Array:
        Xb, Yb = minibatch

        def _loss_fn(f):
            m = vm.to_model(f)
            return loss_minibatch(m, Xb, Yb)

        return eqx.filter_grad(_loss_fn)(flat)

    return _grad_fn
