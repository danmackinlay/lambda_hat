# llc/samplers/adapters.py
"""
Batched adapters only. Single-chain adapters were removed. Each adapter times its own
warmup/tuning and sets `BatchedResult.warmup_time_seconds`/`warmup_grads`. No per-step
Python hooks; all diagnostics flow via `extras` and `L_hist`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import blackjax

from .base import (
    SamplerSpec,
    precond_update,
)

Array = jnp.ndarray

# ---------- SGLD (BlackJAX 1.2.5 returns new_position only) ----------


def sgld_spec(
    *,
    grad_logpost_minibatch,
    X,
    Y,
    n_data,
    step_size: float,
    batch_size: int,
    precond_mode: str = "none",
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bias_correction: bool = True,
) -> SamplerSpec:
    """Create SamplerSpec for SGLD with optional preconditioning."""
    precond = (precond_mode or "none").lower()

    if precond == "none":
        # Plain SGLD
        sgld = blackjax.sgld(grad_logpost_minibatch)
        step_single = jax.jit(sgld.step)

        def step_one(k, theta):
            k_noise, k_batch = jax.random.split(k)
            idx = jax.random.randint(k_batch, (batch_size,), 0, n_data)
            new_theta = step_single(k_noise, theta, (X[idx], Y[idx]), step_size)
            return new_theta.astype(theta.dtype), None

        step_vmapped = jax.jit(jax.vmap(step_one, in_axes=(0, 0)))
        position_fn = lambda s: s

    else:
        # Preconditioned SGLD using generic precond_update
        @jax.jit
        def precond_step_single(key, state):
            theta, precond_state = state
            k_noise, k_batch = jax.random.split(key)
            idx = jax.random.randint(k_batch, (batch_size,), 0, n_data)
            g = grad_logpost_minibatch(theta, (X[idx], Y[idx]))

            # Use generic preconditioning update
            inv_sqrt, new_precond_state, drift_m = precond_update(
                g, precond_state, precond, beta1, beta2, eps, bias_correction
            )
            drift = 0.5 * step_size * (drift_m * inv_sqrt)

            noise = (
                jax.random.normal(k_noise, theta.shape) * jnp.sqrt(step_size) * inv_sqrt
            )
            theta_new = theta + drift + noise
            return (
                theta_new.astype(theta.dtype),
                new_precond_state,
            ), None

        step_vmapped = jax.jit(jax.vmap(precond_step_single, in_axes=(0, 0)))
        position_fn = lambda s: s[0]  # extract theta from (theta, DiagPrecondState)

    return SamplerSpec(
        name="sgld",
        step_vmapped=step_vmapped,
        position_fn=position_fn,
        info_extractors={},
        grads_per_step=1.0,
    )


# ---------- SGHMC (Stochastic Gradient Hamiltonian Monte Carlo) ----------


def sghmc_spec(
    *,
    grad_logpost_minibatch,
    X,
    Y,
    n_data,
    step_size: float,
    temperature: float = 1.0,
    batch_size: int,
) -> SamplerSpec:
    """Create SamplerSpec for SGHMC."""
    sghmc = blackjax.sghmc(grad_logpost_minibatch)
    step_single = jax.jit(sghmc.step)
    step_vmapped = jax.jit(jax.vmap(step_single, in_axes=(0, 0, 0, None, None)))

    def step_fn_vmapped(keys_t, thetas_t):
        # Sample per-chain minibatch indices
        idx = jax.vmap(lambda k: jax.random.randint(k, (batch_size,), 0, n_data))(
            keys_t
        )
        Xb = X[idx]  # (C, B, ...)
        Yb = Y[idx]  # (C, B, ...)
        thetas_new = step_vmapped(keys_t, thetas_t, (Xb, Yb), step_size, temperature)
        # Keep dtype stable for scan carry
        thetas_new = thetas_new.astype(thetas_t.dtype)
        return thetas_new, None

    return SamplerSpec(
        name="sghmc",
        step_vmapped=step_fn_vmapped,
        position_fn=lambda s: s,
        info_extractors={},
        grads_per_step=1.0,
    )


# ---------- HMC (with window adaptation) ----------


def hmc_spec(
    *,
    logpost_and_grad,
    L: int,
) -> SamplerSpec:
    """Create SamplerSpec for HMC. Adaptation handled separately."""

    def logdensity(theta):
        val, _ = logpost_and_grad(theta)
        return val

    # Note: This assumes adaptation has already been done and params are available
    # The actual step function will be built with tuned parameters at runtime
    def make_step_vmapped(step_size, inverse_mass_matrix):
        step_kernel = blackjax.hmc(
            logdensity,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            num_integration_steps=L,
        ).step
        return jax.jit(jax.vmap(jax.jit(step_kernel), in_axes=(0, 0)))

    # For SamplerSpec, we need a concrete step function, but HMC needs adaptation first
    # This is a placeholder that will be replaced after adaptation
    placeholder_step = lambda keys, states: (states, None)

    info_extractors = {
        "accept": lambda info: getattr(info, "acceptance_rate", jnp.zeros(())),
        "energy": lambda info: getattr(info, "energy", jnp.zeros(())),
    }

    return SamplerSpec(
        name="hmc",
        step_vmapped=placeholder_step,  # Will be replaced after adaptation
        position_fn=lambda st: st.position,
        info_extractors=info_extractors,
        grads_per_step=L + 1,  # Velocity-Verlet integration
    )


# ---------- MCLMC (unadjusted; tuned L & step_size) ----------


def mclmc_spec(
    *,
    logdensity_fn,
    integrator_name="isokinetic_mclachlan",
) -> SamplerSpec:
    """Create SamplerSpec for MCLMC. Tuning handled separately."""
    integrator = getattr(blackjax.mcmc.integrators, integrator_name)

    # For SamplerSpec, we need a concrete step function, but MCLMC needs tuning first
    # This is a placeholder that will be replaced after tuning
    placeholder_step = lambda keys, states: (states, None)

    info_extractors = {
        "energy": lambda info: getattr(info, "energy_change", jnp.zeros(()))
    }

    return SamplerSpec(
        name="mclmc",
        step_vmapped=placeholder_step,  # Will be replaced after tuning
        position_fn=lambda st: st.position,
        info_extractors=info_extractors,
        grads_per_step=1.0,  # MCLMC does one gradient-like step per iteration
    )
