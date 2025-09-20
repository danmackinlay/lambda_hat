# llc/samplers/adapters.py
"""
Batched adapters only. Single-chain adapters were removed. Each adapter times its own
warmup/tuning and sets `BatchedResult.warmup_time_seconds`/`warmup_grads`. No per-step
Python hooks; all diagnostics flow via `extras` and `L_hist`.
"""

from __future__ import annotations
import time

import jax
import jax.numpy as jnp
import blackjax

from .base import (
    drive_chains_batched,
    BatchedResult,
    SamplerSpec,
    DiagPrecondState,
    precond_update,
)

Array = jnp.ndarray

# ---------- SGLD (BlackJAX 1.2.5 returns new_position only) ----------


def run_sgld_chains_batched(
    *,
    rng_key,  # PRNGKey
    init_thetas,  # (C, d)
    grad_logpost_minibatch,  # (theta, (Xb,Yb)) -> grad
    X,
    Y,
    n_data,
    step_size: float,
    n_steps: int,
    warmup: int,
    batch_size: int,
    eval_every: int,
    thin: int,
    Ln_eval_f64,
    tiny_store_fn=None,
    # Optional preconditioning parameters
    precond_mode: str = "none",
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bias_correction: bool = True,
) -> BatchedResult:
    """
    Run multiple SGLD chains in parallel using vmap + lax.scan.

    This replaces the Python for-loop over chains with a single compiled
    program that steps all chains together, providing significant speedup.
    """
    precond = (precond_mode or "none").lower()

    if precond == "none":
        # Plain SGLD: vmapped BlackJAX step
        sgld = blackjax.sgld(grad_logpost_minibatch)
        step_single = jax.jit(sgld.step)

        def step_one(k, theta):
            k_noise, k_batch = jax.random.split(k)
            idx = jax.random.randint(k_batch, (batch_size,), 0, n_data)
            new_theta = step_single(k_noise, theta, (X[idx], Y[idx]), step_size)
            # Keep dtype stable to satisfy lax.scan's carry type check
            return new_theta.astype(theta.dtype), None

        step_vmapped = jax.jit(jax.vmap(step_one, in_axes=(0, 0)))
        init_state = init_thetas

        def position_fn(s):
            return s

    else:
        # Preconditioned SGLD: batched states (theta, DiagPrecondState)
        @jax.jit
        def precond_step_single(key, state):
            theta, precond_state = state
            k_noise, k_batch = jax.random.split(key)
            idx = jax.random.randint(k_batch, (batch_size,), 0, n_data)
            g = grad_logpost_minibatch(theta, (X[idx], Y[idx]))

            # Use generic preconditioning update
            inv_sqrt, new_precond_state = precond_update(
                g, precond_state, precond, beta1, beta2, eps, bias_correction
            )

            # For Adam, use the first moment for drift; for RMSProp, use raw gradient
            if precond == "adam":
                drift = 0.5 * step_size * (new_precond_state.m * inv_sqrt)
            else:  # rmsprop or fallback
                drift = 0.5 * step_size * (g * inv_sqrt)

            noise = (
                jax.random.normal(k_noise, theta.shape) * jnp.sqrt(step_size) * inv_sqrt
            )
            theta_new = theta + drift + noise
            return (
                theta_new.astype(theta.dtype),
                new_precond_state,
            ), None

        # vmap the preconditioned step across chains
        step_vmapped = jax.jit(jax.vmap(precond_step_single, in_axes=(0, 0)))

        # Initialize batched state with DiagPrecondState
        zeros = jnp.zeros_like(init_thetas)
        time_init = jnp.zeros((init_thetas.shape[0],), dtype=init_thetas.dtype)
        precond_states = DiagPrecondState(m=zeros, v=zeros, t=time_init)
        init_state = (init_thetas, precond_states)

        def position_fn(s):
            return s[0]  # extract theta from (theta, DiagPrecondState)

    # Prepare RNG table (T, C) - for all steps
    n_chains = init_thetas.shape[0]
    keys = jax.random.split(rng_key, n_steps * n_chains)
    keys = keys.reshape(n_steps, n_chains, -1)  # (T, C, 2)

    # Batch the loss evaluation
    Ln_vmapped = jax.jit(jax.vmap(Ln_eval_f64))

    # Use the batched driver
    result = drive_chains_batched(
        rng_keys=keys,
        init_state=init_state,
        step_fn_vmapped=step_vmapped,
        n_steps=n_steps,
        warmup=warmup,
        eval_every=eval_every,
        thin=thin,
        position_fn=position_fn,
        Ln_eval_f64_vmapped=Ln_vmapped,
        tiny_store_fn=tiny_store_fn,
        info_extractors={},  # SGLD returns no step info
    )

    # SGLD has no adaptation; set warmup fields to 0
    result.warmup_time_seconds = 0.0
    result.warmup_grads = 0

    return result


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
            inv_sqrt, new_precond_state = precond_update(
                g, precond_state, precond, beta1, beta2, eps, bias_correction
            )

            # For Adam, use the first moment for drift; for RMSProp, use raw gradient
            if precond == "adam":
                drift = 0.5 * step_size * (new_precond_state.m * inv_sqrt)
            else:  # rmsprop or fallback
                drift = 0.5 * step_size * (g * inv_sqrt)

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


def run_sghmc_chains_batched(
    *,
    rng_key: jax.Array,
    init_thetas: jax.Array,  # (C, d)
    grad_logpost_minibatch,
    X: jax.Array,
    Y: jax.Array,
    n_data: int,
    step_size: float,
    temperature: float = 1.0,
    draws: int,
    eval_every: int,
    thin: int,
    batch_size: int,
    Ln_eval_f64,
    tiny_store_fn=None,
    diag_dims=None,
    Rproj=None,
) -> BatchedResult:
    """Parallel SGHMC with vmap + lax.scan (canonical)."""
    from .base import build_tiny_store

    C = init_thetas.shape[0]

    # Build tiny store function if not provided
    if tiny_store_fn is None:
        tiny_store_fn = build_tiny_store(diag_dims, Rproj)

    # Build SGHMC kernel
    sghmc = blackjax.sghmc(grad_logpost_minibatch)
    step_single = jax.jit(sghmc.step)  # (key, theta, (Xb,Yb), step, temp) -> theta_new
    step_vmapped = jax.jit(jax.vmap(step_single, in_axes=(0, 0, 0, None, None)))

    # RNG table (T, C, 2)
    keys = jax.random.split(rng_key, draws * C).reshape(draws, C, -1)

    # Initial “state” per chain is just θ
    init_state = init_thetas

    # Build vmapped full-data Ln evaluator
    Ln_vmapped = jax.jit(jax.vmap(Ln_eval_f64))

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

    # Drive the batched scan
    result = drive_chains_batched(
        rng_keys=keys,
        init_state=init_state,
        step_fn_vmapped=step_fn_vmapped,
        n_steps=draws,
        warmup=0,
        eval_every=eval_every,
        thin=thin,
        position_fn=lambda s: s,  # state is θ
        Ln_eval_f64_vmapped=Ln_vmapped,
        tiny_store_fn=tiny_store_fn,
        info_extractors={},  # no per-step scalars recorded
    )
    return result


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


def run_hmc_chains_batched(
    *,
    rng_key,
    init_thetas,  # (C, d)  (start near θ*)
    logpost_and_grad,  # theta -> (logpost, grad)
    draws: int,
    warmup_draws: int,
    L: int,
    eval_every: int,
    thin: int,
    Ln_eval_f64,
    tiny_store_fn=None,
) -> BatchedResult:
    """
    Run multiple HMC chains in parallel using vmap + lax.scan.

    Adaptation is done once on a single chain, then parameters are
    broadcast to all chains for fast parallel sampling.
    """

    def logdensity(theta):
        val, _ = logpost_and_grad(theta)
        return val

    # --- 1) Single-chain adaptation ---
    wa = blackjax.window_adaptation(
        blackjax.hmc, logdensity, is_mass_matrix_diagonal=True, num_integration_steps=L
    )
    k_adapt, k_draws = jax.random.split(rng_key)

    # Time the warmup phase
    t_warmup = time.time()
    (state1, params), _ = wa.run(k_adapt, init_thetas[0], num_steps=warmup_draws)
    warmup_time = time.time() - t_warmup

    # Calculate warmup grads (Velocity-Verlet ~ (L+1) grads per draw)
    warmup_grads = warmup_draws * (L + 1)

    invM = params.get("inverse_mass_matrix", jnp.ones_like(init_thetas[0]))

    # --- 2) Build kernel with tuned params & broadcast initial states ---
    step_kernel = blackjax.hmc(
        logdensity,
        step_size=params["step_size"],
        inverse_mass_matrix=invM,
        num_integration_steps=L,
    ).step
    step_kernel = jax.jit(step_kernel)

    # Initial HMC states per chain (same tuned params, positions = init_thetas)
    def init_state(theta):
        return state1._replace(position=theta)

    state0 = jax.vmap(init_state)(init_thetas)  # pytree with leading (C, ...)

    # vmapped step: (keys[C], state[C]) -> (state[C], info[C])
    step_vmapped = jax.jit(jax.vmap(step_kernel, in_axes=(0, 0)))

    # RNG table
    n_chains = init_thetas.shape[0]
    keys = jax.random.split(k_draws, draws * n_chains)
    keys = keys.reshape(draws, n_chains, -1)  # (T, C, 2)

    Ln_vmapped = jax.jit(jax.vmap(Ln_eval_f64))

    # Record acceptance and energy at eval points (scalar per chain)
    info_extractors = {
        "accept": lambda info: getattr(
            info, "acceptance_rate", jnp.zeros((init_thetas.shape[0],))
        ),
        "energy": lambda info: getattr(
            info, "energy", jnp.zeros((init_thetas.shape[0],))
        ),
    }

    result = drive_chains_batched(
        rng_keys=keys,
        init_state=state0,
        step_fn_vmapped=step_vmapped,
        n_steps=draws,
        warmup=0,  # warmup already done
        eval_every=eval_every,
        thin=thin,
        position_fn=lambda st: st.position,
        Ln_eval_f64_vmapped=Ln_vmapped,
        tiny_store_fn=tiny_store_fn,
        info_extractors=info_extractors,
    )

    # Set warmup timing and work
    result.warmup_time_seconds = warmup_time
    result.warmup_grads = warmup_grads

    return result


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


def run_mclmc_chains_batched(
    *,
    rng_key,
    init_thetas,  # (C, d)
    logdensity_fn,
    draws: int,
    eval_every: int,
    thin: int,
    Ln_eval_f64,
    tuner_steps=2000,
    diagonal_preconditioning=False,
    desired_energy_var=5e-4,
    integrator_name="isokinetic_mclachlan",
    tiny_store_fn=None,
) -> BatchedResult:
    """
    Run multiple MCLMC chains in parallel using vmap + lax.scan.

    Tuning is done once on a single chain, then parameters are
    broadcast to all chains for fast parallel sampling.
    """
    integrator = getattr(blackjax.mcmc.integrators, integrator_name)

    def build_kernel(inverse_mass_matrix):
        return blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=integrator,
            inverse_mass_matrix=inverse_mass_matrix,
        )

    # Tune on one chain
    state0 = blackjax.mcmc.mclmc.init(
        position=init_thetas[0], logdensity_fn=logdensity_fn, rng_key=rng_key
    )

    # Time the tuning phase
    t_tuning = time.time()
    tuned_state, tuned_params, _ = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=build_kernel,
        num_steps=tuner_steps,
        state=state0,
        rng_key=rng_key,
        diagonal_preconditioning=diagonal_preconditioning,
        desired_energy_var=desired_energy_var,
    )
    tuning_time = time.time() - t_tuning
    alg = blackjax.mclmc(
        logdensity_fn, L=tuned_params.L, step_size=tuned_params.step_size
    )
    step = jax.jit(alg.step)

    # Broadcast initial state
    def init_state(theta):
        return tuned_state._replace(position=theta)

    state_init = jax.vmap(init_state)(init_thetas)

    step_vmapped = jax.jit(jax.vmap(step, in_axes=(0, 0)))

    # RNG table
    n_chains = init_thetas.shape[0]
    keys = jax.random.split(rng_key, draws * n_chains)
    keys = keys.reshape(draws, n_chains, -1)  # (T, C, 2)

    Ln_vmapped = jax.jit(jax.vmap(Ln_eval_f64))

    info_extractors = {
        "energy": lambda info: getattr(
            info, "energy_change", jnp.zeros((init_thetas.shape[0],))
        )
    }

    result = drive_chains_batched(
        rng_keys=keys,
        init_state=state_init,
        step_fn_vmapped=step_vmapped,
        n_steps=draws,
        warmup=0,
        eval_every=eval_every,
        thin=thin,
        position_fn=lambda st: st.position,
        Ln_eval_f64_vmapped=Ln_vmapped,
        tiny_store_fn=tiny_store_fn,
        info_extractors=info_extractors,
    )

    # Set tuning time as warmup time (MCLMC doesn't count grads in tuning)
    result.warmup_time_seconds = tuning_time
    result.warmup_grads = 0  # We don't count grads for MCLMC tuning

    return result


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
