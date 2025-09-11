# llc/samplers/adapters.py
from __future__ import annotations
from typing import Callable
import time

import jax
import jax.numpy as jnp
import numpy as np
import blackjax

from .base import drive_chain, ChainResult, default_tiny_store

Array = jnp.ndarray

# ---------- SGLD (BlackJAX 1.2.5 returns new_position only) ----------


def run_sgld_chain(
    *,
    rng_key: Array,
    init_theta: Array,
    grad_logpost_minibatch,  # (theta, (Xb,Yb)) -> grad
    X: Array,
    Y: Array,
    n_data: int,
    step_size: float,
    n_steps: int,
    warmup: int,
    batch_size: int,
    eval_every: int,
    thin: int,
    Ln_eval_f64,
    tiny_store_fn=default_tiny_store,
    use_tqdm=True,
    progress_label="SGLD",
    progress_update_every=50,
    # Optional accounting callback: stats.n_sgld_minibatch_grads += 1 per step
    work_bump: Callable | None = None,
    # NEW: optional preconditioning
    precond_mode: str = "none",  # "none" | "rmsprop" | "adam"
    beta1: float = 0.9,  # Adam first moment
    beta2: float = 0.999,  # RMSProp/Adam second moment
    eps: float = 1e-8,  # numerical stabilizer
    bias_correction: bool = True,  # Adam bias correction
) -> ChainResult:
    precond = (precond_mode or "none").lower()

    if precond == "none":
        # Original plain SGLD path (unchanged)
        sgld = blackjax.sgld(grad_logpost_minibatch)
        step_plain = jax.jit(sgld.step)

        def step_fn(key: Array, theta: Array):
            # minibatch indices from key
            k_noise, k_batch = jax.random.split(key)
            idx = jax.random.randint(k_batch, (batch_size,), 0, n_data)
            new_theta = step_plain(k_noise, theta, (X[idx], Y[idx]), step_size)
            if work_bump:
                work_bump()  # one minibatch gradient
            return new_theta  # no info

        init_state = init_theta
        position_fn = lambda s: s
        step_returns_info = False

    else:
        # Preconditioned SGLD: RMSProp-style ("rmsprop") or Adam-style ("adam")
        # State = (theta, m, v, t) where m is first moment (adam), v is second moment EMA
        @jax.jit
        def precond_step(key: Array, state):
            theta, m, v, t = state
            k_noise, k_batch = jax.random.split(key)
            idx = jax.random.randint(k_batch, (batch_size,), 0, n_data)
            g = grad_logpost_minibatch(theta, (X[idx], Y[idx]))  # ascent on log π

            if precond == "rmsprop":
                m_new = m  # unused
                v_new = beta2 * v + (1.0 - beta2) * (g * g)
                v_hat = v_new  # no bias correction
                inv_sqrt = jax.lax.rsqrt(v_hat + eps)  # 1/sqrt(v_hat + eps)
                drift = 0.5 * step_size * (g * inv_sqrt)
            else:  # "adam"
                m_new = beta1 * m + (1.0 - beta1) * g
                v_new = beta2 * v + (1.0 - beta2) * (g * g)
                if bias_correction:
                    t1 = t + 1.0
                    m_hat = m_new / (1.0 - beta1**t1)
                    v_hat = v_new / (1.0 - beta2**t1)
                else:
                    m_hat, v_hat = m_new, v_new
                inv_sqrt = jax.lax.rsqrt(v_hat + eps)
                drift = 0.5 * step_size * (m_hat * inv_sqrt)

            noise = jax.random.normal(k_noise, theta.shape) * jnp.sqrt(step_size) * inv_sqrt
            theta_new = theta + drift + noise
            return (theta_new, m_new, v_new, t + 1.0)

        # Wrap for drive_chain
        def step_fn(key: Array, state):
            new_state = precond_step(key, state)
            if work_bump:
                work_bump()  # one minibatch gradient per step
            return new_state

        # Initialize state for this chain
        zeros = jnp.zeros_like(init_theta)
        init_state = (init_theta, zeros, zeros, jnp.array(0.0, dtype=init_theta.dtype))
        position_fn = lambda s: s[0]  # extract theta
        step_returns_info = False

    return drive_chain(
        rng_key=rng_key,
        init_state=init_state,
        step_fn=step_fn,
        step_returns_info=step_returns_info,
        n_steps=n_steps,
        warmup=warmup,
        eval_every=eval_every,
        thin=thin,
        position_fn=position_fn,
        Ln_eval_f64=Ln_eval_f64,
        tiny_store_fn=tiny_store_fn,
        use_tqdm=use_tqdm,
        progress_label=progress_label,
        progress_update_every=progress_update_every,
    )


# ---------- HMC (with window adaptation) ----------


def run_hmc_chain(
    *,
    rng_key: Array,
    init_theta: Array,
    logpost_and_grad,  # theta -> (logpost, grad)
    draws: int,
    warmup: int,
    L: int,
    eval_every: int,
    thin: int,
    Ln_eval_f64,
    tiny_store_fn=default_tiny_store,
    use_tqdm=True,
    progress_label="HMC",
    progress_update_every=50,
    work_bump: Callable | None = None,  # bump by (L+1) per draw
) -> ChainResult:
    def logdensity(theta):
        val, _ = logpost_and_grad(theta)
        return val

    wa = blackjax.window_adaptation(
        blackjax.hmc, logdensity, is_mass_matrix_diagonal=True, num_integration_steps=L
    )

    # Time the warmup phase
    t0_warmup = time.time()
    (state, params), _ = wa.run(rng_key, init_theta, num_steps=warmup)
    warmup_time = time.time() - t0_warmup
    invM = params.get("inverse_mass_matrix", jnp.ones_like(init_theta))
    if invM.ndim == 1:  # promote vector -> diagonal matrix
        invM = jnp.diag(invM)
    kernel = blackjax.hmc(
        logdensity,
        step_size=params["step_size"],
        inverse_mass_matrix=invM,
        num_integration_steps=L,
    ).step
    step = jax.jit(kernel)

    # Prime compilation
    k_pre, k_draws = jax.random.split(rng_key)
    k_stream = jax.random.split(k_draws, max(1, draws))
    state, info = step(k_stream[0], state)  # compile
    if work_bump:
        work_bump(L + 1)

    def step_fn(key: Array, st):
        st, inf = step(key, st)
        if work_bump:
            # Velocity-Verlet ~ (L+1) grad evals per draw
            work_bump(getattr(inf, "num_integration_steps", L) + 1)
        return st, inf

    # info hook: record acceptance into extras["accept"]
    def info_hook(info, ctx):
        acc = getattr(info, "acceptance_rate", np.nan)
        if not np.isnan(acc):
            ctx["put_extra"]("accept", float(acc))

    result = drive_chain(
        rng_key=k_draws,
        init_state=state,
        step_fn=step_fn,
        step_returns_info=True,
        n_steps=draws,
        warmup=0,  # warmup already done above
        eval_every=eval_every,
        thin=thin,
        position_fn=lambda st: st.position,
        Ln_eval_f64=Ln_eval_f64,
        tiny_store_fn=tiny_store_fn,
        use_tqdm=use_tqdm,
        progress_label=progress_label,
        progress_update_every=progress_update_every,
        info_hooks=[info_hook],
    )

    # Add warmup timing and work to extras
    result.extras["warmup_time"] = np.array([warmup_time])
    # Estimate warmup work: L+1 grads per warmup step
    warmup_grads = warmup * (L + 1)
    result.extras["warmup_grads"] = np.array([warmup_grads])

    return result


# ---------- MCLMC (unadjusted; tuned L & step_size) ----------


def run_mclmc_chain(
    *,
    rng_key: Array,
    init_theta: Array,
    logdensity_fn,  # pure log π(θ)
    draws: int,
    eval_every: int,
    thin: int,
    Ln_eval_f64,
    tiny_store_fn=default_tiny_store,
    tuner_steps: int = 2000,
    diagonal_preconditioning: bool = False,
    desired_energy_var: float = 5e-4,
    integrator_name: str = "isokinetic_mclachlan",
    use_tqdm=True,
    progress_label="MCLMC",
    progress_update_every=50,
    work_bump: Callable | None = None,  # if provided, bump per step
) -> ChainResult:
    integrator = getattr(blackjax.mcmc.integrators, integrator_name)

    def build_kernel(inverse_mass_matrix):
        return blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=integrator,
            inverse_mass_matrix=inverse_mass_matrix,
        )

    state0 = blackjax.mcmc.mclmc.init(
        position=init_theta, logdensity_fn=logdensity_fn, rng_key=rng_key
    )
    tuned_state, tuned_params, _ = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=build_kernel,
        num_steps=tuner_steps,
        state=state0,
        rng_key=rng_key,
        diagonal_preconditioning=diagonal_preconditioning,
        desired_energy_var=desired_energy_var,
    )
    alg = blackjax.mclmc(
        logdensity_fn, L=tuned_params.L, step_size=tuned_params.step_size
    )
    step = jax.jit(alg.step)

    # Prime compilation
    k_pre, k_draws = jax.random.split(rng_key)
    ks = jax.random.split(k_draws, max(1, draws))
    st, info = step(ks[0], tuned_state)

    def step_fn(key: Array, st):
        st, inf = step(key, st)
        if work_bump:
            work_bump(1)
        return st, inf

    # info hook: record energy change into extras["energy"]
    def info_hook(info, ctx):
        dE = getattr(info, "energy_change", None)
        if dE is not None:
            ctx["put_extra"]("energy", float(dE))

    return drive_chain(
        rng_key=k_draws,
        init_state=tuned_state,
        step_fn=step_fn,
        step_returns_info=True,
        n_steps=draws,
        warmup=0,
        eval_every=eval_every,
        thin=thin,
        position_fn=lambda st: st.position,
        Ln_eval_f64=Ln_eval_f64,
        tiny_store_fn=tiny_store_fn,
        use_tqdm=use_tqdm,
        progress_label=progress_label,
        progress_update_every=progress_update_every,
        info_hooks=[info_hook],
    )
