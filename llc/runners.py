# llc/runners.py
"""
Batched runners only. Each runner returns a `SamplerResult` with: ragged `Ln_histories`,
thinned `theta`, `acceptance`/`energy` (if any), and **filled** `timings`/`work`.
Timing = wall-clock − eval-time estimate − warmup; Work = deterministic counts.
Use these for ESS/sec and WNV in metrics.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
import numpy as np
from jax import numpy as jnp

from .samplers.adapters import (
    sgld_spec,
    sghmc_spec,
    hmc_spec,
    mclmc_spec,
)
from .samplers.base import build_tiny_store, run_sampler_spec, DiagPrecondState
from .types import SamplerResult

if TYPE_CHECKING:
    pass


def tic():
    """Start timing"""
    return time.perf_counter()


def toc(t0):
    """End timing and return elapsed seconds"""
    return time.perf_counter() - t0


# ---------- Batched (fast) versions of the above runners ----------


def run_sgld_online_batched(
    key,
    init_thetas,
    grad_logpost_minibatch,
    X,
    Y,
    n,
    step_size,
    num_steps,
    warmup,
    batch_size,
    eval_every,
    thin,
    Ln_full64,
    diag_dims=None,
    Rproj=None,
    # preconditioning options
    precond_mode: str = "none",
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bias_correction: bool = True,
    # Ignored parameters for compatibility
    use_tqdm=None,
    progress_update_every=None,
    stats=None,
):
    """Batched SGLD runner using SamplerSpec + run_sampler_spec"""
    tiny_store = build_tiny_store(diag_dims, Rproj)

    # Create SGLD SamplerSpec
    spec = sgld_spec(
        grad_logpost_minibatch=grad_logpost_minibatch,
        X=X,
        Y=Y,
        n_data=n,
        step_size=step_size,
        batch_size=batch_size,
        precond_mode=precond_mode,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        bias_correction=bias_correction,
    )

    # Wrap initial state for preconditioning if needed
    C, d = init_thetas.shape
    if precond_mode != "none":
        precond_state = DiagPrecondState(
            m=jnp.zeros((C, d), dtype=init_thetas.dtype),
            v=jnp.zeros((C, d), dtype=init_thetas.dtype),
            t=jnp.zeros((C,), dtype=init_thetas.dtype),
        )
        init_states = (init_thetas, precond_state)
        # Sanity check for future-proofing
        if not isinstance(init_states, tuple):
            raise RuntimeError("SGLD preconditioning requires tuple state: (theta, DiagPrecondState).")
    else:
        init_states = init_thetas

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key,
        init_states=init_states,
        n_steps=num_steps,
        warmup=warmup,
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64_vmapped=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time

    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(
        total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M
    )

    # Convert back to the expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    means = np.asarray(result.mean_L)  # (C,)
    vars_ = np.asarray(result.var_L)  # (C,)
    ns = np.asarray(result.n_L)  # (C,)
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]

    # Build timing and work dictionaries
    timings = {
        "warmup": float(result.warmup_time_seconds),
        "sampling": float(sampling_time),
    }

    # SGLD work computation
    chains = init_thetas.shape[0]
    work = {
        "n_minibatch_grads": int(chains * num_steps),
        "n_full_loss": int(chains * M),
    }

    return SamplerResult(
        Ln_histories=L_histories,
        theta_thin=kept_stacked,
        acceptance=None,
        energy=None,
        timings=timings,
        work=work,
    )


def run_sghmc_online_batched(
    key,
    init_thetas,
    grad_logpost_minibatch,
    X,
    Y,
    n_data,
    step_size,
    temperature,
    draws,
    eval_every,
    thin,
    batch_size,
    Ln_full64,
    diag_dims=None,
    Rproj=None,
    # Ignored parameters for compatibility
    use_tqdm=None,
    progress_update_every=None,
    stats=None,
):
    """Batched SGHMC runner using SamplerSpec + run_sampler_spec"""
    tiny_store = build_tiny_store(diag_dims, Rproj)

    # Create SGHMC SamplerSpec
    spec = sghmc_spec(
        grad_logpost_minibatch=grad_logpost_minibatch,
        X=X,
        Y=Y,
        n_data=n_data,
        step_size=step_size,
        temperature=temperature,
        batch_size=batch_size,
    )

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key,
        init_states=init_thetas,
        n_steps=draws,
        warmup=0,  # SGHMC has no warmup
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64_vmapped=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time

    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(
        total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M
    )

    # Convert back to the expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]

    # Build timing and work dictionaries
    timings = {
        "warmup": float(result.warmup_time_seconds),
        "sampling": float(sampling_time),
    }

    # SGHMC work computation (similar to SGLD)
    chains = init_thetas.shape[0]
    work = {
        "n_minibatch_grads": int(chains * draws),
        "n_full_loss": int(chains * M),
    }

    return SamplerResult(
        Ln_histories=L_histories,
        theta_thin=kept_stacked,
        acceptance=None,
        energy=None,
        timings=timings,
        work=work,
    )


def run_hmc_online_batched(
    key,
    init_thetas,
    logpost_and_grad,
    draws,
    warmup,
    L,
    eval_every,
    thin,
    Ln_full64,
    diag_dims=None,
    Rproj=None,
    # Ignored parameters for compatibility
    use_tqdm=None,
    progress_update_every=None,
    stats=None,
):
    """Batched HMC runner using SamplerSpec + run_sampler_spec"""
    import blackjax
    import jax
    from jax import numpy as jnp

    tiny_store = build_tiny_store(diag_dims, Rproj)

    def logdensity(theta):
        val, _ = logpost_and_grad(theta)
        return val

    # ---- Window adaptation (single chain) ----
    # Use the first chain's init as the adaptation reference; cheap and robust.
    key_tune, key_run = jax.random.split(key)
    x0 = init_thetas[0]
    # Target acceptance ~ 0.8 is a common default
    wa = blackjax.window_adaptation(blackjax.hmc, logdensity, num_integration_steps=L)
    (adapted_state, adaptation_info) = wa.run(key_tune, x0, num_steps=warmup)
    # Extract tuned parameters from adapted_state.parameters
    step_size = adapted_state.parameters["step_size"]
    inv_mass = adapted_state.parameters["inverse_mass_matrix"]

    # ---- Build the tuned kernel; then vmap across chains ----
    hmc_kernel = blackjax.hmc(
        logdensity,
        step_size=step_size,
        inverse_mass_matrix=inv_mass,
        num_integration_steps=L,
    )
    init_states = jax.vmap(hmc_kernel.init)(init_thetas)
    step_vmapped = jax.jit(jax.vmap(jax.jit(hmc_kernel.step), in_axes=(0, 0)))

    # Create SamplerSpec with proper HMC components
    from llc.samplers.base import SamplerSpec
    spec = SamplerSpec(
        name="hmc",
        step_vmapped=lambda keys, states: step_vmapped(keys, states),
        position_fn=lambda st: st.position,
        info_extractors={
            "accept": lambda info: info.acceptance_rate,
            "energy": lambda info: info.energy,
        },
        grads_per_step=L + 1,
    )

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key_run,
        init_states=init_states,  # Now passing HMCState objects
        n_steps=draws,
        warmup=0,  # Warmup already done during adaptation
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64_vmapped=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time

    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(
        total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M
    )

    # Convert back to expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    means = np.asarray(result.mean_L)  # (C,)
    vars_ = np.asarray(result.var_L)  # (C,)
    ns = np.asarray(result.n_L)  # (C,)
    acc = result.extras.get("accept", jnp.zeros_like(result.L_hist))
    acc_list = [np.asarray(acc[c]) for c in range(acc.shape[0])]
    energy = result.extras.get("energy", jnp.zeros_like(result.L_hist))
    energy_list = [np.asarray(energy[c]) for c in range(energy.shape[0])]
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]

    # Build timing and work dictionaries
    timings = {
        "warmup": float(result.warmup_time_seconds),
        "sampling": float(sampling_time),
    }

    # HMC work computation
    chains = init_thetas.shape[0]
    work = {
        "n_leapfrog_grads": int(chains * draws * (L + 1)),
        "n_full_loss": int(chains * M),
        "n_warmup_leapfrog_grads": int(result.warmup_grads),
    }

    return SamplerResult(
        Ln_histories=L_histories,
        theta_thin=kept_stacked,
        acceptance=acc_list,
        energy=energy_list,
        timings=timings,
        work=work,
    )


def run_mclmc_online_batched(
    key,
    init_thetas,
    logdensity_fn,
    draws,
    eval_every,
    thin,
    Ln_full64,
    tuner_steps=2000,
    diagonal_preconditioning=False,
    desired_energy_var=5e-4,
    integrator_name="isokinetic_mclachlan",
    diag_dims=None,
    Rproj=None,
    # Ignored parameters for compatibility
    use_tqdm=None,
    progress_update_every=None,
    stats=None,
):
    """Batched MCLMC runner using SamplerSpec + run_sampler_spec"""
    import blackjax
    import jax
    from jax import numpy as jnp

    tiny_store = build_tiny_store(diag_dims, Rproj)

    # ---- Tune (L, step_size) once, then share across chains ----
    key_tune, key_run = jax.random.split(key)
    x0 = init_thetas[0]
    integrator = getattr(blackjax.mcmc.integrators, integrator_name)
    # The find_* helper returns tuned integers/floats to build the kernel
    L_tuned, eps_tuned, _info = blackjax.mclmc_find_L_and_step_size(
        logdensity_fn,
        key_tune,
        x0,
        desired_energy_var=desired_energy_var,
        frac_tune3=0.1,
        num_steps_tune1=tuner_steps // 3,
        num_steps_tune2=tuner_steps // 3,
        num_steps_tune3=tuner_steps // 3,
        integrator=integrator,
        diagonal_preconditioning=bool(diagonal_preconditioning),
    )
    mclmc_kernel = blackjax.mclmc(
        logdensity_fn, L=int(L_tuned), step_size=float(eps_tuned), integrator=integrator
    )
    step_single = jax.jit(mclmc_kernel.step)

    # Build a concrete SamplerSpec with vmapped step
    from llc.samplers.base import SamplerSpec
    spec = SamplerSpec(
        name="mclmc",
        step_vmapped=jax.jit(jax.vmap(step_single, in_axes=(0, 0))),
        position_fn=lambda st: st.position,
        info_extractors={"energy": lambda info: info.energy_change},
        grads_per_step=1.0,
    )

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key_run,
        init_states=init_thetas,
        n_steps=draws,
        warmup=0,  # MCLMC uses tuner_steps, not warmup
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64_vmapped=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time

    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(
        total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M
    )

    # Convert back to expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    means = np.asarray(result.mean_L)  # (C,)
    vars_ = np.asarray(result.var_L)  # (C,)
    ns = np.asarray(result.n_L)  # (C,)
    dE = result.extras.get("energy", jnp.zeros_like(result.L_hist))
    dE_list = [np.asarray(dE[c]) for c in range(dE.shape[0])]
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]

    # Build timing and work dictionaries
    timings = {
        "warmup": float(result.warmup_time_seconds),
        "sampling": float(sampling_time),
    }

    # MCLMC work computation
    chains = init_thetas.shape[0]
    work = {
        "n_steps": int(chains * draws),
        "n_full_loss": int(chains * M),
    }

    return SamplerResult(
        Ln_histories=L_histories,
        theta_thin=kept_stacked,
        acceptance=None,  # MCLMC doesn't have acceptance rates
        energy=dE_list,  # MCLMC has energy changes
        timings=timings,
        work=work,
    )
