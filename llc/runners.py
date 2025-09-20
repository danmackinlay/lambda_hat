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
import jax
import numpy as np
from jax import numpy as jnp

from .samplers.adapters import (
    sgld_spec,
    sghmc_spec,
    hmc_spec,
    mclmc_spec,
)
from .samplers.base import build_tiny_store, run_sampler_spec
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

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key,
        init_thetas=init_thetas,
        n_steps=num_steps,
        warmup=warmup,
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time
    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M)

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
        init_thetas=init_thetas,
        n_steps=draws,
        warmup=0,  # SGHMC has no warmup
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time
    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M)

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
    tiny_store = build_tiny_store(diag_dims, Rproj)

    # Create HMC SamplerSpec
    spec = hmc_spec(
        logpost_and_grad=logpost_and_grad,
        L=L,
    )

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key,
        init_thetas=init_thetas,
        n_steps=draws,
        warmup=warmup,
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time
    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M)

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
    tiny_store = build_tiny_store(diag_dims, Rproj)

    # Create MCLMC SamplerSpec
    spec = mclmc_spec(
        logdensity_fn=logdensity_fn,
        tuner_steps=tuner_steps,
        diagonal_preconditioning=diagonal_preconditioning,
        desired_energy_var=desired_energy_var,
        integrator_name=integrator_name,
    )

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key,
        init_thetas=init_thetas,
        n_steps=draws,
        warmup=0,  # MCLMC uses tuner_steps, not warmup
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time
    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M)

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
