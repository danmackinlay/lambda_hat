# llc/runners.py
"""Sampler orchestration and runner utilities"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
import jax
import numpy as np

from .samplers.base import default_tiny_store
from .samplers.adapters import (
    run_sgld_chain, run_hmc_chain, run_mclmc_chain,
    run_sgld_chains_batched, run_hmc_chains_batched, run_mclmc_chains_batched
)

if TYPE_CHECKING:
    from .config import Config
else:
    # Runtime import to avoid circular dependency
    Config = "Config"


@dataclass
class RunStats:
    """Statistics tracking for computational work and timing"""

    # wall-clock
    t_build: float = 0.0
    t_train: float = 0.0
    t_sgld_warmup: float = 0.0
    t_sgld_sampling: float = 0.0
    t_hmc_warmup: float = 0.0
    t_hmc_sampling: float = 0.0
    t_mclmc_warmup: float = 0.0
    t_mclmc_sampling: float = 0.0

    # work counters (proxy for computational work)
    # Count "gradient-equivalent" evaluations to compare samplers.
    # - SGLD: ~1 minibatch gradient per step -> +1
    # - HMC: ~num_integration_steps gradients per draw (leapfrog) -> +L
    # - Add full-data loss evals and log-prob grads as separate counters for transparency.
    n_sgld_minibatch_grads: int = 0
    n_sgld_full_loss: int = 0
    n_hmc_leapfrog_grads: int = 0
    n_hmc_full_loss: int = 0
    n_hmc_warmup_leapfrog_grads: int = 0  # estimated during adaptation
    n_mclmc_steps: int = 0
    n_mclmc_full_loss: int = 0


def tic():
    """Start timing"""
    return time.perf_counter()


def toc(t0):
    """End timing and return elapsed seconds"""
    return time.perf_counter() - t0


def llc_from_running_mean(E_L, L0, n, beta):
    """Compute LLC from running mean of loss values"""
    return float(n * beta * (E_L - L0))


def stack_thinned(kept_list):  # list of (draws, dim)
    """Stack thinned samples, truncating to common length to avoid NaN padding"""
    m = min(k.shape[0] for k in kept_list)
    if m == 0:
        return np.empty((len(kept_list), 0, kept_list[0].shape[1]))
    return np.stack([k[:m] for k in kept_list], axis=0)


def run_sgld_online(
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
    use_tqdm=True,
    progress_update_every=50,
    stats: RunStats | None = None,
    diag_dims=None,
    Rproj=None,
    # NEW: preconditioning options (threaded through to adapter)
    precond_mode: str = "none",
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bias_correction: bool = True,
):
    """Run SGLD chains with online LLC evaluation"""
    chains = init_thetas.shape[0]
    kept_all, means, vars_, ns, L_histories = [], [], [], [], []

    def tiny_store(vec: np.ndarray):
        return default_tiny_store(vec, diag_dims, Rproj)

    for c in range(chains):
        ck = jax.random.fold_in(key, c)
        # Work accounting: per SGLD step add one minibatch grad
        work_bump = (
            (
                lambda: setattr(
                    stats, "n_sgld_minibatch_grads", stats.n_sgld_minibatch_grads + 1
                )
            )
            if stats
            else None
        )

        # Time the sampling
        t0 = time.time()
        res = run_sgld_chain(
            rng_key=ck,
            init_theta=init_thetas[c],
            grad_logpost_minibatch=grad_logpost_minibatch,
            X=X,
            Y=Y,
            n_data=n,
            step_size=step_size,
            n_steps=num_steps,
            warmup=warmup,
            batch_size=batch_size,
            eval_every=eval_every,
            thin=thin,
            Ln_eval_f64=Ln_full64,
            tiny_store_fn=tiny_store,
            use_tqdm=use_tqdm,
            progress_label=f"SGLD(c{c})",
            progress_update_every=progress_update_every,
            work_bump=work_bump,
            precond_mode=precond_mode,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            bias_correction=bias_correction,
        )
        # Accumulate sampling time (subtract eval time)
        elapsed = time.time() - t0
        if stats and hasattr(res, "eval_time_seconds"):
            net_time = max(0.0, elapsed - res.eval_time_seconds)
            # naive split: attribute first (warmup/num_steps) fraction to warmup
            frac = float(warmup) / float(max(1, num_steps))
            stats.t_sgld_warmup += frac * net_time
            stats.t_sgld_sampling += (1.0 - frac) * net_time

        kept_all.append(res.kept)
        means.append(res.mean_L)
        vars_.append(res.var_L)
        ns.append(res.n_L)
        L_histories.append(res.L_hist)
        if stats:
            stats.n_sgld_full_loss += int(res.n_L)

    samples_thin = stack_thinned(kept_all)
    return samples_thin, np.array(means), np.array(vars_), np.array(ns), L_histories


def run_hmc_online_with_adaptation(
    key,
    init_thetas,
    logpost_and_grad,
    num_draws,
    warmup_draws,
    L,
    eval_every,
    thin,
    Ln_full64,
    use_tqdm=True,
    progress_update_every=50,
    stats: RunStats | None = None,
    diag_dims=None,
    Rproj=None,
):
    """Run HMC chains with window adaptation and online LLC evaluation"""
    chains = init_thetas.shape[0]
    kept_all, means, vars_, ns, accs, L_histories = [], [], [], [], [], []

    def tiny_store(vec: np.ndarray):
        return default_tiny_store(vec, diag_dims, Rproj)

    for c in range(chains):
        ck = jax.random.fold_in(key, c)
        # Work accounting
        work_bump = (
            (
                lambda n_grads=1: setattr(
                    stats, "n_hmc_leapfrog_grads", stats.n_hmc_leapfrog_grads + n_grads
                )
            )
            if stats
            else None
        )

        # Time the sampling
        t0 = time.time()
        res = run_hmc_chain(
            rng_key=ck,
            init_theta=init_thetas[c],
            logpost_and_grad=logpost_and_grad,
            draws=num_draws,
            warmup=warmup_draws,
            L=L,
            eval_every=eval_every,
            thin=thin,
            Ln_eval_f64=Ln_full64,
            tiny_store_fn=tiny_store,
            use_tqdm=use_tqdm,
            progress_label=f"HMC(c{c})",
            progress_update_every=progress_update_every,
            work_bump=work_bump,
        )
        # Accumulate sampling time (subtract eval time)
        elapsed = time.time() - t0
        if stats and hasattr(res, "eval_time_seconds"):
            stats.t_hmc_sampling += max(0.0, elapsed - res.eval_time_seconds)

        kept_all.append(res.kept)
        means.append(res.mean_L)
        vars_.append(res.var_L)
        ns.append(res.n_L)
        L_histories.append(res.L_hist)
        # Extract acceptance rates from extras
        if "accept" in res.extras:
            accs.append(np.asarray(res.extras["accept"]))
        else:
            accs.append(np.asarray([]))
        if stats:
            stats.n_hmc_full_loss += int(res.n_L)
            # Extract warmup timing and work from extras
            if "warmup_time" in res.extras:
                stats.t_hmc_warmup += float(res.extras["warmup_time"][0])
            if "warmup_grads" in res.extras:
                stats.n_hmc_warmup_leapfrog_grads += int(res.extras["warmup_grads"][0])

    samples_thin = stack_thinned(kept_all)
    return (
        samples_thin,
        np.array(means),
        np.array(vars_),
        np.array(ns),
        accs,
        L_histories,
    )


def run_mclmc_online(
    key,
    init_thetas,
    logdensity_fn,
    num_draws,
    eval_every,
    thin,
    Ln_full64,
    tuner_steps=2000,
    diagonal_preconditioning=False,
    desired_energy_var=5e-4,
    integrator_name="isokinetic_mclachlan",
    use_tqdm=True,
    progress_update_every=50,
    stats: RunStats | None = None,
    diag_dims=None,
    Rproj=None,
):
    """Run MCLMC chains with auto-tuning and online LLC evaluation"""
    chains = init_thetas.shape[0]
    kept_all, means, vars_, ns, energy_deltas, L_histories = [], [], [], [], [], []

    def tiny_store(vec: np.ndarray):
        return default_tiny_store(vec, diag_dims, Rproj)

    for c in range(chains):
        ck = jax.random.fold_in(key, c)
        # Work accounting
        work_bump = (
            (
                lambda n_steps=1: setattr(
                    stats, "n_mclmc_steps", stats.n_mclmc_steps + n_steps
                )
            )
            if stats
            else None
        )

        # Time the sampling
        t0 = time.time()
        res = run_mclmc_chain(
            rng_key=ck,
            init_theta=init_thetas[c],
            logdensity_fn=logdensity_fn,
            draws=num_draws,
            eval_every=eval_every,
            thin=thin,
            Ln_eval_f64=Ln_full64,
            tuner_steps=tuner_steps,
            diagonal_preconditioning=diagonal_preconditioning,
            desired_energy_var=desired_energy_var,
            integrator_name=integrator_name,
            tiny_store_fn=tiny_store,
            use_tqdm=use_tqdm,
            progress_label=f"MCLMC(c{c})",
            progress_update_every=progress_update_every,
            work_bump=work_bump,
        )
        # Accumulate sampling time (subtract eval time)
        elapsed = time.time() - t0
        if stats and hasattr(res, "eval_time_seconds"):
            net_time = max(0.0, elapsed - res.eval_time_seconds)
            # naive split: attribute (tuner_steps/(tuner_steps+num_draws)) to warmup
            total_steps = tuner_steps + num_draws
            frac = float(tuner_steps) / float(max(1, total_steps))
            stats.t_mclmc_warmup += frac * net_time
            stats.t_mclmc_sampling += (1.0 - frac) * net_time

        kept_all.append(res.kept)
        means.append(res.mean_L)
        vars_.append(res.var_L)
        ns.append(res.n_L)
        L_histories.append(res.L_hist)
        # Extract energy deltas from extras
        if "energy" in res.extras:
            energy_deltas.append(np.asarray(res.extras["energy"]))
        else:
            energy_deltas.append(np.asarray([]))
        if stats:
            stats.n_mclmc_full_loss += int(res.n_L)

    samples_thin = stack_thinned(kept_all)
    return (
        samples_thin,
        np.array(means),
        np.array(vars_),
        np.array(ns),
        energy_deltas,
        L_histories,
    )


def run_sampler(sampler_name: str, sampler_cfg, **shared_kwargs):
    """Unified sampler dispatcher"""
    if sampler_name == "sgld":
        return run_sgld_online(**sampler_cfg, **shared_kwargs)
    elif sampler_name == "hmc":
        return run_hmc_online_with_adaptation(**sampler_cfg, **shared_kwargs)
    elif sampler_name == "mclmc":
        return run_mclmc_online(**sampler_cfg, **shared_kwargs)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


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
    """Batched SGLD runner using vmap + lax.scan for speed"""
    def tiny_store(vec_batch):
        # vec_batch is (C, d), apply tiny store to each chain
        if diag_dims is not None:
            return vec_batch[:, diag_dims]  # (C, k)
        elif Rproj is not None:
            return jax.vmap(lambda v: Rproj @ v)(vec_batch)  # (C, k)
        else:
            return default_tiny_store(vec_batch)

    result = run_sgld_chains_batched(
        rng_key=key,
        init_thetas=init_thetas,
        grad_logpost_minibatch=grad_logpost_minibatch,
        X=X, Y=Y, n_data=n,
        step_size=step_size,
        n_steps=num_steps,
        warmup=warmup,
        batch_size=batch_size,
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64=Ln_full64,
        tiny_store_fn=tiny_store,
        precond_mode=precond_mode,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        bias_correction=bias_correction,
    )
    
    # Convert back to the expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    means = np.asarray(result.mean_L)  # (C,)
    vars_ = np.asarray(result.var_L)   # (C,)
    ns = np.asarray(result.n_L)        # (C,)
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]
    
    return kept_stacked, means, vars_, ns, L_histories


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
    """Batched HMC runner using vmap + lax.scan for speed"""
    def tiny_store(vec_batch):
        if diag_dims is not None:
            return vec_batch[:, diag_dims]
        elif Rproj is not None:
            return jax.vmap(lambda v: Rproj @ v)(vec_batch)
        else:
            return default_tiny_store(vec_batch)

    result = run_hmc_chains_batched(
        rng_key=key,
        init_thetas=init_thetas,
        logpost_and_grad=logpost_and_grad,
        draws=draws,
        warmup_draws=warmup,
        L=L,
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    
    # Convert back to expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    means = np.asarray(result.mean_L)       # (C,)
    vars_ = np.asarray(result.var_L)        # (C,)
    ns = np.asarray(result.n_L)             # (C,)
    acc = result.extras.get("accept", jnp.zeros_like(result.L_hist))
    acc_list = [np.asarray(acc[c]) for c in range(acc.shape[0])]
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]
    
    return kept_stacked, means, vars_, ns, acc_list, L_histories


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
    """Batched MCLMC runner using vmap + lax.scan for speed"""
    def tiny_store(vec_batch):
        if diag_dims is not None:
            return vec_batch[:, diag_dims]
        elif Rproj is not None:
            return jax.vmap(lambda v: Rproj @ v)(vec_batch)
        else:
            return default_tiny_store(vec_batch)

    result = run_mclmc_chains_batched(
        rng_key=key,
        init_thetas=init_thetas,
        logdensity_fn=logdensity_fn,
        draws=draws,
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64=Ln_full64,
        tuner_steps=tuner_steps,
        diagonal_preconditioning=diagonal_preconditioning,
        desired_energy_var=desired_energy_var,
        integrator_name=integrator_name,
        tiny_store_fn=tiny_store,
    )
    
    # Convert back to expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    means = np.asarray(result.mean_L)       # (C,)
    vars_ = np.asarray(result.var_L)        # (C,)
    ns = np.asarray(result.n_L)             # (C,)
    dE = result.extras.get("energy", jnp.zeros_like(result.L_hist))
    dE_list = [np.asarray(dE[c]) for c in range(dE.shape[0])]
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]
    
    return kept_stacked, means, vars_, ns, dE_list, L_histories
