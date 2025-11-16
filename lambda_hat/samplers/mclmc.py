# lambda_hat/samplers/mclmc.py
"""MCLMC sampler (Microcanonical Langevin Monte Carlo) - FLAT INTERFACE ONLY"""

import time
from typing import Optional

import blackjax
import blackjax.mcmc.integrators as bj_integrators
import jax
import jax.numpy as jnp

from lambda_hat.posterior import Posterior
from lambda_hat.samplers.common import inference_loop_extended
from lambda_hat.types import SamplerRunResult

Array = jnp.ndarray


def run_mclmc(
    key: jax.random.PRNGKey,
    posterior: Posterior,
    num_samples: int,
    num_chains: int,
    config,
    loss_full_fn: Optional[callable] = None,
    n_data: Optional[int] = None,
    beta: Optional[float] = None,
    L0: Optional[float] = None,
) -> SamplerRunResult:
    """Run MCLMC sampler - FLAT INTERFACE ONLY

    Args:
        key: JRNG key
        posterior: Posterior with flat-space log density
        num_samples: Number of samples per chain
        num_chains: Number of chains to run in parallel
        config: MCLMC configuration (L, step_size, integrator)
        loss_full_fn: Optional function to compute loss for recording
        n_data: Number of data points (for LLC computation)
        beta: Temperature parameter (for LLC computation)
        L0: Reference loss (for LLC computation)

    Returns:
        SamplerRunResult with traces and timing information
    """
    # Get flat initial parameters
    flat0 = posterior.flat0
    dtype = posterior.vm.dtype

    # Diversify starting points in FLAT SPACE
    key, k_init = jax.random.split(key)
    init_thetas = flat0 + 0.01 * jax.random.normal(k_init, (num_chains, flat0.size), dtype=dtype)

    # Pick integrator
    integrators = {
        "isokinetic_mclachlan": bj_integrators.isokinetic_mclachlan,
        "isokinetic_velocity_verlet": bj_integrators.isokinetic_velocity_verlet,
    }
    integrator = integrators[config.integrator]

    # Create MCLMC kernel with flat-space log density
    mclmc = blackjax.mclmc(
        logdensity_fn=posterior.logpost_flat,
        L=config.L,
        step_size=config.step_size,
        integrator=integrator,
    )

    # init STATES — BlackJAX 1.2.5 still needs RNG keys
    key, init_key = jax.random.split(key)
    init_keys = jax.random.split(init_key, num_chains)
    init_states = jax.vmap(mclmc.init)(init_thetas, init_keys)

    # Calculate work per step (FGEs): MCLMC uses full gradients.
    # Number of integration steps is approximately L / step_size.
    num_integration_steps = jnp.ceil(config.L / config.step_size)
    work_per_step = float(num_integration_steps)

    # Define aux function for recording loss (state.position is flat array)
    def aux_fn(state):
        if loss_full_fn is not None:
            # Convert flat position back to model for loss computation
            model = posterior.vm.to_model(state.position)
            return {"Ln": loss_full_fn(model)}
        else:
            return {"Ln": jnp.nan}

    # Run all chains
    key, k_sample = jax.random.split(key)
    chain_keys = jax.random.split(k_sample, num_chains)

    # Start sampling timer
    sampling_start_time = time.time()

    traces = jax.vmap(
        lambda k, s: inference_loop_extended(
            k,
            mclmc.step,
            s,
            num_samples=num_samples,
            aux_fn=aux_fn,
            aux_every=1,
            work_per_step=work_per_step,
        )
    )(chain_keys, init_states)

    # Ensure sampling is finished
    jax.block_until_ready(traces)
    sampling_time = time.time() - sampling_start_time

    timings = {
        "adaptation": 0.0,
        "sampling": sampling_time,
        "total": sampling_time,
    }

    # Compute LLC from Ln if parameters provided
    if n_data is not None and beta is not None and L0 is not None and "Ln" in traces:
        traces["llc"] = float(n_data) * float(beta) * (traces["Ln"] - L0)

    work = {
        "n_full_loss": float(num_samples * num_chains),
        "n_minibatch_grads": 0.0,
        "sampler_flavour": "markov",
    }

    return SamplerRunResult(traces=traces, timings=timings, work=work)
