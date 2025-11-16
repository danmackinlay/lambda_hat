# lambda_hat/samplers/hmc.py
"""HMC sampler with window adaptation - FLAT INTERFACE ONLY"""

import time
from typing import Optional

import blackjax
import jax
import jax.numpy as jnp

from lambda_hat.posterior import Posterior
from lambda_hat.samplers.common import inference_loop_extended
from lambda_hat.types import SamplerRunResult

Array = jnp.ndarray


def run_hmc(
    key: jax.random.PRNGKey,
    posterior: Posterior,
    num_samples: int,
    num_chains: int,
    step_size: float = 0.01,
    num_integration_steps: int = 10,
    adaptation_steps: int = 1000,
    target_acceptance: float = 0.8,
    loss_full_fn: Optional[callable] = None,
    n_data: Optional[int] = None,
    beta: Optional[float] = None,
    L0: Optional[float] = None,
) -> SamplerRunResult:
    """Run HMC with optional adaptation - FLAT INTERFACE ONLY

    Args:
        key: JRNG key
        posterior: Posterior with flat-space log density and gradient
        num_samples: Number of samples per chain
        num_chains: Number of chains to run in parallel
        step_size: Initial step size
        num_integration_steps: Number of leapfrog steps
        adaptation_steps: Number of adaptation steps (0 to disable)
        target_acceptance: Target acceptance rate for adaptation
        loss_full_fn: Optional function to compute loss for recording
        n_data: Number of data points (for LLC computation)
        beta: Temperature parameter (for LLC computation)
        L0: Reference loss (for LLC computation)

    Returns:
        SamplerRunResult with traces and timing information
    """
    # Setup keys
    key, init_key, sample_key = jax.random.split(key, 3)
    init_keys = jax.random.split(init_key, num_chains)

    # Get flat initial parameters and dimension
    flat0 = posterior.flat0
    D = posterior.vm.size
    dtype = posterior.vm.dtype

    # 1) Jitter initial positions in FLAT SPACE (no pytrees!)
    def jitter_flat(k: jax.random.PRNGKey, flat: Array) -> Array:
        """Jitter flat parameters with small noise"""
        noise = jax.random.normal(k, flat.shape, dtype=dtype)
        return flat + 0.01 * noise

    # Vmap over chains to create diverse initial positions
    init_positions = jax.vmap(jitter_flat, in_axes=(0, None))(init_keys, flat0)

    # 2) Warmup on one chain (Adaptation)
    adaptation_start_time = time.time()

    # Initialize defaults
    step_size_adapted = step_size
    inv_mass = jnp.ones(D, dtype=dtype)

    if adaptation_steps > 0:
        wa = blackjax.window_adaptation(
            blackjax.hmc,
            posterior.logpost_flat,  # Use flat-space log density
            target_acceptance_rate=target_acceptance,
            num_integration_steps=num_integration_steps,
        )

        # Use first chain's initial position for adaptation
        one_pos = init_positions[0]

        # Run adaptation
        warmup_result = wa.run(
            jax.random.split(key)[0],
            one_pos,
            num_steps=adaptation_steps,
        )
        # Ensure adaptation is finished before stopping the timer
        jax.block_until_ready(warmup_result)

        # Unpack the result - BlackJAX returns ((state, params), info)
        (final_state, params), _ = warmup_result

        # Extract adaptation results from BlackJAX 1.2.5 expected format
        if isinstance(params, dict):
            step_size_adapted = params.get("step_size", step_size_adapted)
            inv_mass_adapted = params.get("inverse_mass_matrix", inv_mass)
            if hasattr(inv_mass_adapted, "ndim") and inv_mass_adapted.ndim in [1, 2]:
                inv_mass = inv_mass_adapted
        else:
            # Handle tuple format (step_size, inv_mass)
            step_size_adapted, inv_mass = params

    # Stop adaptation timer
    adaptation_time = time.time() - adaptation_start_time

    # 3) Build kernel and init all chains with flat vectors
    hmc = blackjax.hmc(
        posterior.logpost_flat,  # Use flat-space log density
        step_size=step_size_adapted,
        num_integration_steps=num_integration_steps,
        inverse_mass_matrix=inv_mass,
    )

    # Initialize states for all chains (BlackJAX works with flat arrays)
    init_states = jax.vmap(hmc.init)(init_positions)

    # 4) Drive all chains using the efficient loop
    sample_keys = jax.random.split(sample_key, num_chains)

    # Define aux_fn for recording loss (state.position is flat array)
    def aux_fn(state):
        if loss_full_fn is not None:
            # Convert flat position back to model for loss computation
            model = posterior.vm.to_model(state.position)
            return {"Ln": loss_full_fn(model)}
        else:
            return {"Ln": jnp.nan}

    # Calculate work per step (FGEs): HMC uses full gradients
    work_per_step = float(num_integration_steps)

    # Start sampling timer
    sampling_start_time = time.time()

    # Use vmap with the optimized inference loop
    traces = jax.vmap(
        lambda k, s: inference_loop_extended(
            k,
            hmc.step,
            s,
            num_samples=num_samples,
            aux_fn=aux_fn,
            aux_every=1,  # HMC records every step
            work_per_step=work_per_step,
        )
    )(sample_keys, init_states)

    # Ensure sampling is finished before stopping the timer
    jax.block_until_ready(traces)
    sampling_time = time.time() - sampling_start_time

    timings = {
        "adaptation": adaptation_time,
        "sampling": sampling_time,
        "total": adaptation_time + sampling_time,
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
