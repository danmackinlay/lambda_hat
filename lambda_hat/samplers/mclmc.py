# lambda_hat/samplers/mclmc.py
"""MCLMC sampler (Microcanonical Langevin Monte Carlo)"""

import time
from typing import Callable, Optional

import blackjax
import blackjax.mcmc.integrators as bj_integrators
import jax
import jax.numpy as jnp

from lambda_hat.samplers.common import inference_loop_extended
from lambda_hat.types import SamplerRunResult


def run_mclmc(
    rng_key,
    logdensity_fn,
    initial_params,
    num_samples,
    num_chains,
    config,
    loss_full_fn: Optional[Callable] = None,
    n_data: Optional[int] = None,
    beta: Optional[float] = None,
    L0: Optional[float] = None,
) -> SamplerRunResult:
    # -- flatten params once for MCLMC's state vector
    leaves, treedef = jax.tree_util.tree_flatten(initial_params)
    sizes = [x.size for x in leaves]
    shapes = [x.shape for x in leaves]
    theta0 = jnp.concatenate([x.reshape(-1) for x in leaves])  # (d,)

    def flatten(params):
        ls = jax.tree_util.tree_leaves(params)
        return jnp.concatenate([x.reshape(-1) for x in ls])

    def unflatten(theta):
        out = []
        i = 0
        for shp, sz in zip(shapes, sizes):
            out.append(theta[i : i + sz].reshape(shp))
            i += sz
        return jax.tree_util.tree_unflatten(treedef, out)

    if loss_full_fn is None:
        raise ValueError("loss_full_fn must be provided for Ln recording in MCLMC.")

    # Create flat loss function (REPLACE existing definition to ensure JIT)
    # This relies on 'unflatten' being defined earlier in the function scope.
    def loss_full_flat_raw(theta_flat):
        params = unflatten(theta_flat)
        return loss_full_fn(params)

    # JIT this function for use inside the scan loop
    loss_full_flat = jax.jit(loss_full_flat_raw)

    # diversify starting points
    key, k_init = jax.random.split(rng_key)
    init_thetas = theta0 + 0.01 * jax.random.normal(
        k_init, (num_chains, theta0.size), dtype=theta0.dtype
    )

    # pick integrator
    integrators = {
        "isokinetic_mclachlan": bj_integrators.isokinetic_mclachlan,
        "isokinetic_velocity_verlet": bj_integrators.isokinetic_velocity_verlet,
    }
    integrator = integrators[config.integrator]

    # Create flattened logdensity function
    def logdensity_flat(theta):
        params = unflatten(theta)
        return logdensity_fn(params)

    # BlackJAX MCLMC constructor does not take inverse_mass_matrix.
    # It uses sqrt_diag_cov internally.
    mclmc = blackjax.mclmc(
        logdensity_fn=logdensity_flat,
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
    # Use jnp.ceil to ensure it's an integer count of steps
    num_integration_steps = jnp.ceil(config.L / config.step_size)
    work_per_step = float(num_integration_steps)

    # Define the aux function for the loop
    # MCLMC state has 'position' attribute which holds the flattened vector.
    def aux_fn(st):
        theta_flat = st.position
        return {"Ln": loss_full_flat(theta_flat)}

    # run all chains
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
            work_per_step=work_per_step,  # Pass work
        )
    )(chain_keys, init_states)

    # Ensure sampling is finished
    jax.block_until_ready(traces)
    sampling_time = time.time() - sampling_start_time

    timings = {
        # Note: This implementation assumes MCLMC adaptation is not used or timed separately.
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
