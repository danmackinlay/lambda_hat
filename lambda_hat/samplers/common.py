# lambda_hat/samplers/common.py
"""Common utilities for all samplers"""

import jax
import jax.numpy as jnp


def inference_loop_extended(
    rng_key, kernel, initial_state, num_samples, aux_fn, aux_every=1, work_per_step=1.0
):
    """
    Efficient inference loop using jax.lax.scan that records Ln, diagnostics,
    and cumulative work (FGEs).

    Args:
        aux_fn: A function that takes the state and returns a dict {"Ln": val}.
        aux_every: Thinning factor for the trace.
        work_per_step: The amount of work (in FGEs) performed by one call to the kernel.
    """
    # Seed the aux cache once from the initial state (outside of scan)
    init_aux = aux_fn(initial_state)

    @jax.jit
    def one_step(carry, rng_key):
        state, cumulative_work, t, last_aux = carry
        new_state, info = kernel(rng_key, state)

        # Update cumulative work (ensure dtype consistency)
        current_work = jnp.asarray(work_per_step, dtype=cumulative_work.dtype)
        new_cumulative_work = cumulative_work + current_work

        # Compute Aux only every aux_every steps; otherwise reuse cached value
        def do_aux(_):
            return aux_fn(new_state)

        aux_data = jax.lax.cond(
            ((t + 1) % aux_every) == 0, do_aux, lambda _: last_aux, operand=None
        )

        # Combine data for trace
        trace_data = aux_data.copy()
        trace_data["cumulative_fge"] = new_cumulative_work

        # Extract diagnostics robustly
        trace_data["acceptance_rate"] = getattr(
            info, "acceptance_rate", getattr(info, "acceptance_probability", jnp.nan)
        )
        trace_data["energy"] = getattr(info, "energy", jnp.nan)
        # Standardize divergence key (handle both 'is_divergent' and 'diverging')
        trace_data["is_divergent"] = getattr(
            info, "is_divergent", getattr(info, "diverging", False)
        )

        return (new_state, new_cumulative_work, t + 1, aux_data), trace_data

    keys = jax.random.split(rng_key, num_samples)
    # Initialize work accumulation with high precision (float64)
    initial_work = jnp.array(0.0, dtype=jnp.float64)

    # Run the scan
    # The carry tuple is (state, cumulative_work, t, last_aux)
    _, trace = jax.lax.scan(
        one_step, (initial_state, initial_work, jnp.array(0, jnp.int32), init_aux), keys
    )

    # Apply thinning AFTER the scan (efficient JAX pattern)
    if aux_every > 1:
        trace = jax.tree.map(lambda x: x[::aux_every], trace)

    return trace
