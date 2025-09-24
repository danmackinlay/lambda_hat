import time
import jax
import jax.numpy as jnp
import blackjax
import numpy as np
from .utils import DiagPrecondState, precond_update, drive_chains_batched, build_tiny_store

def run_sgld_batched(*, key, init_thetas, grad_logpost_minibatch, X, Y, n, step_size,
                     num_steps, warmup, batch_size, eval_every, thin, Ln_full64_vmapped,
                     tiny_store_fn, precond_mode="none", beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
    """Return (Ln_histories list, kept ndarray, timings/work dicts)."""

    # Prepare SGLD step function
    if precond_mode == "none":
        # Plain SGLD using BlackJAX
        sgld = blackjax.sgld(grad_logpost_minibatch)
        step_single = jax.jit(sgld.step)

        def step_one(k, theta):
            k_noise, k_batch = jax.random.split(k)
            idx = jax.random.randint(k_batch, (batch_size,), 0, n)
            new_theta = step_single(k_noise, theta, (X[idx], Y[idx]), step_size)
            return new_theta.astype(theta.dtype), None

        step_vmapped = jax.jit(jax.vmap(step_one, in_axes=(0, 0)))
        position_fn = lambda s: s
        init_states = init_thetas

    else:
        # Preconditioned SGLD
        C, d = init_thetas.shape
        precond_state = DiagPrecondState(
            m=jnp.zeros((C, d), dtype=init_thetas.dtype),
            v=jnp.zeros((C, d), dtype=init_thetas.dtype),
            t=jnp.zeros((C,), dtype=init_thetas.dtype),
        )

        @jax.jit
        def precond_step_single(key, state):
            theta, precond_state = state
            k_noise, k_batch = jax.random.split(key)
            idx = jax.random.randint(k_batch, (batch_size,), 0, n)
            g = grad_logpost_minibatch(theta, (X[idx], Y[idx]))

            # Use generic preconditioning update
            inv_sqrt, new_precond_state, drift_m = precond_update(
                g, precond_state, precond_mode, beta1, beta2, eps, bias_correction
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
        init_states = (init_thetas, precond_state)

    # Generate per-step keys for all steps and chains
    C = init_thetas.shape[0]
    rng_keys = jax.random.split(key, num_steps * C).reshape(num_steps, C, -1)

    # Run batched sampler
    t0 = time.perf_counter()
    result = drive_chains_batched(
        rng_keys=rng_keys,
        init_state=init_states,
        step_fn_vmapped=step_vmapped,
        n_steps=num_steps,
        warmup=warmup,
        eval_every=eval_every,
        thin=thin,
        position_fn=position_fn,
        Ln_eval_f64_vmapped=Ln_full64_vmapped,
        tiny_store_fn=tiny_store_fn,
        info_extractors={},
    )
    total_time = time.perf_counter() - t0

    # Convert result to expected format
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]
    theta_thin = np.asarray(result.kept)

    # Timing and work statistics
    M = result.L_hist.shape[1]  # number of evaluations

    # Estimate time spent on full evaluations vs sampling
    eval_time_estimate = M * 0.001  # rough estimate
    sampling_time = max(0.0, total_time - eval_time_estimate)

    timings = {
        "warmup": float(result.warmup_time_seconds),
        "sampling": float(sampling_time),
    }

    work = {
        "n_minibatch_grads": int(C * num_steps),
        "n_full_loss": int(C * M),
    }

    return L_histories, theta_thin, None, None, timings, work