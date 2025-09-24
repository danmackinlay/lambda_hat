import time
import jax
import jax.numpy as jnp
import numpy as np
from .utils import drive_chains_batched

def run_sgnht_batched(*, key, init_thetas, grad_logpost_minibatch, X, Y, n, step_size,
                      num_steps, warmup, batch_size, eval_every, thin, Ln_full64_vmapped,
                      tiny_store_fn, alpha0=0.01):
    """SGNHT (Stochastic Gradient Nose-Hoover Thermostat) sampler.

    Implements Algorithm 5 from App. D.3 with thermostat variable and minibatch gradients.
    Returns (Ln_histories list, kept ndarray, acceptance, energy, timings/work dicts).
    """

    C, d = init_thetas.shape

    # SGNHT state: (theta, momentum p, thermostat alpha)
    init_momentum = jnp.zeros_like(init_thetas)
    init_alpha = jnp.full((C,), alpha0, dtype=init_thetas.dtype)
    init_states = (init_thetas, init_momentum, init_alpha)

    @jax.jit
    def sgnht_step_single(key, state):
        """Single SGNHT step for one chain following Algorithm 5."""
        theta, p, alpha = state

        k_noise, k_batch = jax.random.split(key)
        # Sample minibatch
        idx = jax.random.randint(k_batch, (batch_size,), 0, n)

        # Compute gradient on minibatch
        g = grad_logpost_minibatch(theta, (X[idx], Y[idx]))

        # Ensure g is a JAX array (in case it's a tuple or other structure)
        if not isinstance(g, jnp.ndarray):
            g = jnp.asarray(g)

        # SGNHT update equations (Algorithm 5)
        # Δp_t = -ε/2 * (γ(w_t - w_0) + β̃g_t) - α_t * p_t + √(2α_t * ε) * η_t

        # For local posterior, we assume w_0 = 0 and γ, β̃ are absorbed in gradient
        # So the drift term is just the gradient
        drift_term = -0.5 * step_size * g
        friction_term = -alpha * p
        noise_term = jnp.sqrt(2 * alpha * step_size) * jax.random.normal(k_noise, theta.shape)

        # Momentum update
        p_new = p + drift_term + friction_term + noise_term

        # Position update: w_{t+1} = w_t + p_{t+1}
        theta_new = theta + p_new

        # Thermostat update: α_{t+1} = α_t + ||p_{t+1}||/d - ε
        alpha_new = alpha + jnp.linalg.norm(p_new) / d - step_size

        new_state = (
            theta_new.astype(theta.dtype),
            p_new.astype(theta.dtype),
            alpha_new.astype(theta.dtype)
        )

        # Return extras for diagnostics (optional)
        extras = {"alpha_mean": jnp.mean(alpha_new), "momentum_norm": jnp.linalg.norm(p_new)}

        return new_state, extras

    step_vmapped = jax.jit(jax.vmap(sgnht_step_single, in_axes=(0, 0)))
    position_fn = lambda s: s[0]  # extract theta from (theta, p, alpha)

    # Generate per-step keys for all steps and chains
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
        info_extractors={
            "alpha_mean": lambda extras: extras["alpha_mean"],
            "momentum_norm": lambda extras: extras["momentum_norm"],
        },
    )
    total_time = time.perf_counter() - t0

    # Convert result to expected format
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]
    theta_thin = np.asarray(result.kept)

    # SGNHT doesn't have acceptance rate like HMC/MCLMC
    acceptance = None

    # Extract energy-like diagnostics from extras if available
    energy = None
    if hasattr(result, 'extras') and result.extras:
        energy = {
            "alpha_mean": np.asarray(result.extras.get("alpha_mean", [])),
            "momentum_norm": np.asarray(result.extras.get("momentum_norm", [])),
        }

    # Timing and work statistics
    M = result.L_hist.shape[1]  # number of evaluations

    timings = {
        "warmup": float(result.warmup_time_seconds),
        "sampling": float(total_time),
    }

    work = {
        "n_minibatch_grads": int(C * num_steps),
        "n_full_loss": int(C * M),
    }

    return L_histories, theta_thin, acceptance, energy, timings, work