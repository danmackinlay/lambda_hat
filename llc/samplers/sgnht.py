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
        theta, p, alpha = state
        k_noise, k_batch = jax.random.split(key)

        # minibatch
        idx = jax.random.randint(k_batch, (batch_size,), 0, n)
        g = grad_logpost_minibatch(theta, (X[idx], Y[idx]))  # ∇ log π(θ)

        # momentum: p ← p + ε*(g − α p) + √(2 α ε) ξ
        eps = jnp.asarray(step_size, dtype=theta.dtype)
        # for stability, only clip the α used in noise std, not alpha itself
        alpha_noise = jnp.clip(alpha, 1e-8, None)
        noise = jax.random.normal(k_noise, theta.shape) * jnp.sqrt(2.0 * alpha_noise * eps)
        p_new = p + eps * (g - alpha[:, None] * p) + noise

        # position: θ ← θ + ε p
        theta_new = theta + eps * p_new

        # thermostat: α ← α + ε*(||p||^2/d − 1)
        d_param = theta.shape[-1]
        kinetic_mean = jnp.mean(p_new * p_new, axis=-1)  # == ||p||^2/d
        alpha_new = alpha + eps * (kinetic_mean - 1.0)

        new_state = (theta_new, p_new, alpha_new)

        extras = {
            "alpha_mean": jnp.mean(alpha_new),
            "momentum_norm": jnp.linalg.norm(p_new)
        }
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