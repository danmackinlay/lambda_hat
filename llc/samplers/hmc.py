import time
import json
import os
import jax
import jax.numpy as jnp
import blackjax
import numpy as np
from .utils import drive_chains_batched, build_tiny_store

def run_hmc_batched(*, key, init_thetas, logpost_and_grad, draws, warmup, L,
                    eval_every, thin, Ln_full64_vmapped, tiny_store_fn, tuned_dir=None):
    """Window-adapt once on chain 0, vmapped tuned kernel; record acceptance & energy."""

    def logdensity(theta):
        val, _ = logpost_and_grad(theta)
        return val

    # Try to load cached tuned parameters
    step_size = None
    inv_mass = None

    if tuned_dir:
        tuned_file = os.path.join(tuned_dir, "hmc_tuned.json")
        if os.path.exists(tuned_file):
            try:
                with open(tuned_file, 'r') as f:
                    cached_params = json.load(f)
                step_size = cached_params.get("step_size")
                inv_mass_list = cached_params.get("inverse_mass_matrix")
                if inv_mass_list is not None:
                    inv_mass = jnp.array(inv_mass_list)
            except Exception:
                pass  # Fall back to adaptation

    if step_size is None or inv_mass is None:
        # Window adaptation on first chain
        key_tune, key_run = jax.random.split(key)
        x0 = init_thetas[0]

        t0_warmup = time.perf_counter()
        wa = blackjax.window_adaptation(blackjax.hmc, logdensity, num_integration_steps=L)
        (adapted_state, adaptation_info) = wa.run(key_tune, x0, num_steps=warmup)
        warmup_time = time.perf_counter() - t0_warmup

        step_size = adapted_state.parameters["step_size"]
        inv_mass = adapted_state.parameters["inverse_mass_matrix"]

        # Save tuned parameters
        if tuned_dir:
            tuned_file = os.path.join(tuned_dir, "hmc_tuned.json")
            tuned_params = {
                "step_size": float(step_size),
                "inverse_mass_matrix": inv_mass.tolist(),
            }
            os.makedirs(tuned_dir, exist_ok=True)
            with open(tuned_file, 'w') as f:
                json.dump(tuned_params, f, indent=2)
    else:
        key_run = key
        warmup_time = 0.0

    # Build tuned kernel and vmap
    hmc_kernel = blackjax.hmc(
        logdensity,
        step_size=step_size,
        inverse_mass_matrix=inv_mass,
        num_integration_steps=L,
    )
    init_states = jax.vmap(hmc_kernel.init)(init_thetas)
    step_vmapped = jax.jit(jax.vmap(jax.jit(hmc_kernel.step), in_axes=(0, 0)))

    # Info extractors for acceptance and energy
    def extract_acceptance(info):
        return getattr(info, "acceptance_rate", jnp.zeros(()))

    def extract_energy(info):
        return getattr(info, "energy", jnp.zeros(()))

    info_extractors = {
        "accept": extract_acceptance,
        "energy": extract_energy,
    }

    position_fn = lambda st: st.position

    # Generate per-step keys
    C = init_thetas.shape[0]
    rng_keys = jax.random.split(key_run, draws * C).reshape(draws, C, -1)

    # Run batched HMC
    t0 = time.perf_counter()
    result = drive_chains_batched(
        rng_keys=rng_keys,
        init_state=init_states,
        step_fn_vmapped=step_vmapped,
        n_steps=draws,
        warmup=0,  # We already did warmup during adaptation
        eval_every=eval_every,
        thin=thin,
        position_fn=position_fn,
        Ln_eval_f64_vmapped=Ln_full64_vmapped,
        tiny_store_fn=tiny_store_fn,
        info_extractors=info_extractors,
    )
    total_time = time.perf_counter() - t0

    # Convert results
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]
    theta_thin = np.asarray(result.kept)

    # Extract acceptance and energy
    acceptance = np.asarray(result.extras.get("accept", np.zeros((C, result.L_hist.shape[1]))))
    energy = np.asarray(result.extras.get("energy", np.zeros((C, result.L_hist.shape[1]))))

    # Timing and work
    M = result.L_hist.shape[1]

    timings = {
        "warmup": float(warmup_time),
        "sampling": float(total_time),
    }

    work = {
        "n_minibatch_grads": 0,  # HMC doesn't use minibatches
        "n_full_loss": int(C * M),
        "n_gradients": int(C * draws * (L + 1)),  # Leapfrog steps + 1 for initial
    }

    return L_histories, theta_thin, acceptance, energy, timings, work