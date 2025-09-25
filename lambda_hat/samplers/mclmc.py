import time
import json
import os
import jax
import jax.numpy as jnp
import blackjax
import numpy as np
from .utils import drive_chains_batched, build_tiny_store

def run_mclmc_batched(*, key, init_thetas, logdensity_fn, draws, eval_every, thin,
                      Ln_full64_vmapped, tiny_store_fn,
                      num_steps, frac_tune1, frac_tune2, frac_tune3,
                      diagonal_preconditioning, desired_energy_var, trust_in_estimate,
                      num_effective_samples, integrator_name, tuned_dir=None):
    """Use fractional tuner (1.2.5) to get (L, step_size); vmapped sampling; record energy change."""

    # Try to load cached tuned parameters
    L_tuned = None
    eps_tuned = None

    if tuned_dir:
        tuned_file = os.path.join(tuned_dir, "mclmc_tuned.json")
        if os.path.exists(tuned_file):
            try:
                with open(tuned_file, 'r') as f:
                    cached_params = json.load(f)
                L_tuned = cached_params.get("L")
                eps_tuned = cached_params.get("step_size")
            except Exception:
                pass  # Fall back to tuning

    if L_tuned is None or eps_tuned is None:
        # Perform tuning
        key_tune, key_run = jax.random.split(key)
        x0 = init_thetas[0]

        # Get integrator
        integrator = getattr(blackjax.mcmc.integrators, integrator_name)

        # Build MCLMC kernel factory
        key_init, key_tune_actual = jax.random.split(key_tune)
        init_state = blackjax.mclmc.init(x0, logdensity_fn, rng_key=key_init)
        kernel_factory = lambda inv_mass: blackjax.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=inv_mass,
            integrator=integrator,
        )

        # Validate tuning phases
        phase_steps = (
            int(num_steps * float(frac_tune1)),
            int(num_steps * float(frac_tune2)),
            int(num_steps * float(frac_tune3)),
        )
        fracs = [frac_tune1, frac_tune2, frac_tune3]
        bad = [i+1 for i, n in enumerate(phase_steps) if (n == 0 and fracs[i] > 0)]
        if bad:
            raise ValueError(
                f"MCLMC config invalid: num_steps={num_steps} too small for nonzero frac_tune{bad}."
                " Increase num_steps or set those fractions to 0.0"
            )

        t0_warmup = time.perf_counter()
        # Use BlackJAX 1.2.5 fractional API
        state_tuned, tuned_params, _info = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel_factory,
            num_steps=num_steps,
            state=init_state,
            rng_key=key_tune_actual,
            frac_tune1=frac_tune1,
            frac_tune2=frac_tune2,
            frac_tune3=frac_tune3,
            desired_energy_var=desired_energy_var,
            trust_in_estimate=trust_in_estimate,
            num_effective_samples=num_effective_samples,
            diagonal_preconditioning=bool(diagonal_preconditioning),
        )
        warmup_time = time.perf_counter() - t0_warmup

        # Extract tuned parameters
        L_tuned = int(getattr(tuned_params, "L"))
        eps_tuned = float(getattr(tuned_params, "step_size"))

        if L_tuned <= 0:
            raise ValueError(f"MCLMC tuner returned invalid L={L_tuned} (must be > 0)")
        if eps_tuned <= 0:
            raise ValueError(f"MCLMC tuner returned invalid step_size={eps_tuned} (must be > 0)")

        # Save tuned parameters
        if tuned_dir:
            tuned_file = os.path.join(tuned_dir, "mclmc_tuned.json")
            tuned_payload = {
                "L": int(L_tuned),
                "step_size": float(eps_tuned),
                "desired_energy_var": float(desired_energy_var),
                "integrator_name": integrator_name,
                "diagonal_preconditioning": bool(diagonal_preconditioning),
            }
            os.makedirs(tuned_dir, exist_ok=True)
            with open(tuned_file, 'w') as f:
                json.dump(tuned_payload, f, indent=2)
    else:
        key_run = key
        warmup_time = 0.0

    # Build final MCLMC algorithm for sampling
    integrator = getattr(blackjax.mcmc.integrators, integrator_name)
    final_mclmc_algo = blackjax.mclmc(
        logdensity_fn, L=int(L_tuned), step_size=float(eps_tuned), integrator=integrator
    )
    step_single = jax.jit(final_mclmc_algo.step)

    # Initialize states for all chains
    chains = init_thetas.shape[0]
    keys_run = jax.random.split(key_run, chains)
    init_states = jax.vmap(lambda theta, k: final_mclmc_algo.init(theta, rng_key=k))(
        init_thetas, keys_run
    )

    # Info extractor for energy
    def extract_energy(info):
        return getattr(info, "energy_change", jnp.zeros(()))

    info_extractors = {"energy": extract_energy}
    position_fn = lambda st: st.position
    step_vmapped = jax.jit(jax.vmap(step_single, in_axes=(0, 0)))

    # Generate per-step keys
    rng_keys = jax.random.split(key_run, draws * chains).reshape(draws, chains, -1)

    # Run batched MCLMC
    t0 = time.perf_counter()
    result = drive_chains_batched(
        rng_keys=rng_keys,
        init_state=init_states,
        step_fn_vmapped=step_vmapped,
        n_steps=draws,
        warmup=0,  # We already did warmup during tuning
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

    # Extract energy
    energy = np.asarray(result.extras.get("energy", np.zeros((chains, result.L_hist.shape[1]))))

    # Timing and work
    M = result.L_hist.shape[1]

    timings = {
        "warmup": float(warmup_time),
        "sampling": float(total_time),
    }

    work = {
        "n_minibatch_grads": 0,  # MCLMC doesn't use minibatches
        "n_full_loss": int(chains * M),
        "n_gradients": int(chains * draws),  # One gradient per MCLMC step
    }

    return L_histories, theta_thin, None, energy, timings, work