# llc/runners.py
"""
Batched runners only. Each runner returns a `SamplerResult` with: ragged `Ln_histories`,
thinned `theta`, `acceptance`/`energy` (if any), and **filled** `timings`/`work`.
Timing = wall-clock − eval-time estimate − warmup; Work = deterministic counts.
Use these for ESS/sec and WNV in metrics.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
import numpy as np
from jax import numpy as jnp

from .samplers.adapters import (
    sgld_spec,
    sghmc_spec,
    hmc_spec,
    mclmc_spec,
)
from .samplers.base import build_tiny_store, run_sampler_spec, DiagPrecondState
from .types import SamplerResult

if TYPE_CHECKING:
    pass


def tic():
    """Start timing"""
    return time.perf_counter()


def toc(t0):
    """End timing and return elapsed seconds"""
    return time.perf_counter() - t0


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
    """Batched SGLD runner using SamplerSpec + run_sampler_spec"""
    tiny_store = build_tiny_store(diag_dims, Rproj)

    # Create SGLD SamplerSpec
    spec = sgld_spec(
        grad_logpost_minibatch=grad_logpost_minibatch,
        X=X,
        Y=Y,
        n_data=n,
        step_size=step_size,
        batch_size=batch_size,
        precond_mode=precond_mode,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        bias_correction=bias_correction,
    )

    # Wrap initial state for preconditioning if needed
    C, d = init_thetas.shape
    if precond_mode != "none":
        precond_state = DiagPrecondState(
            m=jnp.zeros((C, d), dtype=init_thetas.dtype),
            v=jnp.zeros((C, d), dtype=init_thetas.dtype),
            t=jnp.zeros((C,), dtype=init_thetas.dtype),
        )
        init_states = (init_thetas, precond_state)
        # Sanity check for future-proofing
        if not isinstance(init_states, tuple):
            raise RuntimeError("SGLD preconditioning requires tuple state: (theta, DiagPrecondState).")
    else:
        init_states = init_thetas

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key,
        init_states=init_states,
        n_steps=num_steps,
        warmup=warmup,
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64_vmapped=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time

    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(
        total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M
    )

    # Convert back to the expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    means = np.asarray(result.mean_L)  # (C,)
    vars_ = np.asarray(result.var_L)  # (C,)
    ns = np.asarray(result.n_L)  # (C,)
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]

    # Build timing and work dictionaries
    timings = {
        "warmup": float(result.warmup_time_seconds),
        "sampling": float(sampling_time),
    }

    # SGLD work computation
    chains = init_thetas.shape[0]
    work = {
        "n_minibatch_grads": int(chains * num_steps),
        "n_full_loss": int(chains * M),
    }

    return SamplerResult(
        Ln_histories=L_histories,
        theta_thin=kept_stacked,
        acceptance=None,
        energy=None,
        timings=timings,
        work=work,
    )


def run_sghmc_online_batched(
    key,
    init_thetas,
    grad_logpost_minibatch,
    X,
    Y,
    n_data,
    step_size,
    temperature,
    draws,
    eval_every,
    thin,
    batch_size,
    Ln_full64,
    diag_dims=None,
    Rproj=None,
    # Ignored parameters for compatibility
    use_tqdm=None,
    progress_update_every=None,
    stats=None,
):
    """Batched SGHMC runner using SamplerSpec + run_sampler_spec"""
    tiny_store = build_tiny_store(diag_dims, Rproj)

    # Create SGHMC SamplerSpec
    spec = sghmc_spec(
        grad_logpost_minibatch=grad_logpost_minibatch,
        X=X,
        Y=Y,
        n_data=n_data,
        step_size=step_size,
        temperature=temperature,
        batch_size=batch_size,
    )

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key,
        init_states=init_thetas,
        n_steps=draws,
        warmup=0,  # SGHMC has no warmup
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64_vmapped=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time

    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(
        total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M
    )

    # Convert back to the expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]

    # Build timing and work dictionaries
    timings = {
        "warmup": float(result.warmup_time_seconds),
        "sampling": float(sampling_time),
    }

    # SGHMC work computation (similar to SGLD)
    chains = init_thetas.shape[0]
    work = {
        "n_minibatch_grads": int(chains * draws),
        "n_full_loss": int(chains * M),
    }

    return SamplerResult(
        Ln_histories=L_histories,
        theta_thin=kept_stacked,
        acceptance=None,
        energy=None,
        timings=timings,
        work=work,
    )


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
    tuned_path=None,
    # Ignored parameters for compatibility
    use_tqdm=None,
    progress_update_every=None,
    stats=None,
):
    """Batched HMC runner using SamplerSpec + run_sampler_spec"""
    import blackjax
    import jax
    from jax import numpy as jnp

    tiny_store = build_tiny_store(diag_dims, Rproj)

    def logdensity(theta):
        val, _ = logpost_and_grad(theta)
        return val

    # ---- Try to load tuned parameters, or perform window adaptation ----
    step_size = None
    inv_mass = None

    if tuned_path:
        from llc.artifacts import load_tuned_params
        cached_params = load_tuned_params(tuned_path, "hmc")
        if cached_params:
            step_size = cached_params.get("step_size")
            inv_mass_list = cached_params.get("inverse_mass_matrix")
            if inv_mass_list is not None:
                inv_mass = jnp.array(inv_mass_list)

    if step_size is None or inv_mass is None:
        # ---- Window adaptation (single chain) ----
        # Use the first chain's init as the adaptation reference; cheap and robust.
        key_tune, key_run = jax.random.split(key)
        x0 = init_thetas[0]
        # Target acceptance ~ 0.8 is a common default
        wa = blackjax.window_adaptation(blackjax.hmc, logdensity, num_integration_steps=L)
        (adapted_state, adaptation_info) = wa.run(key_tune, x0, num_steps=warmup)
        # Extract tuned parameters from adapted_state.parameters
        step_size = adapted_state.parameters["step_size"]
        inv_mass = adapted_state.parameters["inverse_mass_matrix"]

        # Save tuned parameters if tuned_path is provided
        if tuned_path:
            from llc.artifacts import save_tuned_params
            tuned_params = {
                "step_size": float(step_size),
                "inverse_mass_matrix": inv_mass,
            }
            save_tuned_params(tuned_path, "hmc", tuned_params)
    else:
        # Using cached parameters, split key for run only
        key_run = key

    # ---- Build the tuned kernel; then vmap across chains ----
    hmc_kernel = blackjax.hmc(
        logdensity,
        step_size=step_size,
        inverse_mass_matrix=inv_mass,
        num_integration_steps=L,
    )
    init_states = jax.vmap(hmc_kernel.init)(init_thetas)
    step_vmapped = jax.jit(jax.vmap(jax.jit(hmc_kernel.step), in_axes=(0, 0)))

    # Create SamplerSpec with proper HMC components
    from llc.samplers.base import SamplerSpec
    spec = SamplerSpec(
        name="hmc",
        step_vmapped=lambda keys, states: step_vmapped(keys, states),
        position_fn=lambda st: st.position,
        info_extractors={
            "accept": lambda info: info.acceptance_rate,
            "energy": lambda info: info.energy,
        },
        grads_per_step=L + 1,
    )

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key_run,
        init_states=init_states,  # Now passing HMCState objects
        n_steps=draws,
        warmup=0,  # Warmup already done during adaptation
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64_vmapped=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time

    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(
        total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M
    )

    # Convert back to expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    means = np.asarray(result.mean_L)  # (C,)
    vars_ = np.asarray(result.var_L)  # (C,)
    ns = np.asarray(result.n_L)  # (C,)
    acc = result.extras.get("accept", jnp.zeros_like(result.L_hist))
    acc_list = [np.asarray(acc[c]) for c in range(acc.shape[0])]
    energy = result.extras.get("energy", jnp.zeros_like(result.L_hist))
    energy_list = [np.asarray(energy[c]) for c in range(energy.shape[0])]
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]

    # Build timing and work dictionaries
    timings = {
        "warmup": float(result.warmup_time_seconds),
        "sampling": float(sampling_time),
    }

    # HMC work computation
    chains = init_thetas.shape[0]
    work = {
        "n_leapfrog_grads": int(chains * draws * (L + 1)),
        "n_full_loss": int(chains * M),
        "n_warmup_leapfrog_grads": int(result.warmup_grads),
    }

    return SamplerResult(
        Ln_histories=L_histories,
        theta_thin=kept_stacked,
        acceptance=acc_list,
        energy=energy_list,
        timings=timings,
        work=work,
    )


def run_mclmc_online_batched(
    key,
    init_thetas,
    logdensity_fn,
    draws,
    eval_every,
    thin,
    Ln_full64,
    # BlackJAX 1.2.5 fractional API parameters
    num_steps=2000,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    diagonal_preconditioning=False,
    desired_energy_var=5e-4,
    trust_in_estimate=1.0,
    num_effective_samples=150.0,
    integrator_name="isokinetic_mclachlan",
    diag_dims=None,
    Rproj=None,
    tuned_path=None,
    # Ignored parameters for compatibility
    use_tqdm=None,
    progress_update_every=None,
    stats=None,
):
    """Batched MCLMC runner using SamplerSpec + run_sampler_spec"""
    import blackjax
    import jax
    from jax import numpy as jnp

    tiny_store = build_tiny_store(diag_dims, Rproj)

    # ---- Try to load tuned parameters, or perform tuning ----
    L_tuned = None
    eps_tuned = None

    if tuned_path:
        from llc.artifacts import load_tuned_params
        cached_params = load_tuned_params(tuned_path, "mclmc")
        if cached_params:
            L_tuned = cached_params.get("L")
            eps_tuned = cached_params.get("step_size")

    if L_tuned is None or eps_tuned is None:
        # ---- Tune (L, step_size) once, then share across chains ----
        key_tune, key_run = jax.random.split(key)
        x0 = init_thetas[0]
        integrator = getattr(blackjax.mcmc.integrators, integrator_name)

        # Build integrator and kernel factory expected by the tuner.
        # The tuner expects a function: (inverse_mass_matrix) -> kernel(rng_key, state, L, step_size)
        key_init, key_tune_actual = jax.random.split(key_tune)
        init_state = blackjax.mclmc.init(x0, logdensity_fn, rng_key=key_init)  # IntegratorState
        kernel_factory = lambda inv_mass: blackjax.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=inv_mass,
            integrator=integrator,
        )

        # Note: BlackJAX 1.2.5 requires pre-built kernel, not logdensity_fn parameter

        # Fail fast if someone reintroduces unsupported kwargs here
        import inspect
        _sig = inspect.signature(blackjax.mclmc_find_L_and_step_size)
        assert "integrator" not in _sig.parameters, "Do not pass integrator to mclmc_find_L_and_step_size"

        # Fail fast if any tuning phase would be empty; BlackJAX will choke later.
        # At least 1 step in each of the 3 tuning phases if its fraction > 0.
        phase_steps = (
            int(num_steps * float(frac_tune1)),
            int(num_steps * float(frac_tune2)),
            int(num_steps * float(frac_tune3)),
        )
        bad = [i+1 for i, n in enumerate(phase_steps) if (n == 0 and [frac_tune1, frac_tune2, frac_tune3][i] > 0)]
        if bad:
            raise SystemExit(
                f"MCLMC config invalid: num_steps={num_steps} too small for nonzero frac_tune{bad}."
                " Increase num_steps or set those fractions to 0.0"
            )
        if num_steps <= 0:
            raise SystemExit("MCLMC config invalid: num_steps must be > 0.")

        # Use BlackJAX 1.2.5 fractional API for tuning (pass kernel+state+rng_key)
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
        # Unpack tuned parameters (dataclass-like)
        try:
            L_tuned = int(getattr(tuned_params, "L"))
            eps_tuned = float(getattr(tuned_params, "step_size"))
            logger.debug(f"MCLMC tuner results: L={L_tuned}, step_size={eps_tuned}")

            # Fail fast on invalid tuning results
            if L_tuned <= 0:
                raise SystemExit(f"MCLMC tuner returned invalid L={L_tuned} (must be > 0)")
            if eps_tuned <= 0:
                raise SystemExit(f"MCLMC tuner returned invalid step_size={eps_tuned} (must be > 0)")

        except Exception as e:
            raise SystemExit(f"MCLMC tuner returned unexpected params type: {type(tuned_params)!r}: {tuned_params}") from e

        # Save tuned parameters if tuned_path is provided
        if tuned_path:
            from llc.artifacts import save_tuned_params
            tuned_payload = {
                "L": int(L_tuned),
                "step_size": float(eps_tuned),
                "desired_energy_var": float(desired_energy_var),
                "integrator_name": integrator_name,
                "diagonal_preconditioning": bool(diagonal_preconditioning),
            }
            save_tuned_params(tuned_path, "mclmc", tuned_payload)
    else:
        # Using cached parameters, split key for run only
        key_run = key

    # Build final SamplingAlgorithm for sampling (integrator specified here)
    integrator = getattr(blackjax.mcmc.integrators, integrator_name)
    final_mclmc_algo = blackjax.mclmc(
        logdensity_fn, L=int(L_tuned), step_size=float(eps_tuned), integrator=integrator
    )
    step_single = jax.jit(final_mclmc_algo.step)

    # Prepare per-chain initial STATES (not raw arrays)
    chains = init_thetas.shape[0]
    keys_run = jax.random.split(key_run, chains)
    init_states = jax.vmap(lambda theta, k: final_mclmc_algo.init(theta, rng_key=k))(init_thetas, keys_run)

    # Sanity: a state must have .position (per BlackJAX IntegratorState)
    try:
        _ = init_states[0].position  # type: ignore[attr-defined]
    except Exception:
        raise SystemExit("MCLMC driver expected BlackJAX state objects. "
                         "Did you pass raw arrays instead of algo.init(...)?)")

    # Build a concrete SamplerSpec with vmapped step
    from llc.samplers.base import SamplerSpec
    spec = SamplerSpec(
        name="mclmc",
        step_vmapped=jax.jit(jax.vmap(step_single, in_axes=(0, 0))),
        position_fn=lambda st: st.position,
        info_extractors={"energy": lambda info: info.energy_change},
        grads_per_step=1.0,
    )

    # Run using unified driver
    t0 = time.perf_counter()
    result = run_sampler_spec(
        spec=spec,
        rng_key=key_run,
        init_states=init_states,
        n_steps=draws,
        warmup=0,  # MCLMC uses tuner_steps, not warmup
        eval_every=eval_every,
        thin=thin,
        Ln_eval_f64_vmapped=Ln_full64,
        tiny_store_fn=tiny_store,
    )
    total_time = time.perf_counter() - t0

    # Estimate and subtract eval time
    from llc.runners_common import estimate_sampling_time

    C, M = result.L_hist.shape  # chains, eval points
    sampling_time = estimate_sampling_time(
        total_time, result.warmup_time_seconds, Ln_full64, init_thetas, M
    )

    # Convert back to expected format
    kept_stacked = np.asarray(result.kept)  # (C, K, k)
    means = np.asarray(result.mean_L)  # (C,)
    vars_ = np.asarray(result.var_L)  # (C,)
    ns = np.asarray(result.n_L)  # (C,)
    dE = result.extras.get("energy", jnp.zeros_like(result.L_hist))
    dE_list = [np.asarray(dE[c]) for c in range(dE.shape[0])]
    L_histories = [np.asarray(result.L_hist[c]) for c in range(result.L_hist.shape[0])]

    # Build timing and work dictionaries
    timings = {
        "warmup": float(result.warmup_time_seconds),
        "sampling": float(sampling_time),
    }

    # MCLMC work computation
    chains = init_thetas.shape[0]
    work = {
        "n_steps": int(chains * draws),
        "n_full_loss": int(chains * M),
    }

    return SamplerResult(
        Ln_histories=L_histories,
        theta_thin=kept_stacked,
        acceptance=None,  # MCLMC doesn't have acceptance rates
        energy=dE_list,  # MCLMC has energy changes
        timings=timings,
        work=work,
    )
