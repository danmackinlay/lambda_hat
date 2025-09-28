# lambda_hat/sampling.py
"""Clean, idiomatic JAX/BlackJAX sampling loops"""

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, NamedTuple, Optional, Tuple

import blackjax
import blackjax.mcmc.integrators as bj_integrators
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

if TYPE_CHECKING:
    from lambda_hat.config import SGLDConfig


# Define Preconditioner State structure
class PreconditionerState(NamedTuple):
    t: jnp.ndarray  # Timestep (scalar)
    m: Any  # First moment estimate (PyTree)
    v: Any  # Second moment estimate (PyTree)


class SGLDState(NamedTuple):
    position: Any  # Parameter PyTree
    precond_state: PreconditionerState


class SGLDInfo(NamedTuple):
    acceptance_rate: jnp.ndarray = jnp.array(1.0)
    energy: jnp.ndarray = jnp.nan
    is_divergent: bool = False


class SamplerRunResult(NamedTuple):
    traces: Dict[str, jnp.ndarray]
    timings: Dict[str, float]  # {'adaptation': 0.0, 'sampling': 0.0, 'total': 0.0}


def initialize_preconditioner(params: Any) -> PreconditionerState:
    """Initialize the state for the preconditioner.
    Note: v is initialized to 1 as per Hitchcock and Hoogland Algorithms 2 and 3.
    """

    t = jnp.array(0, dtype=jnp.int32)
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.ones_like, params)
    return PreconditionerState(t=t, m=m, v=v)


def update_preconditioner(
    config: "SGLDConfig", grad_loss: Any, state: PreconditionerState
) -> Tuple[PreconditionerState, Any, Any]:
    """Update preconditioner state and compute adaptive tensors."""
    t, m, v = state.t, state.m, state.v
    t = t + 1

    # Default values (Vanilla SGLD)

    P_t = jax.tree.map(jnp.ones_like, grad_loss)
    adapted_loss_drift = grad_loss

    if config.precond == "none":
        pass  # Vanilla SGLD
    elif config.precond == "adam" or config.precond == "rmsprop":
        # Update moments
        if config.precond == "adam":
            m = jax.tree.map(
                lambda m_prev, g: config.beta1 * m_prev + (1 - config.beta1) * g,
                m,
                grad_loss,
            )

        v = jax.tree.map(
            lambda v_prev, g: config.beta2 * v_prev + (1 - config.beta2) * g**2,
            v,
            grad_loss,
        )

        # Optional bias correction (as explicitly used in the paper's algorithms)
        m_hat = m
        v_hat = v
        if getattr(config, "bias_correction", True):
            # Ensure t is cast to float for exponentiation
            t_float = t.astype(jnp.float32)
            if config.precond == "adam":
                m_hat = jax.tree.map(lambda m_val: m_val / (1 - config.beta1**t_float), m)
            v_hat = jax.tree.map(lambda v_val: v_val / (1 - config.beta2**t_float), v)

        # Compute Preconditioner Tensor P_t = 1 / (sqrt(v_hat) + eps)
        P_t = jax.tree.map(lambda vh: 1.0 / (jnp.sqrt(vh) + config.eps), v_hat)

        # Determine adapted loss drift
        if config.precond == "adam":
            adapted_loss_drift = m_hat
        # else (rmsprop): adapted_loss_drift remains grad_loss
    else:
        # Raise error for unknown preconditioner
        raise ValueError(
            f"Unknown SGLD preconditioner: {config.precond}. Supported: 'none', 'adam', 'rmsprop'."
        )

    new_state = PreconditionerState(t=t, m=m, v=v)
    return new_state, P_t, adapted_loss_drift


# === Optimized Inference Loop ===
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
        trace = tree_map(lambda x: x[::aux_every], trace)

    return trace


def run_hmc(
    rng_key: jax.random.PRNGKey,
    logdensity_fn: Callable,
    initial_params: Dict[str, Any],
    num_samples: int,
    num_chains: int,
    step_size: float = 0.01,
    num_integration_steps: int = 10,
    adaptation_steps: int = 1000,
    target_acceptance: float = 0.8,
    loss_full_fn: Optional[Callable] = None,
) -> SamplerRunResult:
    """Run HMC with optional adaptation

    Args:
        rng_key: JRNG key
        logdensity_fn: Log posterior function
        initial_params: Initial parameter values (Haiku params)
        num_samples: Number of samples per chain
        num_chains: Number of chains to run in parallel
        step_size: Initial step size
        num_integration_steps: Number of leapfrog steps
        adaptation_steps: Number of adaptation steps (0 to disable)
        loss_full_fn: Optional function to compute loss for recording

    Returns:
        SamplerRunResult with traces and timing information
    """
    # Setup keys
    key, init_key, sample_key = jax.random.split(rng_key, 3)
    init_keys = jax.random.split(init_key, num_chains)

    # 1) diversify initial positions
    def jitter(key, params):
        leaves, treedef = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(key, len(leaves))
        keytree = jax.tree_util.tree_unflatten(treedef, keys)
        return jax.tree.map(
            lambda p, k: p + 0.01 * jax.random.normal(k, p.shape, p.dtype),
            params,
            keytree,
        )

    # Map over keys (chains), broadcast params
    init_positions = jax.vmap(jitter, in_axes=(0, None))(init_keys, initial_params)

    # Flatten parameters for dimensionality calculation
    D = sum(p.size for p in jax.tree_util.tree_leaves(initial_params))
    ref_dtype = jax.tree_util.tree_leaves(initial_params)[0].dtype

    # 2) warmup on one chain (Adaptation)
    adaptation_start_time = time.time()

    # Initialize defaults
    step_size_adapted = step_size
    # CRITICAL FIX: Default mass matrix must be a 1D array of size D
    inv_mass = jnp.ones(D, dtype=ref_dtype)

    if adaptation_steps > 0:
        wa = blackjax.window_adaptation(
            blackjax.hmc,
            logdensity_fn,
            target_acceptance_rate=target_acceptance,
            num_integration_steps=num_integration_steps,
        )

        one_pos = jax.tree.map(lambda x: x[0], init_positions)

        # PASS ONLY num_steps (the warmup length) to .run()
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

    # 3) build kernel and init all chains
    hmc = blackjax.hmc(
        logdensity_fn,
        step_size=step_size_adapted,
        num_integration_steps=num_integration_steps,
        inverse_mass_matrix=inv_mass,
    )

    # Initialize states for all chains
    init_states = jax.vmap(hmc.init)(init_positions)

    # 4) drive all chains using the efficient loop
    sample_keys = jax.random.split(sample_key, num_chains)

    # Define aux_fn for recording loss
    def aux_fn(state):
        if loss_full_fn is not None:
            return {"Ln": loss_full_fn(state.position)}
        else:
            return {"Ln": jnp.nan}

    # Calculate work per step (FGEs): HMC uses full gradients.
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

    return SamplerRunResult(traces=traces, timings=timings)


# === SGLD / pSGLD ===
def run_sgld(
    rng_key: jax.random.PRNGKey,
    grad_loss_fn: Callable,
    initial_params: Dict[str, Any],
    params0: Dict[str, Any],  # ERM center for localization (w_0)
    data: Tuple[jnp.ndarray, jnp.ndarray],
    config: "SGLDConfig",
    num_chains: int,
    beta: float,
    gamma: float,
    loss_full_fn: Optional[Callable] = None,
) -> SamplerRunResult:
    """Run SGLD or pSGLD (AdamSGLD/RMSPropSGLD) with minibatching."""
    X, Y = data
    # Convert to JAX arrays if they're NumPy arrays (needed for indexing in JIT)
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)
    n_data = X.shape[0]
    batch_size = config.batch_size
    base_step_size = config.step_size

    # Ensure scalars match the parameter dtype
    ref_dtype = jax.tree_util.tree_leaves(initial_params)[0].dtype
    # beta_tilde = n*beta (scaling factor for loss gradient)
    beta_tilde = jnp.asarray(beta * n_data, dtype=ref_dtype)
    gamma_val = jnp.asarray(gamma, dtype=ref_dtype)

    # Setup keys
    key, init_key, sample_key = jax.random.split(rng_key, 3)
    init_keys = jax.random.split(init_key, num_chains)
    sample_keys = jax.random.split(sample_key, num_chains)

    # Initialize states (position + preconditioner)
    def init_chain(key, params_init):
        # Perturb initial position slightly for diversity

        noise = jax.tree.map(
            lambda p: 0.01 * jax.random.normal(key, p.shape, dtype=p.dtype), params_init
        )
        position = jax.tree.map(lambda p, n: p + n, params_init, noise)
        precond_state = initialize_preconditioner(position)
        return SGLDState(position=position, precond_state=precond_state)

    # Vmap initialization across chains
    initial_states = jax.vmap(init_chain, in_axes=(0, None))(init_keys, initial_params)

    # Build the pSGLD kernel (defined inside to close over context)
    def sgld_kernel(rng_key, state: SGLDState):
        key_batch, key_sgld = jax.random.split(rng_key)
        w_t = state.position
        precond_state = state.precond_state

        # 1. Sample minibatch indices. If the configured batch_size exceeds n_data,
        # fall back to sampling *with* replacement (static shape; JIT-friendly).
        replace_flag = batch_size > n_data  # Python bool; static
        indices = jax.random.choice(key_batch, n_data, shape=(batch_size,), replace=replace_flag)
        minibatch = (X[indices], Y[indices])

        # 2. Compute loss gradient (g_t)
        grad_loss = grad_loss_fn(w_t, minibatch)

        # 3. Update preconditioner (uses only loss gradient)
        new_precond_state, P_t, adapted_loss_drift = update_preconditioner(
            config, grad_loss, precond_state
        )

        # 4. Calculate localization (prior) term: γ(w_t - w_0)
        # Note: params0 is captured by closure
        localization_term = jax.tree.map(lambda w, w0: gamma_val * (w - w0), w_t, params0)

        # 5. Calculate the total drift term
        # Drift = localization_term + beta_tilde * adapted_loss_drift
        total_drift = jax.tree.map(
            lambda loc, loss_drift: loc + beta_tilde * loss_drift,
            localization_term,
            adapted_loss_drift,
        )

        # 6. Calculate adaptive step sizes and apply update
        # Δw_t = -(ε_t/2) * Drift + sqrt(ε_t) * η_t, where ε_t = ε * P_t

        # Flatten parameter tree for independent noise sampling
        theta_t, unravel = jax.flatten_util.ravel_pytree(w_t)
        P_flat, _ = jax.flatten_util.ravel_pytree(P_t)
        drift_flat, _ = jax.flatten_util.ravel_pytree(total_drift)
        adaptive_step = base_step_size * P_flat
        update_flat = -0.5 * adaptive_step * drift_flat
        noise = jax.random.normal(key_sgld, theta_t.shape, dtype=theta_t.dtype)
        theta_next = theta_t + update_flat + jnp.sqrt(adaptive_step) * noise
        w_next = unravel(theta_next)

        new_state = SGLDState(position=w_next, precond_state=new_precond_state)
        info = SGLDInfo()

        return new_state, info

    # Define aux_fn for recording loss
    def aux_fn(state):
        if loss_full_fn is not None:
            return {"Ln": loss_full_fn(state.position)}
        else:
            return {"Ln": jnp.nan}

    # Calculate work per step (FGEs): SGLD uses minibatch.
    work_per_step = float(batch_size) / float(n_data)

    # Use eval_every from config if available, otherwise default to 10 (matches config.py)
    eval_every = getattr(config, "eval_every", 10)

    # Start sampling timer
    sampling_start_time = time.time()

    # Use vmap with the optimized inference loop
    traces = jax.vmap(
        lambda k, s: inference_loop_extended(
            k,
            sgld_kernel,
            s,
            num_samples=config.steps,
            aux_fn=aux_fn,
            aux_every=eval_every,
            work_per_step=work_per_step,
        )
    )(sample_keys, initial_states)

    # Ensure sampling is finished
    jax.block_until_ready(traces)
    sampling_time = time.time() - sampling_start_time

    timings = {
        "adaptation": 0.0,  # SGLD has no separate adaptation phase
        "sampling": sampling_time,
        "total": sampling_time,
    }

    return SamplerRunResult(traces=traces, timings=timings)


def run_mclmc(
    rng_key,
    logdensity_fn,
    initial_params,
    num_samples,
    num_chains,
    config,
    loss_full_fn: Optional[Callable] = None,
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
        # Remove the incorrect inverse_mass_matrix argument from Doc 1 L453
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

    return SamplerRunResult(traces=traces, timings=timings)
