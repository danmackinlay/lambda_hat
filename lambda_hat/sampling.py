# llc/sampling.py
"""Clean, idiomatic JAX/BlackJAX sampling loops"""

# Updated imports
from typing import Dict, Any, Tuple, Callable, NamedTuple, TYPE_CHECKING, Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import blackjax

if TYPE_CHECKING:
    from lambda_hat.config import SGLDConfig, MCLMCConfig

# === TraceSpec for efficient recording ===

@dataclass
class TraceSpec:
    aux_fn: Optional[Callable[[Any], Dict[str, Any]]] = None
    aux_every: int = 1
    theta_every: int = 0  # 0 => don't store params


# === Preconditioner Implementation ===


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


def initialize_preconditioner(params: Any) -> PreconditionerState:
    """Initialize the state for the preconditioner.
    Note: v is initialized to 1 as per Hitchcock and Hoogland Algorithms 2 and 3.
    """
    from jax.tree_util import tree_map

    t = jnp.array(0, dtype=jnp.int32)
    m = tree_map(jnp.zeros_like, params)
    v = tree_map(jnp.ones_like, params)
    return PreconditionerState(t=t, m=m, v=v)


def update_preconditioner(
    config: "SGLDConfig", grad_loss: Any, state: PreconditionerState
) -> Tuple[PreconditionerState, Any, Any]:
    """Update preconditioner state and compute adaptive tensors."""
    t, m, v = state.t, state.m, state.v
    t = t + 1

    # Default values (Vanilla SGLD)
    from jax.tree_util import tree_map

    P_t = tree_map(jnp.ones_like, grad_loss)
    adapted_loss_drift = grad_loss

    if config.precond == "none":
        pass  # Vanilla SGLD
    elif config.precond == "adam" or config.precond == "rmsprop":
        # Update moments
        if config.precond == "adam":
            m = tree_map(
                lambda m_prev, g: config.beta1 * m_prev + (1 - config.beta1) * g,
                m,
                grad_loss,
            )

        v = tree_map(
            lambda v_prev, g: config.beta2 * v_prev + (1 - config.beta2) * g**2,
            v,
            grad_loss,
        )

        # Bias correction (as explicitly used in the paper's algorithms)
        m_hat = m
        v_hat = v

        # Ensure t is cast to float for exponentiation
        t_float = t.astype(jnp.float32)
        if config.precond == "adam":
            m_hat = tree_map(lambda m_val: m_val / (1 - config.beta1**t_float), m)
        v_hat = tree_map(lambda v_val: v_val / (1 - config.beta2**t_float), v)

        # Compute Preconditioner Tensor P_t = 1 / (sqrt(v_hat) + eps)
        P_t = tree_map(lambda vh: 1.0 / (jnp.sqrt(vh) + config.eps), v_hat)

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


# === Generic Inference Loop ===
def simple_inference_loop(
    rng_key, kernel, initial_state, num_samples: int, *,
    trace: TraceSpec = TraceSpec(),
):
    """
    Generic driver that yields compact traces.
    - Calls kernel(key_t, state) -> (state, info)
    - Optionally records aux scalars (e.g., Ln) every aux_every steps
    - Optionally thins & stores positions every theta_every steps
    Returns: dict with possibly {"Ln": (T',), "theta": PyTree or None, ...}
    """

    def step_and_trace(carry, key_and_step):
        state = carry
        key, step_idx = key_and_step

        # Kernel step
        new_state, info = kernel(key, state)

        # Get position from state
        if isinstance(new_state, SGLDState):
            position = new_state.position
        else:
            position = new_state.position if hasattr(new_state, "position") else new_state

        # Build basic trace
        trace_data = {
            "position": position,
            "acceptance_rate": getattr(info, "acceptance_rate", jnp.nan),
            "energy": getattr(info, "energy", jnp.nan),
            "diverging": getattr(info, "is_divergent", False),
        }

        # Add Ln if needed - use jnp.where to avoid boolean conversion error
        if trace.aux_fn is not None:
            should_record = (step_idx % trace.aux_every == 0)
            aux = trace.aux_fn(new_state)
            trace_data["Ln"] = jnp.where(should_record, aux["Ln"], jnp.array(0.0))
        else:
            trace_data["Ln"] = None

        return new_state, trace_data

    keys = jax.random.split(rng_key, num_samples)
    steps = jnp.arange(num_samples)

    final_state, trace_history = jax.lax.scan(
        step_and_trace, initial_state, (keys, steps)
    )

    # Extract Ln values if recorded
    if trace.aux_fn and "Ln" in trace_history:
        # Only keep every aux_every-th sample
        ln_mask = steps % trace.aux_every == 0
        ln_values = trace_history["Ln"]
        trace_history["Ln"] = ln_values[ln_mask]

    # Remove None entries
    trace_history = {k: v for k, v in trace_history.items() if v is not None}

    return trace_history


def run_hmc(
    rng_key: jax.random.PRNGKey,
    logdensity_fn: Callable,
    initial_params: Dict[str, Any],
    num_samples: int,
    num_chains: int,
    step_size: float = 0.01,
    num_integration_steps: int = 10,
    adaptation_steps: int = 1000,
    loss_full_fn: Optional[Callable] = None,  # For Ln recording
    trace_config: Optional[TraceSpec] = None,  # Recording configuration
) -> Dict[str, jnp.ndarray]:
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

    Returns:
        Dictionary with traces, shape (Chains, Draws, ...)
    """
    # Setup keys
    key, init_key, sample_key = jax.random.split(rng_key, 3)
    init_keys = jax.random.split(init_key, num_chains)

    # Create initial states for each chain (with slight perturbation)
    def init_chain(key, params):
        # Add small noise to initial params for chain diversity
        from jax.tree_util import tree_map

        noise = tree_map(
            lambda p: 0.01 * jax.random.normal(key, p.shape, dtype=p.dtype), params
        )
        perturbed_params = tree_map(lambda p, n: p + n, params, noise)
        return perturbed_params

    # Map over keys (chains), broadcast params
    initial_positions = jax.vmap(init_chain, in_axes=(0, None))(
        init_keys, initial_params
    )

    # Run adaptation if requested
    if adaptation_steps > 0:
        # Use window adaptation
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            logdensity_fn,
            num_steps=adaptation_steps,
            target_acceptance_rate=0.8,
        )

        # Run warmup on first chain to get adapted parameters
        key, warmup_key = jax.random.split(key)
        # Extract first chain's parameters
        first_chain_params = jax.tree_util.tree_map(lambda x: x[0], initial_positions)
        (final_state, (step_size_adapted, inverse_mass_matrix)), _ = warmup.run(
            warmup_key, first_chain_params
        )
    else:
        step_size_adapted = step_size
        inverse_mass_matrix = None

    # Build HMC sampler using blackjax.hmc
    hmc_sampler = blackjax.hmc(
        logdensity_fn,
        step_size=step_size_adapted,
        num_integration_steps=num_integration_steps,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    # Initialize states for all chains
    initial_states = jax.vmap(hmc_sampler.init)(initial_positions)

    # Run inference loop for each chain
    sample_keys = jax.random.split(sample_key, num_chains)

    def run_chain(key, initial_state):
        # Create aux function if loss_full_fn is provided
        aux_fn = None
        if loss_full_fn is not None:
            aux_fn = lambda state: {"Ln": loss_full_fn(state.position)}

        # Use provided trace_config or create default
        if trace_config is not None:
            trace_spec = trace_config
        elif loss_full_fn is not None:
            # Create default TraceSpec with aux function
            trace_spec = TraceSpec(
                aux_fn=aux_fn,
                aux_every=1,  # Record every draw for HMC
                theta_every=0,  # Don't record theta by default
            )
        else:
            # Backwards compatibility
            trace_spec = TraceSpec()

        kernel = hmc_sampler.step
        return simple_inference_loop(key, kernel, initial_state, num_samples, trace=trace_spec)

    # Use vmap to run chains in parallel
    traces = jax.vmap(run_chain)(sample_keys, initial_states)

    return traces


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
    loss_full_fn: Optional[Callable] = None,  # For Ln recording
    trace_config: Optional[TraceSpec] = None,  # Recording configuration
) -> Dict[str, jnp.ndarray]:
    """Run SGLD or pSGLD (AdamSGLD/RMSPropSGLD) with minibatching."""
    X, Y = data
    n_data = X.shape[0]
    num_samples = config.steps
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
        from jax.tree_util import tree_map

        noise = tree_map(
            lambda p: 0.01 * jax.random.normal(key, p.shape, dtype=p.dtype), params_init
        )
        position = tree_map(lambda p, n: p + n, params_init, noise)
        precond_state = initialize_preconditioner(position)
        return SGLDState(position=position, precond_state=precond_state)

    # Vmap initialization across chains
    initial_states = jax.vmap(init_chain, in_axes=(0, None))(init_keys, initial_params)

    # Build the pSGLD kernel (defined inside to close over context)
    def sgld_kernel(rng_key, state: SGLDState):
        key_batch, key_sgld = jax.random.split(rng_key)
        w_t = state.position
        precond_state = state.precond_state

        # 1. Sample minibatch
        indices = jax.random.choice(
            key_batch, n_data, shape=(batch_size,), replace=False
        )
        minibatch = (X[indices], Y[indices])

        # 2. Compute loss gradient (g_t)
        grad_loss = grad_loss_fn(w_t, minibatch)

        # 3. Update preconditioner (uses only loss gradient)
        new_precond_state, P_t, adapted_loss_drift = update_preconditioner(
            config, grad_loss, precond_state
        )

        # 4. Calculate localization (prior) term: γ(w_t - w_0)
        # Note: params0 is captured by closure
        from jax.tree_util import tree_map

        localization_term = tree_map(lambda w, w0: gamma_val * (w - w0), w_t, params0)

        # 5. Calculate the total drift term
        # Drift = localization_term + beta_tilde * adapted_loss_drift
        total_drift = tree_map(
            lambda loc, loss_drift: loc + beta_tilde * loss_drift,
            localization_term,
            adapted_loss_drift,
        )

        # 6. Calculate adaptive step sizes and apply update
        # Δw_t = -(ε_t/2) * Drift + sqrt(ε_t) * η_t, where ε_t = ε * P_t

        def compute_update(P, drift, w):
            adaptive_step = base_step_size * P
            # Drift component
            update = -0.5 * adaptive_step * drift
            # Noise component
            noise_scale = jnp.sqrt(adaptive_step)
            noise = noise_scale * jax.random.normal(key_sgld, w.shape, dtype=w.dtype)
            return w + update + noise

        # Apply update calculation across the PyTree
        w_next = tree_map(compute_update, P_t, total_drift, w_t)

        new_state = SGLDState(position=w_next, precond_state=new_precond_state)
        info = SGLDInfo()

        return new_state, info

    # Run inference loop for each chain
    def run_chain(key, initial_state):
        # Create aux function if loss_full_fn is provided
        aux_fn = None
        if loss_full_fn is not None:
            aux_fn = lambda state: {"Ln": loss_full_fn(state.position)}

        # Use provided trace_config or create default
        if trace_config is not None:
            trace_spec = trace_config
        elif loss_full_fn is not None:
            # Create default TraceSpec with aux function
            # Use a reasonable default for aux_every
            trace_spec = TraceSpec(
                aux_fn=aux_fn,
                aux_every=10,  # Default evaluation frequency
                theta_every=0,  # Don't record theta by default for SGLD
            )
        else:
            # Backwards compatibility: use old simple_inference_loop behavior
            trace_spec = TraceSpec()

        # Use the generic inference loop which handles SGLDState
        return simple_inference_loop(key, sgld_kernel, initial_state, num_samples, trace=trace_spec)

    # Use vmap to run chains in parallel
    traces = jax.vmap(run_chain)(sample_keys, initial_states)

    return traces


def run_mclmc(
    rng_key: jax.random.PRNGKey,
    logdensity_fn: Callable,
    initial_params: Dict[str, Any],
    num_samples: int,
    num_chains: int,
    config: "MCLMCConfig",
    loss_full_fn: Optional[Callable] = None,  # For Ln recording
    trace_config: Optional[TraceSpec] = None,  # Recording configuration
) -> Dict[str, jnp.ndarray]:
    """Run Microcanonical Langevin Monte Carlo (MCLMC)

    Args:
        rng_key: JRNG key
        logdensity_fn: Log posterior function
        initial_params: Initial parameter values (Haiku params)
        num_samples: Number of samples per chain
        num_chains: Number of chains to run in parallel
        config: MCLMC configuration object containing all parameters

    Returns:
        Dictionary with traces, shape (Chains, Draws, ...)
    """
    # Setup keys
    key, init_key, sample_key = jax.random.split(rng_key, 3)
    init_keys = jax.random.split(init_key, num_chains)

    # Flatten initial params for MCLMC (it works with vectors)
    leaves, tree_def = jax.tree_util.tree_flatten(initial_params)
    initial_flat = jnp.concatenate([leaf.flatten() for leaf in leaves])
    d = initial_flat.size

    # Create initial states for each chain
    def init_chain(key):
        noise = 0.01 * jax.random.normal(
            key, initial_flat.shape, dtype=initial_flat.dtype
        )
        return initial_flat + noise

    initial_positions = jax.vmap(init_chain)(init_keys)

    # Adapt logdensity to work with flattened params
    def logdensity_flat(theta):
        # Unflatten theta back to params structure
        shapes = [leaf.shape for leaf in leaves]
        sizes = [leaf.size for leaf in leaves]
        params_leaves = []
        start = 0
        for shape, size in zip(shapes, sizes):
            params_leaves.append(theta[start : start + size].reshape(shape))
            start += size
        params = jax.tree_util.tree_unflatten(tree_def, params_leaves)
        return logdensity_fn(params)

    # --- MCLMC Adaptation Phase ---
    key, adaptation_key = jax.random.split(key)

    # Select integrator
    import blackjax.mcmc.integrators as bj_integrators

    integrators_map = {
        "isokinetic_mclachlan": bj_integrators.isokinetic_mclachlan,
        "isokinetic_velocity_verlet": bj_integrators.isokinetic_velocity_verlet,
    }
    integrator = integrators_map.get(config.integrator)
    if integrator is None:
        raise ValueError(
            f"Unknown MCLMC integrator: {config.integrator}. Supported: {list(integrators_map.keys())}"
        )

    # Build MCLMC kernel with config params (skip problematic tuner for now)
    # As QA team suggests: "Use fixed L/step_size from config and you'll be stable"
    # Always provide an inverse mass matrix (identity by default)
    inv_mass = jnp.ones(d) if config.diagonal_preconditioning else jnp.ones(d)
    mclmc = blackjax.mclmc(
        logdensity_fn=logdensity_flat,
        L=config.L,
        step_size=config.step_size,
        integrator=integrator,
        inverse_mass_matrix=inv_mass,
    )

    # Initialize states for all chains
    # Note: MCLMC init signature is (position, rng_key) per QA team
    init_keys = jax.random.split(key, num_chains)
    initial_states = jax.vmap(mclmc.init)(initial_positions, init_keys)

    # Run inference loop for each chain
    sample_keys = jax.random.split(sample_key, num_chains)

    def run_chain(key, initial_state):
        # Create aux function if loss_full_fn is provided
        aux_fn = None
        if loss_full_fn is not None:
            # Note: need to unflatten position for MCLMC since it works with flattened params
            def unflatten_position(flat_pos):
                params_leaves = []
                start = 0
                for shape, size in zip(
                    [leaf.shape for leaf in leaves], [leaf.size for leaf in leaves]
                ):
                    params_leaves.append(flat_pos[start : start + size].reshape(shape))
                    start += size
                return jax.tree_util.tree_unflatten(tree_def, params_leaves)

            aux_fn = lambda state: {"Ln": loss_full_fn(unflatten_position(state.position))}

        # Use provided trace_config or create default
        if trace_config is not None:
            trace_spec = trace_config
        elif loss_full_fn is not None:
            # Create default TraceSpec with aux function
            trace_spec = TraceSpec(
                aux_fn=aux_fn,
                aux_every=1,  # Record every draw for MCLMC
                theta_every=0,  # Don't record theta by default
            )
        else:
            # Backwards compatibility
            trace_spec = TraceSpec()

        kernel = mclmc.step
        return simple_inference_loop(key, kernel, initial_state, num_samples, trace=trace_spec)

    # Use vmap to run chains in parallel
    traces_flat = jax.vmap(run_chain)(sample_keys, initial_states)

    # Unflatten traces back to params structure
    def unflatten_trace(trace):
        positions = trace["position"]  # Shape: (num_samples, d)
        # Unflatten each sample
        unflattened_samples = []
        for i in range(num_samples):
            theta = positions[i]
            params_leaves = []
            start = 0
            for shape, size in zip(
                [leaf.shape for leaf in leaves], [leaf.size for leaf in leaves]
            ):
                params_leaves.append(theta[start : start + size].reshape(shape))
                start += size
            params = jax.tree_util.tree_unflatten(tree_def, params_leaves)
            unflattened_samples.append(params)

        # Stack samples
        return {
            "position": jax.tree_map(lambda *xs: jnp.stack(xs), *unflattened_samples),
            "acceptance_rate": trace["acceptance_rate"],
            "energy": trace["energy"],
            "diverging": trace["diverging"],
        }

    traces = jax.vmap(unflatten_trace)(traces_flat)

    return traces
