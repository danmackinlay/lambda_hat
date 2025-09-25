# llc/sampling.py
"""Clean, idiomatic JAX/BlackJAX sampling loops"""

# Updated imports
from typing import Dict, Any, Tuple, Callable, NamedTuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import blackjax
# Updated to use new JAX tree API (jax>=0.4.28)

if TYPE_CHECKING:
    from lambda_hat.config import SGLDConfig, MCLMCConfig

# === TraceSpec removed - using optimized inference_loop_ln_only instead ===


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

        # Bias correction (as explicitly used in the paper's algorithms)
        m_hat = m
        v_hat = v

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
def inference_loop_ln_only(rng_key, kernel, initial_state, num_samples, loss_fn, record_every=1):
    """
    Efficient inference loop using jax.lax.scan that only records Ln and diagnostics.

    Args:
        loss_fn: A function that takes the position (PyTree or flat vector)
                 and returns the loss Ln. Must handle unflattening internally if needed.
        record_every: Thinning factor for the trace.
    """
    @jax.jit
    def one_step(state, rng_key):
        new_state, info = kernel(rng_key, state)

        # Extract position (defensive programming removed - use position directly)
        position = new_state.position

        # Compute Loss (Ln). We compute this every step and thin post-scan.
        ln_val = loss_fn(position)

        trace_data = {
            "Ln": ln_val,
            "acceptance_rate": getattr(info, "acceptance_rate", jnp.nan),
            "energy": getattr(info, "energy", jnp.nan),
            # Standardize the key for divergence
            "is_divergent": getattr(info, "is_divergent", getattr(info, "diverging", False)),
        }
        return new_state, trace_data

    keys = jax.random.split(rng_key, num_samples)
    # Run the scan (replaces inefficient Python loop)
    _, trace = jax.lax.scan(one_step, initial_state, keys)

    # Apply thinning AFTER the scan (efficient JAX pattern)
    if record_every > 1:
        trace = jax.tree.map(lambda x: x[::record_every], trace)

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
    loss_full: Callable,
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

        noise = jax.tree.map(
            lambda p: 0.01 * jax.random.normal(key, p.shape, dtype=p.dtype), params
        )
        perturbed_params = jax.tree.map(lambda p, n: p + n, params, noise)
        return perturbed_params

    # Map over keys (chains), broadcast params
    initial_positions = jax.vmap(init_chain, in_axes=(0, None))(
        init_keys, initial_params
    )

    # Run adaptation if requested
    if adaptation_steps > 0:
        # Use BlackJAX 1.2.5 API directly (defensive programming removed)
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            logdensity_fn,
            target_acceptance_rate=0.8,
            num_integration_steps=num_integration_steps,
        )

        # Run warmup on first chain to get adapted parameters
        key, warmup_key = jax.random.split(key)
        one_pos = jax.tree.map(lambda x: x[0], initial_positions)

        (final_state, (step_size_adapted, inverse_mass_matrix)), _ = warmup.run(
            warmup_key, one_pos, num_steps=adaptation_steps)
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
        kernel = hmc_sampler.step

        # HMC uses the original parameter structure.
        loss_fn_for_loop = lambda position: loss_full(position)

        # Use the efficient loop, removing the if/else trace_spec logic.
        return inference_loop_ln_only(
            key, kernel, initial_state, num_samples, loss_fn_for_loop, record_every=1
        )

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
    loss_full: Callable,
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

        def compute_update(P, drift, w):
            adaptive_step = base_step_size * P
            # Drift component
            update = -0.5 * adaptive_step * drift
            # Noise component
            noise_scale = jnp.sqrt(adaptive_step)
            noise = noise_scale * jax.random.normal(key_sgld, w.shape, dtype=w.dtype)
            return w + update + noise

        # Apply update calculation across the PyTree
        w_next = jax.tree.map(compute_update, P_t, total_drift, w_t)

        new_state = SGLDState(position=w_next, precond_state=new_precond_state)
        info = SGLDInfo()

        return new_state, info

    # Run inference loop for each chain
    def run_chain(key, initial_state):
        # SGLD uses the original parameter structure.
        loss_fn_for_loop = lambda position: loss_full(position)

        eval_every = getattr(config, 'eval_every', 10)

        # Use the efficient loop, removing the if/else trace_spec logic.
        return inference_loop_ln_only(
            key, sgld_kernel, initial_state, num_samples, loss_fn_for_loop, record_every=eval_every
        )

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
    loss_full: Callable,
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

    # Flatten initial params for MCLMC (modernized with ravel_pytree)
    initial_flat, unflatten_fn = jax.flatten_util.ravel_pytree(initial_params)
    d = initial_flat.size

    # Create initial states for each chain
    def init_chain(key):
        noise = 0.01 * jax.random.normal(
            key, initial_flat.shape, dtype=initial_flat.dtype
        )
        return initial_flat + noise

    initial_positions = jax.vmap(init_chain)(init_keys)

    # Adapt logdensity to work with flattened params (modernized with unflatten_fn)
    def logdensity_flat(theta):
        params = unflatten_fn(theta)
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

    # Create the kernel factory (required by the tuner in BlackJAX 1.2.5)
    def kernel_factory(L, step_size, sqrt_diag_cov):
        return blackjax.mclmc(
            logdensity_fn=logdensity_flat,
            L=L,
            step_size=step_size,
            sqrt_diag_cov=sqrt_diag_cov,
            integrator=integrator,
        )

    # Initialize state for the tuner (using the first chain)
    # In BlackJAX 1.2.5, mclmc.init takes (position, logdensity_fn) - no key
    initial_state_0 = blackjax.mcmc.mclmc.init(initial_positions[0], logdensity_flat)

    if config.num_steps > 0:
        print("Starting MCLMC adaptation...")
        (L_adapted, step_size_adapted, sqrt_diag_cov_adapted), _ = (
            blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel_factory,
                num_steps=config.num_steps,
                state=initial_state_0,
                rng_key=adaptation_key,
                # Pass adaptation parameters from config
                frac_tune1=config.frac_tune1,
                frac_tune2=config.frac_tune2,
                frac_tune3=config.frac_tune3,
                desired_energy_var=config.desired_energy_var,
                trust_in_estimate=config.trust_in_estimate,
                num_effective_samples=config.num_effective_samples,
                diagonal_preconditioning=config.diagonal_preconditioning,
            )
        )
        print(
            f"Adaptation complete. L: {L_adapted:.4f}, Step Size: {step_size_adapted:.4f}"
        )
    else:
        # Use fixed values if adaptation is disabled (num_steps=0)
        L_adapted = config.L
        step_size_adapted = config.step_size
        sqrt_diag_cov_adapted = jnp.ones(d)

    # --- MCLMC Sampling Phase ---
    mclmc_sampler = kernel_factory(L_adapted, step_size_adapted, sqrt_diag_cov_adapted)

    # Initialize states
    # Per-chain init: mclmc_sampler.init takes (position, logdensity_fn) - no key
    initial_states = jax.vmap(lambda pos: mclmc_sampler.init(pos, logdensity_flat))(
        initial_positions
    )

    # Run inference loop for each chain
    sample_keys = jax.random.split(sample_key, num_chains)

    def run_chain(key, initial_state):
        kernel = mclmc_sampler.step

        # Define the loss function that unflattens the parameters first (modernized)
        def loss_fn_for_loop(theta_flat):
            params = unflatten_fn(theta_flat)
            return loss_full(params)

        # Use the efficient loop, removing the if/else trace_spec logic.
        return inference_loop_ln_only(
            key, kernel, initial_state, num_samples, loss_fn_for_loop, record_every=1
        )

    # Use vmap to run chains in parallel
    traces = jax.vmap(run_chain)(sample_keys, initial_states)

    # >>> DELETE THE ENTIRE unflatten_trace BLOCK (lines 709-740 in Doc 17) <<<
    # We no longer trace parameters, so this is obsolete and causes errors.

    return traces
