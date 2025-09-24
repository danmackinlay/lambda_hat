# llc/sampling.py
"""Clean, idiomatic JAX/BlackJAX sampling loops"""

from typing import Dict, Any, Tuple, Callable, Optional
import jax
import jax.numpy as jnp
import blackjax
from blackjax import hmc, nuts, sgld, mclmc


def simple_inference_loop(rng_key, kernel, initial_state, num_samples):
    """Generic inference loop using jax.lax.scan

    Args:
        rng_key: JRNG key
        kernel: BlackJAX kernel function (e.g., hmc.step)
        initial_state: Initial sampler state
        num_samples: Number of samples to generate

    Returns:
        Dictionary of traced values with shape (num_samples, ...)
    """

    @jax.jit
    def one_step(state, rng_key):
        # Kernel step
        state, info = kernel(rng_key, state)

        # Define what to trace (minimal data needed for ArviZ later)
        # Assuming 'state.position' holds the parameters (standard BlackJAX)
        position = state.position if hasattr(state, 'position') else state
        trace_data = {
            'position': position,
            # Add sampler-specific stats if available
            'acceptance_rate': getattr(info, 'acceptance_rate', jnp.nan),
            'energy': getattr(info, 'energy', jnp.nan),
            'diverging': getattr(info, 'is_divergent', False),
        }
        return state, trace_data

    keys = jax.random.split(rng_key, num_samples)
    # Run the scan
    final_state, trace = jax.lax.scan(
        one_step, initial_state, keys
    )
    return trace  # e.g., trace['position'] shape: (Draws, Dimensions)


def run_hmc(
    rng_key: jax.random.PRNGKey,
    logdensity_fn: Callable,
    initial_params: Dict[str, Any],
    num_samples: int,
    num_chains: int,
    step_size: float = 0.01,
    num_integration_steps: int = 10,
    adaptation_steps: int = 1000,
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
        noise = jax.tree_map(
            lambda p: 0.01 * jax.random.normal(key, p.shape, dtype=p.dtype),
            params
        )
        perturbed_params = jax.tree_map(lambda p, n: p + n, params, noise)
        return perturbed_params

    initial_positions = jax.vmap(init_chain)(init_keys, initial_params)

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
        (final_state, (step_size_adapted, inverse_mass_matrix)), _ = warmup.run(
            warmup_key, initial_positions[0], num_steps=num_integration_steps
        )
    else:
        step_size_adapted = step_size
        inverse_mass_matrix = None

    # Build HMC kernel
    if inverse_mass_matrix is not None:
        hmc_kernel = hmc.build_kernel(
            integrator=hmc.integrator.leapfrog,
            divergence_threshold=1000,
        )
        hmc_init = hmc.init
    else:
        # Simple HMC without mass matrix adaptation
        hmc_kernel = lambda rng, state: hmc.kernel(
            logdensity_fn,
            step_size_adapted,
            num_integration_steps,
            divergence_threshold=1000,
        )(rng, state)
        hmc_init = lambda position: hmc.init(position, logdensity_fn)

    # Initialize states for all chains
    initial_states = jax.vmap(hmc_init)(initial_positions)

    # Run inference loop for each chain
    sample_keys = jax.random.split(sample_key, num_chains)

    def run_chain(key, initial_state):
        kernel = lambda rng, state: hmc.kernel(
            logdensity_fn,
            step_size_adapted,
            num_integration_steps,
            divergence_threshold=1000,
        )(rng, state)
        return simple_inference_loop(key, kernel, initial_state, num_samples)

    # Use vmap to run chains in parallel
    traces = jax.vmap(run_chain)(sample_keys, initial_states)

    return traces


def run_sgld(
    rng_key: jax.random.PRNGKey,
    grad_logpost_fn: Callable,
    initial_params: Dict[str, Any],
    data: Tuple[jnp.ndarray, jnp.ndarray],
    num_samples: int,
    num_chains: int,
    step_size: float = 0.001,
    batch_size: int = 32,
) -> Dict[str, jnp.ndarray]:
    """Run SGLD with minibatching

    Args:
        rng_key: JRNG key
        grad_logpost_fn: Gradient of log posterior that accepts (params, minibatch)
        initial_params: Initial parameter values (Haiku params)
        data: Tuple of (X, Y) for minibatching
        num_samples: Number of samples per chain
        num_chains: Number of chains to run in parallel
        step_size: Step size for SGLD
        batch_size: Minibatch size

    Returns:
        Dictionary with traces, shape (Chains, Draws, ...)
    """
    X, Y = data
    n_data = X.shape[0]

    # Setup keys
    key, init_key, sample_key = jax.random.split(rng_key, 3)
    init_keys = jax.random.split(init_key, num_chains)

    # Create initial states for each chain
    def init_chain(key, params):
        noise = jax.tree_map(
            lambda p: 0.01 * jax.random.normal(key, p.shape, dtype=p.dtype),
            params
        )
        return jax.tree_map(lambda p, n: p + n, params, noise)

    initial_positions = jax.vmap(init_chain)(init_keys, initial_params)

    # Build SGLD kernel with minibatching
    def sgld_kernel(rng_key, state):
        key_batch, key_sgld = jax.random.split(rng_key)

        # Sample minibatch
        indices = jax.random.choice(key_batch, n_data, shape=(batch_size,), replace=False)
        minibatch = (X[indices], Y[indices])

        # SGLD update
        grad = grad_logpost_fn(state, minibatch)
        noise_scale = jnp.sqrt(2 * step_size)
        noise = jax.tree_map(
            lambda g: noise_scale * jax.random.normal(key_sgld, g.shape, dtype=g.dtype),
            grad
        )
        new_state = jax.tree_map(
            lambda p, g, n: p + step_size * g + n,
            state, grad, noise
        )

        # Create info dict (SGLD doesn't have acceptance rate)
        info = type('Info', (), {'acceptance_rate': 1.0, 'energy': jnp.nan})()

        return new_state, info

    # Run inference loop for each chain
    sample_keys = jax.random.split(sample_key, num_chains)

    def run_chain(key, initial_state):
        return simple_inference_loop(key, sgld_kernel, initial_state, num_samples)

    # Use vmap to run chains in parallel
    traces = jax.vmap(run_chain)(sample_keys, initial_positions)

    return traces


def run_mclmc(
    rng_key: jax.random.PRNGKey,
    logdensity_fn: Callable,
    initial_params: Dict[str, Any],
    num_samples: int,
    num_chains: int,
    L: float = 1.0,
    step_size: float = 0.1,
    sqrt_diag_cov: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """Run Microcanonical Langevin Monte Carlo (MCLMC)

    Args:
        rng_key: JRNG key
        logdensity_fn: Log posterior function
        initial_params: Initial parameter values (Haiku params)
        num_samples: Number of samples per chain
        num_chains: Number of chains to run in parallel
        L: Trajectory length
        step_size: Step size
        sqrt_diag_cov: Square root of diagonal covariance (for preconditioning)

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
        noise = 0.01 * jax.random.normal(key, initial_flat.shape, dtype=initial_flat.dtype)
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
            params_leaves.append(theta[start:start+size].reshape(shape))
            start += size
        params = jax.tree_util.tree_unflatten(tree_def, params_leaves)
        return logdensity_fn(params)

    # Initialize MCLMC sampler
    mclmc_sampler = mclmc.mclmc(
        logdensity_fn=logdensity_flat,
        L=L,
        step_size=step_size,
        sqrt_diag_cov=sqrt_diag_cov if sqrt_diag_cov is not None else jnp.ones(d),
    )

    # Initialize states
    initial_states = jax.vmap(lambda pos: mclmc_sampler.init(pos))(initial_positions)

    # Run inference loop for each chain
    sample_keys = jax.random.split(sample_key, num_chains)

    def run_chain(key, initial_state):
        kernel = mclmc_sampler.step
        return simple_inference_loop(key, kernel, initial_state, num_samples)

    # Use vmap to run chains in parallel
    traces_flat = jax.vmap(run_chain)(sample_keys, initial_states)

    # Unflatten traces back to params structure
    def unflatten_trace(trace):
        positions = trace['position']  # Shape: (num_samples, d)
        # Unflatten each sample
        unflattened_samples = []
        for i in range(num_samples):
            theta = positions[i]
            params_leaves = []
            start = 0
            for shape, size in zip([leaf.shape for leaf in leaves], [leaf.size for leaf in leaves]):
                params_leaves.append(theta[start:start+size].reshape(shape))
                start += size
            params = jax.tree_util.tree_unflatten(tree_def, params_leaves)
            unflattened_samples.append(params)

        # Stack samples
        return {
            'position': jax.tree_map(lambda *xs: jnp.stack(xs), *unflattened_samples),
            'acceptance_rate': trace['acceptance_rate'],
            'energy': trace['energy'],
            'diverging': trace['diverging'],
        }

    traces = jax.vmap(unflatten_trace)(traces_flat)

    return traces