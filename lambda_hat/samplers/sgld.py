# lambda_hat/samplers/sgld.py
"""SGLD (Stochastic Gradient Langevin Dynamics) with preconditioning - FLAT INTERFACE ONLY"""

import time
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from lambda_hat.posterior import Posterior
from lambda_hat.samplers.common import inference_loop_extended
from lambda_hat.types import SamplerRunResult

if TYPE_CHECKING:
    from lambda_hat.config import SGLDConfig

Array = jnp.ndarray


# Define Preconditioner State structure (FLAT ONLY - no pytrees!)
class PreconditionerState(NamedTuple):
    t: jnp.ndarray  # Timestep (scalar)
    m: Array  # First moment estimate (FLAT array)
    v: Array  # Second moment estimate (FLAT array)


class SGLDState(NamedTuple):
    position: Array  # FLAT parameter vector
    precond_state: PreconditionerState


class SGLDInfo(NamedTuple):
    acceptance_rate: jnp.ndarray = jnp.array(1.0)
    energy: jnp.ndarray = jnp.nan
    is_divergent: bool = False


def initialize_preconditioner(flat_params: Array) -> PreconditionerState:
    """Initialize the state for the preconditioner (FLAT arrays only).
    Note: v is initialized to 1 as per Hitchcock and Hoogland Algorithms 2 and 3.
    """
    t = jnp.array(0, dtype=jnp.int32)
    m = jnp.zeros_like(flat_params)
    v = jnp.ones_like(flat_params)
    return PreconditionerState(t=t, m=m, v=v)


def update_preconditioner(
    config: "SGLDConfig", grad_loss: Array, state: PreconditionerState
) -> Tuple[PreconditionerState, Array, Array]:
    """Update preconditioner state and compute adaptive tensors (FLAT arrays only)."""
    t, m, v = state.t, state.m, state.v
    t = t + 1

    # Default values (Vanilla SGLD)
    P_t = jnp.ones_like(grad_loss)
    adapted_loss_drift = grad_loss

    if config.precond == "none":
        pass  # Vanilla SGLD
    elif config.precond == "adam" or config.precond == "rmsprop":
        # Update moments (flat array operations)
        if config.precond == "adam":
            m = config.beta1 * m + (1 - config.beta1) * grad_loss

        v = config.beta2 * v + (1 - config.beta2) * grad_loss**2

        # Optional bias correction
        m_hat = m
        v_hat = v
        if getattr(config, "bias_correction", True):
            t_float = t.astype(jnp.float32)
            if config.precond == "adam":
                m_hat = m / (1 - config.beta1**t_float)
            v_hat = v / (1 - config.beta2**t_float)

        # Compute Preconditioner Tensor P_t = 1 / (sqrt(v_hat) + eps)
        P_t = 1.0 / (jnp.sqrt(v_hat) + config.eps)

        # Determine adapted loss drift
        if config.precond == "adam":
            adapted_loss_drift = m_hat
        # else (rmsprop): adapted_loss_drift remains grad_loss
    else:
        raise ValueError(
            f"Unknown SGLD preconditioner: {config.precond}. Supported: 'none', 'adam', 'rmsprop'."
        )

    new_state = PreconditionerState(t=t, m=m, v=v)
    return new_state, P_t, adapted_loss_drift


def run_sgld(
    key: jax.random.PRNGKey,
    posterior: Posterior,
    data: Tuple[jnp.ndarray, jnp.ndarray],
    config: "SGLDConfig",
    num_chains: int,
    grad_loss_minibatch: Callable[[Array, Tuple[Array, Array]], Array],  # FLAT gradient
    loss_full_fn: Optional[Callable] = None,
    n_data: Optional[int] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    L0: Optional[float] = None,
) -> SamplerRunResult:
    """Run SGLD or pSGLD (AdamSGLD/RMSPropSGLD) with minibatching - FLAT INTERFACE ONLY

    Args:
        key: JRNG key
        posterior: Posterior with flat-space interface
        data: (X, Y) training data
        config: SGLD configuration
        num_chains: Number of chains to run in parallel
        grad_loss_minibatch: Gradient of minibatch loss in FLAT space
        loss_full_fn: Optional function to compute loss for recording
        n_data: Number of data points (required for minibatch scaling)
        beta: Temperature parameter (required for minibatch scaling)
        gamma: Prior strength parameter (required for localization term)
        L0: Reference loss (for LLC computation)

    Returns:
        SamplerRunResult with traces and timing information
    """
    X, Y = data
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)
    if n_data is None:
        n_data = X.shape[0]
    if beta is None:
        raise ValueError("beta must be provided for SGLD minibatch scaling")
    if gamma is None:
        raise ValueError("gamma must be provided for SGLD localization term")

    batch_size = config.batch_size
    base_step_size = float(config.step_size)

    # Get flat parameters and dtype
    flat0 = posterior.flat0
    dtype = posterior.vm.dtype

    # Scale factors for SGLD drift
    beta_tilde = jnp.asarray(beta * n_data, dtype=dtype)  # n*beta for loss gradient
    gamma_val = jnp.asarray(gamma, dtype=dtype)  # gamma for prior/localization

    # Setup keys
    key, init_key, sample_key = jax.random.split(key, 3)
    init_keys = jax.random.split(init_key, num_chains)
    sample_keys = jax.random.split(sample_key, num_chains)

    # Initialize states (position + preconditioner) in FLAT SPACE
    def init_chain(k: jax.random.PRNGKey, flat_init: Array):
        # Perturb initial position slightly for diversity
        noise = jax.random.normal(k, flat_init.shape, dtype=dtype)
        position = flat_init + 0.01 * noise
        precond_state = initialize_preconditioner(position)
        return SGLDState(position=position, precond_state=precond_state)

    # Vmap initialization across chains
    initial_states = jax.vmap(init_chain, in_axes=(0, None))(init_keys, flat0)

    # Build the pSGLD kernel in FLAT SPACE
    def sgld_kernel(rng_key, state: SGLDState):
        key_batch, key_sgld = jax.random.split(rng_key)
        flat_t = state.position
        precond_state = state.precond_state

        # 1. Sample minibatch
        indices = jax.random.choice(key_batch, n_data, shape=(batch_size,), replace=True)
        minibatch = (X[indices].astype(dtype), Y[indices].astype(dtype))

        # 2. Compute FLAT gradient of minibatch loss
        grad_loss = grad_loss_minibatch(flat_t, minibatch)

        # 3. Update preconditioner (uses only loss gradient)
        new_precond_state, P_t, adapted_loss_drift = update_preconditioner(
            config, grad_loss, precond_state
        )

        # 4. Compute SGLD drift (negative gradient of log posterior with minibatch):
        # drift = -grad_logpost = gamma*(flat_t - flat0) + n*beta*adapted_loss_drift
        total_drift_raw = gamma_val * (flat_t - flat0) + beta_tilde * adapted_loss_drift

        # Apply preconditioning: epsilon_t = epsilon * P_t
        adaptive_step = base_step_size * P_t
        update = -0.5 * adaptive_step * total_drift_raw
        noise = jax.random.normal(key_sgld, flat_t.shape, dtype=dtype)
        flat_next = flat_t + update + jnp.sqrt(adaptive_step) * noise

        new_state = SGLDState(position=flat_next, precond_state=new_precond_state)
        info = SGLDInfo()

        return new_state, info

    # Define aux_fn for recording loss
    def aux_fn(state):
        if loss_full_fn is not None:
            # Convert flat position back to model for loss computation
            model = posterior.vm.to_model(state.position)
            return {"Ln": loss_full_fn(model)}
        else:
            return {"Ln": jnp.nan}

    # Calculate work per step (FGEs): SGLD uses minibatch
    work_per_step = float(batch_size) / float(n_data)

    # Use eval_every from config if available, otherwise default to 10
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

    # Compute LLC from Ln if L0 provided
    if L0 is not None and "Ln" in traces:
        traces["llc"] = float(n_data) * float(beta) * (traces["Ln"] - L0)

    # Compute work: number of full loss evals + minibatch gradient FGEs
    n_Ln_evals = traces["Ln"].shape[1] if "Ln" in traces else 0
    work = {
        "n_full_loss": float(n_Ln_evals * num_chains),
        "n_minibatch_grads": float(config.steps * batch_size / n_data),
        "sampler_flavour": "markov",
    }

    return SamplerRunResult(traces=traces, timings=timings, work=work)
