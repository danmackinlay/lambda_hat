# lambda_hat/samplers/sgld.py
"""SGLD (Stochastic Gradient Langevin Dynamics) with preconditioning"""

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, NamedTuple, Optional, Tuple

import blackjax
import jax
import jax.numpy as jnp

from lambda_hat.samplers.common import inference_loop_extended
from lambda_hat.types import SamplerRunResult

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
    L0: Optional[float] = None,
) -> SamplerRunResult:
    """Run SGLD or pSGLD (AdamSGLD/RMSPropSGLD) with minibatching."""
    X, Y = data
    # Convert to JAX arrays if they're NumPy arrays (needed for indexing in JIT)
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)
    n_data = X.shape[0]
    batch_size = config.batch_size
    base_step_size = float(config.step_size)

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

        # 1. Sample minibatch (potentially in f64)
        indices = jax.random.choice(key_batch, n_data, shape=(batch_size,), replace=True)
        minibatch_raw = (X[indices], Y[indices])

        # 1.5 Cast minibatch to required dtype (e.g., f32 for SGLD)
        minibatch = jax.tree.map(lambda x: x.astype(ref_dtype), minibatch_raw)

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
    # n_full_loss: recorded Ln evaluations (happens every eval_every steps)
    n_Ln_evals = traces["Ln"].shape[1] if "Ln" in traces else 0
    work = {
        "n_full_loss": float(n_Ln_evals * num_chains),
        "n_minibatch_grads": float(config.steps * batch_size / n_data),
        "sampler_flavour": "markov",
    }

    return SamplerRunResult(traces=traces, timings=timings, work=work)


def run_sgld_basic(
    rng_key: jax.random.PRNGKey,
    grad_loss_fn: Callable,  # gradient of mean minibatch loss
    initial_params: Dict[str, Any],
    params0: Dict[str, Any],  # ERM center w0
    data: Tuple[jnp.ndarray, jnp.ndarray],
    config: "SGLDConfig",
    num_chains: int,
    beta: float,
    gamma: float,
    loss_full_fn: Optional[Callable] = None,  # for Ln recording
) -> SamplerRunResult:
    """Reference SGLD using BlackJAX's sgld kernel (no preconditioning)."""
    if loss_full_fn is None:
        raise ValueError("loss_full_fn must be provided for Ln recording in SGLD basic.")
    X, Y = data
    n_data = X.shape[0]
    ref_dtype = jax.tree_util.tree_leaves(initial_params)[0].dtype
    beta_tilde = jnp.asarray(beta * n_data, dtype=ref_dtype)
    gamma_val = jnp.asarray(gamma, dtype=ref_dtype)
    batch_size = config.batch_size

    # Gradient estimator of log posterior: -(γ(w-w0) + nβ * grad_Lmini)
    def grad_logpost_estimator(w, minibatch):
        Xb, Yb = minibatch
        Xb = Xb.astype(ref_dtype)
        Yb = Yb.astype(ref_dtype)
        gL = grad_loss_fn(w, (Xb, Yb))  # gradient of mean minibatch loss
        # -(gamma*(w-w0) + n*beta*gL)
        return jax.tree.map(
            lambda wi, w0i, gi: -(gamma_val * (wi - w0i) + beta_tilde * gi), w, params0, gL
        )

    sgld = blackjax.sgld(grad_logpost_estimator)

    # Init chains (jitter the start a bit, like run_sgld)
    key, init_key, sample_key = jax.random.split(rng_key, 3)
    init_keys = jax.random.split(init_key, num_chains)

    def init_chain(k, p):
        noise = jax.tree.map(lambda q: 0.01 * jax.random.normal(k, q.shape, q.dtype), p)
        return jax.tree.map(lambda q, n: q + n, p, noise)

    init_positions = jax.vmap(init_chain, in_axes=(0, None))(init_keys, initial_params)

    # Kernel: one step of BlackJAX SGLD with our minibatch
    def sgld_basic_kernel(rng_key, position):
        key_batch, key_step = jax.random.split(rng_key)
        replace_flag = batch_size > n_data
        idx = jax.random.choice(key_batch, n_data, shape=(batch_size,), replace=replace_flag)
        minibatch = (X[idx], Y[idx])
        new_position = sgld.step(key_step, position, minibatch, config.step_size)
        # Ensure dtype consistency - cast back to ref_dtype if needed
        new_position = jax.tree.map(lambda x: jnp.asarray(x, dtype=ref_dtype), new_position)
        # Return position as "state" and a lightweight info
        return new_position, SGLDInfo()

    # Aux: Ln(position) - ensure dtype consistency
    loss_full_fn_jitted = jax.jit(loss_full_fn)

    def aux_fn(position):
        Ln = loss_full_fn_jitted(position)
        # Cast to ref_dtype to ensure consistency
        return {"Ln": jnp.asarray(Ln, dtype=ref_dtype)}

    # Work per step (FGEs)
    work_per_step = float(batch_size) / float(n_data)
    eval_every = config.eval_every

    # Drive chains
    chain_keys = jax.random.split(sample_key, num_chains)
    sampling_start_time = time.time()
    traces = jax.vmap(
        lambda k, s: inference_loop_extended(
            k,
            sgld_basic_kernel,
            s,
            num_samples=config.steps,
            aux_fn=aux_fn,
            aux_every=eval_every,
            work_per_step=work_per_step,
        )
    )(chain_keys, init_positions)
    jax.block_until_ready(traces)
    sampling_time = time.time() - sampling_start_time

    timings = {"adaptation": 0.0, "sampling": sampling_time, "total": sampling_time}

    # Compute work for SGLD basic (same as run_sgld)
    n_Ln_evals = traces["Ln"].shape[1] if "Ln" in traces else 0
    work = {
        "n_full_loss": float(n_Ln_evals * num_chains),
        "n_minibatch_grads": float(config.steps * batch_size / n_data),
    }

    return SamplerRunResult(traces=traces, timings=timings, work=work)
