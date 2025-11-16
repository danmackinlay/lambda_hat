# lambda_hat/samplers/vi.py
"""Variational inference sampler - FLAT INTERFACE ONLY"""

import time
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from lambda_hat import vi
from lambda_hat.posterior import Posterior
from lambda_hat.types import SamplerRunResult
from lambda_hat.utils.rng import ensure_typed_key
from lambda_hat.vi.types import Batch, FlatObjective

if TYPE_CHECKING:
    from lambda_hat.config import VIConfig

Array = jnp.ndarray


class _FlatObjective:
    """Adapter implementing FlatObjective protocol.

    Wraps flat loss and gradient functions with optional value_and_grad.
    """

    def __init__(
        self,
        loss_fn: Callable[[Array, Batch], Array],
        grad_fn: Callable[[Array, Batch], Array],
    ):
        self._loss_fn = loss_fn
        self._grad_fn = grad_fn
        # Create value_and_grad via JAX (single-pass computation)
        self._vag = jax.value_and_grad(loss_fn)

    def loss(self, w_flat: Array, batch: Batch) -> Array:
        return self._loss_fn(w_flat, batch)

    def grad(self, w_flat: Array, batch: Batch) -> Array:
        return self._grad_fn(w_flat, batch)

    def value_and_grad(self, w_flat: Array, batch: Batch) -> Tuple[Array, Array]:
        return self._vag(w_flat, batch)


class VIState(NamedTuple):
    """Variational inference state"""

    params: Array  # VI parameters (flat vectors for mixture of factor analyzers)
    opt_state: Any  # Optax optimizer state
    step: jnp.ndarray  # Current optimization step


class VIInfo(NamedTuple):
    """Variational inference step info"""

    acceptance_rate: jnp.ndarray = jnp.array(1.0)  # Always 1.0 for VI (IID samples)
    energy: jnp.ndarray = jnp.nan  # Will store ELBO
    is_divergent: bool = False  # Always False for VI


def run_vi(
    key: jax.random.PRNGKey,
    posterior: Posterior,
    data: Tuple[jnp.ndarray, jnp.ndarray],
    config: "VIConfig",
    num_chains: int,
    loss_minibatch_flat: Callable[
        [Array, Tuple[Array, Array]], Array
    ],  # NEW: scalar value in flat space
    grad_loss_minibatch: Callable[[Array, Tuple[Array, Array]], Array],  # FLAT gradient
    loss_full_flat: Callable[
        [Array], Array
    ],  # NEW: scalar full-data loss in flat space (for Ln_wstar, Eq[L_n])
    n_data: Optional[int] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
) -> SamplerRunResult:
    """Variational inference sampler - FLAT INTERFACE ONLY

    Implements STL (sticking-the-landing) pathwise gradients for continuous params
    and Rao-Blackwellized score gradients for mixture weights.

    Args:
        key: JRNG key
        posterior: Posterior with flat-space interface
        data: Tuple of (X, Y)
        config: VIConfig with M, r, steps, batch_size, lr, etc.
        num_chains: Number of independent VI runs (for consistency checking)
        loss_minibatch_flat: Minibatch loss VALUE in FLAT space
        grad_loss_minibatch: Gradient of minibatch loss in FLAT space
        loss_full_flat: Full-data loss VALUE in FLAT space (for Ln_wstar, Eq[L_n])
        n_data: Number of data points
        beta: Inverse temperature (typically 1/log(n))
        gamma: Localizer strength

    Returns:
        SamplerRunResult with traces, timings, and work tracking
    """
    X, Y = data
    if n_data is None:
        n_data = X.shape[0]
    if beta is None or gamma is None:
        raise ValueError("beta and gamma must be provided for VI")

    dtype = posterior.vm.dtype

    # Start timer
    total_start_time = time.time()

    # Normalize PRNG key to typed threefry format (host-side, before any vmap/jit)
    # Required for FlowJAX compatibility - converts legacy uint32[2] keys and ints
    key = ensure_typed_key(key)

    # Get flat initial parameters from posterior
    params_flat = posterior.flat0

    # === Whitening Pre-Pass (Stage 1) ===
    # Compute diagonal preconditioner A_diag based on whitening_mode
    A_diag = None  # Default: identity whitener

    if config.whitening_mode != "none":
        # Estimate diagonal geometry via gradient moment accumulation
        # Use ~500-1000 minibatch gradients to build robust estimate
        n_whitening_samples = min(1000, max(500, config.steps // 10))

        # Initialize EMA state for gradient moments
        key, key_whitening = jax.random.split(key)
        v_diag = jnp.ones_like(params_flat)  # Second moment (RMSProp/Adam)
        m_diag = jnp.zeros_like(params_flat) if config.whitening_mode == "adam" else None

        # Accumulate gradient moments using flat gradient
        for _ in range(n_whitening_samples):
            key_whitening, key_batch = jax.random.split(key_whitening)
            indices = jax.random.choice(key_batch, n_data, shape=(config.batch_size,), replace=True)
            minibatch = (X[indices].astype(dtype), Y[indices].astype(dtype))

            # Compute FLAT gradient at w*
            grad_flat = grad_loss_minibatch(params_flat, minibatch)

            # Update moments with EMA
            decay = config.whitening_decay
            v_diag = decay * v_diag + (1 - decay) * (grad_flat**2)
            if config.whitening_mode == "adam":
                m_diag = decay * m_diag + (1 - decay) * grad_flat

        # Extract diagonal preconditioner (inverse of sqrt of second moment)
        # A_diag represents the geometry scaling: larger values where gradients are larger
        eps = 1e-8  # Numerical stability
        A_diag = jnp.sqrt(v_diag + eps)  # Diagonal of geometry matrix

    # Create whitener (identity if A_diag is None, diagonal otherwise)
    whitener = vi.make_whitener(A_diag)

    # Build flat objective for VI algorithms (flat-only interface)
    objective = _FlatObjective(loss_minibatch_flat, grad_loss_minibatch)

    # Compute Ln_wstar once at sampler level (algorithms no longer need full-loss callable)
    Ln_wstar = loss_full_flat(params_flat)

    # Run VI fitting and estimation for each chain
    chain_keys = jax.random.split(key, num_chains)

    # Unified dispatch: all algorithms go through registry
    algo = vi.get(config.algo)

    def run_one_chain(chain_key):
        result = algo.run(
            rng_key=chain_key,
            loss_batch_fn=loss_batch_fn_wrapped,
            loss_full_fn=loss_full_fn_wrapped,
            wstar_flat=params_flat,
            unravel_fn=unravel_fn,
            data=data,
            n_data=n_data,
            beta=beta,
            gamma=gamma,
            vi_cfg=config,
            whitener=whitener,  # Pass whitener from pre-pass
        )
        # All algorithms now return dict with same structure
        # Convert to tuple format for vmap compatibility
        lambda_hat = result["lambda_hat"]
        traces = result["traces"]
        extras = result["extras"]
        return (lambda_hat, traces, extras)

    # Run first chain separately to get timings and work dict
    first_result = algo.run(
        rng_key=chain_keys[0],
        loss_batch_fn=loss_batch_fn_wrapped,
        loss_full_fn=loss_full_fn_wrapped,
        wstar_flat=params_flat,
        unravel_fn=unravel_fn,
        data=data,
        n_data=n_data,
        beta=beta,
        gamma=gamma,
        vi_cfg=config,
        whitener=whitener,
    )
    algo_timings = first_result["timings"]
    algo_work = first_result["work"]

    # Validate that algorithm returns only vmap-compatible types (arrays/scalars)
    # Prevents errors like "PjitFunction is not a valid JAX type"
    def _all_leaves_are_arrays(x):
        leaves, _ = jax.tree_util.tree_flatten(x)
        return all(isinstance(leaf, (jax.Array, jnp.ndarray, int, float, bool)) for leaf in leaves)

    assert _all_leaves_are_arrays(
        (first_result["lambda_hat"], first_result["traces"], first_result["extras"])
    ), f"VI algorithm '{config.algo}' returned non-JAX objects; see docs/flow_vmap_issues.md"

    # Vmap across chains: returns (lambda_hats, all_traces, all_extras)
    results = jax.vmap(run_one_chain)(chain_keys)
    lambda_hats, all_traces, all_extras = results

    # Ensure computation is complete
    jax.block_until_ready(results)
    total_time = time.time() - total_start_time

    # Extract final MC estimates (same across all chains for Ln_wstar)
    Eq_Ln_values = all_extras["Eq_Ln"]  # shape: (num_chains,)
    Ln_wstar = all_extras["Ln_wstar"][0]  # scalar, same for all chains

    # Format traces for compatibility with existing analysis code
    # Note: VI traces come from optimization metrics, not periodic Ln evaluation

    # Extract CV metrics (scalars, one per chain) and replicate across steps for trace format
    # After vmap, cv_info is a dict where each key maps to array of shape (num_chains,)
    cv_info_vmapped = all_extras["cv_info"]
    Eq_Ln_mc_vals = cv_info_vmapped["Eq_Ln_mc"]  # (num_chains,)
    Eq_Ln_cv_vals = cv_info_vmapped["Eq_Ln_cv"]  # (num_chains,)
    var_red_vals = cv_info_vmapped["variance_reduction"]  # (num_chains,)

    # Create trace format: replicate scalar values across all steps
    steps_shape = all_traces["elbo"].shape[1]  # Number of optimization steps
    Eq_Ln_mc_trace = jnp.repeat(Eq_Ln_mc_vals[:, None], steps_shape, axis=1)
    Eq_Ln_cv_trace = jnp.repeat(Eq_Ln_cv_vals[:, None], steps_shape, axis=1)
    var_red_trace = jnp.repeat(var_red_vals[:, None], steps_shape, axis=1)

    traces = {
        # MCMC-compatible keys
        "Ln": jnp.full_like(
            all_traces["elbo"], jnp.nan
        ),  # Placeholder - NaN to fail fast if used incorrectly (VI doesn't sample Ln)
        "cumulative_fge": all_traces["cumulative_fge"],
        "acceptance_rate": jnp.ones_like(all_traces["elbo"], dtype=dtype),
        "energy": all_traces["elbo"],  # ELBO serves as "energy"
        "elbo": all_traces["elbo"],  # VI-specific ELBO trace (total objective)
        "is_divergent": jnp.zeros_like(all_traces["elbo"], dtype=bool),
        # Control variate metrics (replicated across steps for observability)
        "Eq_Ln_mc": Eq_Ln_mc_trace,  # Raw MC estimate of E_q[L_n]
        "Eq_Ln_cv": Eq_Ln_cv_trace,  # CV-corrected estimate of E_q[L_n]
        "variance_reduction": var_red_trace,  # Variance reduction factor from CV
        # Common diagnostics (all VI algorithms provide these)
        "grad_norm": all_traces["grad_norm"],
        # Algorithm-specific traces (only include if present)
        **{
            k: all_traces[k]
            for k in [
                "elbo_like",
                "logq",
                "radius2",
                "resp_entropy",  # MFA-specific
                "pi_min",
                "pi_max",
                "pi_entropy",  # MFA mixture weights
                "D_sqrt_min",
                "D_sqrt_max",
                "D_sqrt_med",  # MFA covariance
                "A_col_norm_max",  # MFA low-rank factor
                "d_latent",
                "sigma_perp",  # Flow-specific
            ]
            if k in all_traces
        },
    }

    # Use timings from algorithm and override total with actual wall time
    timings = {
        "adaptation": algo_timings["adaptation"],
        "sampling": algo_timings["sampling"],
        "total": total_time,  # Override with actual wall time (includes all chains)
    }

    # Work tracking: include VI-specific estimates
    # Compute LLC (Local Learning Coefficient) from VI estimates
    # Shape: (num_chains,) -> replicate across all optimization steps for trace format
    steps_shape = traces["elbo"].shape[1]
    llc_per_chain = lambda_hats  # (num_chains,) - already computed above
    llc_trace = jnp.repeat(llc_per_chain[:, None], steps_shape, axis=1)  # (num_chains, steps)
    traces["llc"] = llc_trace

    # Use work dict from algorithm and augment with VI-specific cross-chain statistics
    work = {
        # Core work metrics from algorithm (preserve sampler_flavour)
        **algo_work,
        # VI-specific outputs (for analysis.json)
        "lambda_hat_mean": float(jnp.mean(lambda_hats)),
        "lambda_hat_std": float(jnp.std(lambda_hats)),
        "Eq_Ln_mean": float(jnp.mean(Eq_Ln_values)),
        "Eq_Ln_std": float(jnp.std(Eq_Ln_values)),
        "Ln_wstar": float(Ln_wstar),
        # Audit trail for LLC magnitude investigation
        "beta": float(beta),
        "n_data": int(n_data),
    }

    return SamplerRunResult(traces=traces, timings=timings, work=work)
