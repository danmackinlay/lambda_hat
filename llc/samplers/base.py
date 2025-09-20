# llc/samplers/base.py
"""
Batched-only driver. `BatchedResult` carries small warmup/tuner scalars so runners can produce
WNV and FDE without per-step hooks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Optional, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

Array = jnp.ndarray



@dataclass
class BatchedResult:
    """Result from batched chain execution using drive_chains_batched"""

    kept: Array  # (C, K, k) tiny traces (K kept points)
    L_hist: Array  # (C, M) Ln at eval points (M evals)
    extras: Dict[str, Array]  # optional per-step/chain scalars (C, M_extras)
    mean_L: Array  # (C,) running mean over eval points
    var_L: Array  # (C,) running var over eval points (Welford)
    n_L: Array  # (C,) count of eval points
    eval_time_seconds: float  # fill on host if you want, else 0.0
    warmup_time_seconds: float = 0.0  # time spent in warmup/tuning
    warmup_grads: int = 0  # gradient evaluations in warmup


@dataclass
class SamplerSpec:
    """Interface for extensible sampler integration"""

    name: str
    step_vmapped: Callable[[jax.Array, Any], tuple[Any, Any] | Any]
    position_fn: Callable[[Any], jax.Array]
    info_extractors: Dict[str, Callable[[Any], jax.Array]] = None
    grads_per_step: float = 1.0


@dataclass
class DiagPrecondState:
    """State for diagonal preconditioning (RMSProp/Adam)"""

    m: Array  # first moment (Adam)
    v: Array  # second moment (RMSProp/Adam)
    t: Array  # time (Adam bias correction)


def precond_update(
    g: Array,
    st: DiagPrecondState,
    mode: str,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bias_correction: bool = True,
) -> tuple[Array, DiagPrecondState, Array]:
    """
    Unified diagonal preconditioning update for all samplers.

    Args:
        g: Gradient
        st: Current preconditioning state
        mode: "none", "rmsprop", or "adam"
        beta1: Adam first moment decay
        beta2: Adam/RMSProp second moment decay
        eps: Numerical stabilizer
        bias_correction: Whether to apply Adam bias correction

    Returns:
        inv_sqrt: Inverse square root of diagonal metric
        new_state: Updated preconditioning state
        drift_moment: Moment vector to use for drift calculation
    """
    if mode == "none":
        return jnp.ones_like(g), st, g

    if mode == "rmsprop":
        v_new = beta2 * st.v + (1.0 - beta2) * (g * g)
        inv_sqrt = jax.lax.rsqrt(v_new + eps)
        return inv_sqrt, DiagPrecondState(st.m, v_new, st.t), g  # drift uses raw g

    # adam
    m_new = beta1 * st.m + (1.0 - beta1) * g
    v_new = beta2 * st.v + (1.0 - beta2) * (g * g)

    if bias_correction:
        t_new = st.t + 1.0
        m_hat = m_new / (1.0 - beta1**t_new)
        v_hat = v_new / (1.0 - beta2**t_new)
    else:
        t_new = st.t
        m_hat, v_hat = m_new, v_new

    inv_sqrt = jax.lax.rsqrt(v_hat + eps)
    return inv_sqrt, DiagPrecondState(m_new, v_new, t_new), m_hat


def select_diag_dims(dim, k, seed):
    """Select k random dimensions from d for subset diagnostics"""
    k = min(k, dim)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dim, size=k, replace=False)).astype(int)


def make_projection_matrix(dim, k, seed):
    """Create k random unit vectors for projection diagnostics"""
    k = min(k, dim)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((k, dim)).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-8
    return R  # (k, d)


def prepare_diag_targets(dim, cfg):
    """Prepare diagnostic targets based on config"""
    if cfg.diag_mode == "subset":
        return dict(diag_dims=select_diag_dims(dim, cfg.diag_k, cfg.diag_seed))
    elif cfg.diag_mode == "proj":
        return dict(Rproj=make_projection_matrix(dim, cfg.diag_k, cfg.diag_seed))
    return {}  # none


def build_tiny_store(diag_dims=None, Rproj=None):
    """
    Return a vmapped extractor: theta[C,d] -> tiny[C,k], or None.

    This centralizes the tiny-store logic used across all samplers.
    """
    if diag_dims is not None:
        return lambda vec_batch: vec_batch[:, diag_dims]
    if Rproj is not None:
        return jax.vmap(lambda v: Rproj @ v)
    return None


def drive_chains_batched(
    *,
    rng_keys: Array,  # (T, C) or (T, C, ...), one key per step & chain
    init_state: Any,  # pytree with leading chain axis: (C, …)
    step_fn_vmapped: Callable[
        [Array, Any], Tuple[Any, Any]
    ],  # (keys[C], state[C]) -> (state[C], info[C] or None)
    n_steps: int,
    warmup: int = 0,
    eval_every: int = 1,
    thin: int = 1,
    position_fn: Callable[[Any], Array],  # state[C,…] -> theta[C, d]
    Ln_eval_f64_vmapped: Callable[[Array], Array],  # theta64[C,d] -> Ln[C]
    tiny_store_fn: Callable[[Array], Array] | None = None,  # theta[C,d] -> tiny[C,k]
    info_extractors: Dict[str, Callable[[Any], Array]]
    | None = None,  # info[C] -> scalar/vec per chain
) -> BatchedResult:
    """
    Run C chains in parallel with one compiled program:
      - scan over steps
      - vmap over chains
      - record Ln every eval_every after warmup
      - record tiny theta every 'thin' after warmup
    """
    C = jax.tree_util.tree_leaves(init_state)[0].shape[0]
    T = int(n_steps)
    assert rng_keys.shape[0] == T, "rng_keys must have shape (T, C, ...)"

    # How many eval/keep slots will we fill?
    M = jnp.maximum(0, (T - warmup + (eval_every - 1)) // eval_every)  # Ln points
    K = jnp.maximum(0, (T - warmup + (thin - 1)) // thin)  # tiny keep points

    # Determine tiny dimension by running tiny_store_fn once
    if tiny_store_fn is not None:
        sample_theta = position_fn(init_state)  # (C, d)
        sample_tiny = tiny_store_fn(sample_theta)  # (C, k)
        tiny_dim = sample_tiny.shape[-1]
        tiny_dtype = sample_tiny.dtype
    else:
        tiny_dim = 0
        tiny_dtype = jax.tree_util.tree_leaves(init_state)[0].dtype

    # Pre-allocate arrays we will fill inside scan
    L_hist0 = jnp.zeros((C, M), dtype=jnp.float64)
    kept0 = jnp.zeros((C, K, tiny_dim), dtype=tiny_dtype)  # Match the tiny dtype

    # Running mean/var counters per chain (Welford) - use float64 for Ln statistics
    mean0 = jnp.zeros((C,), dtype=jnp.float64)
    M20 = jnp.zeros((C,), dtype=jnp.float64)
    n0 = jnp.zeros((C,), dtype=jnp.int32)

    # If we want extras (e.g., HMC acceptance, MCLMC energy)
    info_extractors = info_extractors or {}
    extras0 = {name: jnp.zeros((C, M)) for name in info_extractors.keys()}

    def body(carry, t):
        state, L_hist, kept, mean, M2, n, idx_eval, idx_keep, extras = carry

        # One vmapped step across chains
        st_new, info = step_fn_vmapped(rng_keys[t], state)

        # After-warmup masks
        post = t >= warmup
        at_eval = post & (((t - warmup) % eval_every) == 0)
        at_keep = post & (((t - warmup) % thin) == 0)

        # Record Ln for all chains when at_eval
        def do_eval(args):
            st, L_hist, mean, M2, n, idx_eval, extras = args
            theta = position_fn(st)  # (C, d)
            Ln = Ln_eval_f64_vmapped(theta.astype(jnp.float64))  # (C,)

            # Welford update per chain
            n_new = n + 1
            delta = Ln - mean
            mean_new = mean + delta / n_new
            M2_new = M2 + delta * (Ln - mean_new)

            # Write into L_hist[:, idx_eval]
            L_hist = L_hist.at[:, idx_eval].set(Ln)

            # Write extras at eval index - functional update
            extras_new = {}
            for name, fn in info_extractors.items():
                v = fn(info)  # (C,) or (C, p)
                # If shape is (C,), write column; if (C,p), you might want a separate array per p.
                # For simplicity, assume (C,) scalar.
                E = extras[name]
                extras_new[name] = E.at[:, idx_eval].set(v)

            return (st, L_hist, mean_new, M2_new, n_new, idx_eval + 1, extras_new)

        def skip_eval(args):
            st, L_hist, mean, M2, n, idx_eval, extras = args
            return (st, L_hist, mean, M2, n, idx_eval, extras)

        (st_new, L_hist, mean, M2, n, idx_eval, extras) = lax.cond(
            at_eval,
            do_eval,
            skip_eval,
            operand=(st_new, L_hist, mean, M2, n, idx_eval, extras),
        )

        # Record tiny θ when at_keep
        def do_keep(args):
            st, kept, idx_keep = args
            if tiny_store_fn is not None:
                th = position_fn(st)  # (C, d)
                tiny = tiny_store_fn(th)  # (C, k)
                kept = kept.at[:, idx_keep, :].set(tiny)
            return (kept, idx_keep + 1)

        def skip_keep(args):
            st, kept, idx_keep = args
            return (kept, idx_keep)

        kept, idx_keep = lax.cond(
            at_keep, do_keep, skip_keep, operand=(st_new, kept, idx_keep)
        )

        return (st_new, L_hist, kept, mean, M2, n, idx_eval, idx_keep, extras), None

    # Keys per step already include per-chain split: rng_keys[t] has shape (C, ...)
    carry0 = (
        init_state,
        L_hist0,
        kept0,
        mean0,
        M20,
        n0,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        extras0,
    )

    (state_T, L_hist, kept, mean, M2, n, idx_eval, idx_keep, extras), _ = lax.scan(
        body, carry0, jnp.arange(T)
    )

    var = M2 / jnp.maximum(1, n - 1)
    return BatchedResult(
        kept=kept,
        L_hist=L_hist,
        extras=extras,
        mean_L=mean,
        var_L=var,
        n_L=n,
        eval_time_seconds=0.0,
    )


def run_sampler_spec(
    spec: SamplerSpec,
    *,
    rng_key: jax.Array,
    init_states: Any,
    n_steps: int,
    warmup: int = 0,
    eval_every: int = 1,
    thin: int = 1,
    Ln_eval_f64_vmapped: Callable[[Array], Array],
    tiny_store_fn: Callable[[Array], Array] | None = None,
) -> BatchedResult:
    """
    Generic batched runner using SamplerSpec interface.

    Args:
        spec: SamplerSpec defining the sampler behavior
        rng_key: Base random key
        init_states: Initial states for all chains (C, ...)
        n_steps: Total number of steps
        warmup: Warmup steps
        eval_every: Evaluate Ln every N steps after warmup
        thin: Keep tiny theta every N steps after warmup
        Ln_eval_f64_vmapped: Function to evaluate log-likelihood on batch
        tiny_store_fn: Optional function to extract subset/projection of theta

    Returns:
        BatchedResult with chains, evaluations, and diagnostics
    """
    C = jax.tree_util.tree_leaves(init_states)[0].shape[0]

    # Generate keys for all steps and chains
    keys_flat = jax.random.split(rng_key, n_steps * C)
    rng_keys = keys_flat.reshape(n_steps, C, -1)

    return drive_chains_batched(
        rng_keys=rng_keys,
        init_state=init_states,
        step_fn_vmapped=spec.step_vmapped,
        n_steps=n_steps,
        warmup=warmup,
        eval_every=eval_every,
        thin=thin,
        position_fn=spec.position_fn,
        Ln_eval_f64_vmapped=Ln_eval_f64_vmapped,
        tiny_store_fn=tiny_store_fn,
        info_extractors=spec.info_extractors or {},
    )
