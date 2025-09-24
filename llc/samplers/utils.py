from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

@dataclass
class DiagPrecondState:
    m: jnp.ndarray  # first moment (Adam)
    v: jnp.ndarray  # second moment (RMSProp/Adam)
    t: jnp.ndarray  # time (Adam bias correction)

# Register DiagPrecondState as a JAX pytree
def _diag_precond_flatten(state):
    return (state.m, state.v, state.t), None

def _diag_precond_unflatten(aux_data, children):
    m, v, t = children
    return DiagPrecondState(m=m, v=v, t=t)

jax.tree_util.register_pytree_node(
    DiagPrecondState,
    _diag_precond_flatten,
    _diag_precond_unflatten
)

def precond_update(g, st, mode, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
    """RMSProp/Adam preconditioning for SGLD."""
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

def build_tiny_store(diag_dims=None, Rproj=None):
    if diag_dims is not None:
        return lambda V: V[:, diag_dims]
    if Rproj is not None:
        return jax.vmap(lambda v: Rproj @ v)
    return None

@dataclass
class BatchedResult:
    """Result from batched chain execution"""
    kept: jnp.ndarray  # (C, K, k) tiny traces
    L_hist: jnp.ndarray  # (C, M) Ln at eval points
    extras: dict  # optional per-step/chain scalars
    mean_L: jnp.ndarray  # (C,) running mean
    var_L: jnp.ndarray  # (C,) running var
    n_L: jnp.ndarray  # (C,) count
    eval_time_seconds: float = 0.0
    warmup_time_seconds: float = 0.0
    warmup_grads: int = 0

def drive_chains_batched(*, rng_keys, init_state, step_fn_vmapped, n_steps, warmup,
                         eval_every, thin, position_fn, Ln_eval_f64_vmapped, tiny_store_fn, info_extractors):
    """Single-compile scan+vmap driver. Returns kept (C,K,k), L_hist (C,M), extras dict."""
    C = jax.tree_util.tree_leaves(init_state)[0].shape[0]
    T = int(n_steps)

    # How many eval/keep slots
    M = jnp.maximum(0, (T - warmup + (eval_every - 1)) // eval_every)
    K = jnp.maximum(0, (T - warmup + (thin - 1)) // thin)

    # Determine tiny dimension
    if tiny_store_fn is not None:
        sample_theta = position_fn(init_state)
        sample_tiny = tiny_store_fn(sample_theta)
        tiny_dim = sample_tiny.shape[-1]
        tiny_dtype = sample_tiny.dtype
    else:
        tiny_dim = 0
        tiny_dtype = jax.tree_util.tree_leaves(init_state)[0].dtype

    # Pre-allocate
    L_hist0 = jnp.zeros((C, M), dtype=jnp.float64)
    kept0 = jnp.zeros((C, K, tiny_dim), dtype=tiny_dtype)
    mean0 = jnp.zeros((C,), dtype=jnp.float64)
    M20 = jnp.zeros((C,), dtype=jnp.float64)
    n0 = jnp.zeros((C,), dtype=jnp.int32)

    info_extractors = info_extractors or {}
    extras0 = {name: jnp.zeros((C, M)) for name in info_extractors.keys()}

    def body(carry, t):
        state, L_hist, kept, mean, M2, n, idx_eval, idx_keep, extras = carry

        # One vmapped step
        st_new, info = step_fn_vmapped(rng_keys[t], state)

        # After-warmup masks
        post = t >= warmup
        at_eval = post & (((t - warmup) % eval_every) == 0)
        at_keep = post & (((t - warmup) % thin) == 0)

        # Record Ln
        def do_eval(args):
            st, L_hist, mean, M2, n, idx_eval, extras = args
            theta = position_fn(st)
            Ln = Ln_eval_f64_vmapped(theta.astype(jnp.float64))

            # Welford update
            n_new = n + 1
            delta = Ln - mean
            mean_new = mean + delta / n_new
            M2_new = M2 + delta * (Ln - mean_new)

            L_hist = L_hist.at[:, idx_eval].set(Ln)

            # Write extras
            extras_new = {}
            for name, fn in info_extractors.items():
                v = fn(info)
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

        # Record tiny θ
        def do_keep(args):
            st, kept, idx_keep = args
            if tiny_store_fn is not None:
                th = position_fn(st)
                tiny = tiny_store_fn(th)
                kept = kept.at[:, idx_keep, :].set(tiny)
            return (kept, idx_keep + 1)

        def skip_keep(args):
            st, kept, idx_keep = args
            return (kept, idx_keep)

        kept, idx_keep = lax.cond(
            at_keep, do_keep, skip_keep, operand=(st_new, kept, idx_keep)
        )

        return (st_new, L_hist, kept, mean, M2, n, idx_eval, idx_keep, extras), None

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
        warmup_time_seconds=0.0,
        warmup_grads=0,
    )