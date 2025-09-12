# llc/samplers/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Optional, Dict, List, Tuple
import time

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from tqdm.auto import tqdm

Array = jnp.ndarray


@dataclass
class ChainResult:
    kept: np.ndarray  # (draws, k) tiny theta traces (or (0,k) if none)
    L_hist: np.ndarray  # (draws_L,) Ln evaluations
    extras: Dict[str, np.ndarray]  # e.g. {"accept": (draws,), "energy": (draws,)}
    mean_L: float
    var_L: float
    n_L: int
    eval_time_seconds: float  # time spent in L_n evaluations


@dataclass
class BatchedResult:
    """Result from batched chain execution using drive_chains_batched"""
    kept: Array          # (C, K, k) tiny traces (K kept points)
    L_hist: Array        # (C, M) Ln at eval points (M evals)
    extras: Dict[str, Array]  # optional per-step/chain scalars (C, M_extras)
    mean_L: Array        # (C,) running mean over eval points
    var_L: Array         # (C,) running var over eval points (Welford)
    n_L: Array           # (C,) count of eval points
    eval_time_seconds: float  # fill on host if you want, else 0.0


@dataclass
class SamplerSpec:
    """Interface for extensible sampler integration"""
    name: str
    step_vmapped: Callable[[jax.Array, Any], tuple[Any, Any] | Any]
    position_fn: Callable[[Any], jax.Array]
    info_extractors: Dict[str, Callable[[Any], jax.Array]] = None
    grads_per_step: float = 1.0


@dataclass
class RunSummary:
    chains: List[ChainResult]
    kept_stacked: np.ndarray  # (C, draws, k) or (C, 0, k) if no storage
    L_hist_stacked: np.ndarray  # (C, draws_L)
    extras: Dict[str, List[np.ndarray]]  # per-chain lists
    eval_time_seconds: (
        float  # time spent in Ln() (so you can subtract from sampling time)
    )


class RunningMeanVar:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def value(self) -> Tuple[float, float, int]:
        var = self.M2 / (self.n - 1) if self.n > 1 else np.nan
        return float(self.mean), float(var), int(self.n)




def default_tiny_store(
    vec: np.ndarray, diag_dims=None, Rproj=None
) -> Optional[np.ndarray]:
    if diag_dims is not None:
        return vec[diag_dims]
    if Rproj is not None:
        return Rproj @ vec
    return None


def drive_chain(
    *,
    rng_key: Array,
    init_state: Any,
    step_fn: Callable[[Array, Any], Tuple[Any, Any] | Any],
    # If step_fn returns only `state` (SGLD 1.2.5), set `step_returns_info=False`
    step_returns_info: bool,
    n_steps: int,
    warmup: int = 0,
    eval_every: int = 1,
    thin: int = 1,
    # State accessors
    position_fn: Callable[[Any], Array],
    # L_n evaluator (float64): Array[dim] -> scalar
    Ln_eval_f64: Callable[[Array], Array],
    # Tiny storage (subset/projection)
    tiny_store_fn: Callable[[np.ndarray], Optional[np.ndarray]] = default_tiny_store,
    # Progress & timing callbacks
    use_tqdm: bool = True,
    progress_label: str = "",
    progress_update_every: int = 50,
    # Hooks: called every step after step_fn
    info_hooks: List[Callable[[Any, Dict[str, Any]], None]] | None = None,
) -> ChainResult:
    """
    Generic single-chain driver:
      - runs warmup + sampling,
      - periodically evaluates full-data L_n for LLC,
      - stores thinned tiny θ for diagnostics,
      - records sampler-specific extras via hooks.
    """
    info_hooks = info_hooks or []
    rm = RunningMeanVar()
    kept: List[np.ndarray] = []
    Lhist: List[float] = []
    extras_acc: Dict[str, List[float]] = {}  # e.g., {"accept": [...], "energy": [...]}
    eval_time = 0.0

    # warmup boundary handled by `t == warmup`
    keys = jax.random.split(rng_key, n_steps)
    state = init_state

    # Prime extras dict for known keys lazily
    def put_extra(name: str, value: float):
        if name not in extras_acc:
            extras_acc[name] = []
        extras_acc[name].append(float(value))

    rng = range(n_steps)
    pbar = (
        tqdm(rng, total=n_steps, desc=progress_label, leave=False) if use_tqdm else rng
    )

    # Helper: record Ln and tiny θ after warmup only
    def record_if_needed(t: int):
        nonlocal eval_time
        if t < warmup:
            return
        # L_n
        if ((t - warmup) % eval_every) == 0:
            t0 = time.time()
            Ln = float(
                jax.device_get(Ln_eval_f64(position_fn(state).astype(jnp.float64)))
            )
            eval_time += time.time() - t0
            rm.update(Ln)
            Lhist.append(Ln)
        # tiny θ
        if ((t - warmup) % thin) == 0:
            vec = np.array(position_fn(state))
            s = tiny_store_fn(vec)
            if s is not None:
                kept.append(s)

    # First step
    out = step_fn(keys[0], state)
    if step_returns_info:
        state, info = out
    else:
        state, info = out, None
    record_if_needed(0)
    # Hooks
    if info is not None:
        ctx = {"put_extra": put_extra}
        for h in info_hooks:
            h(info, ctx)
    if use_tqdm:
        meanL = rm.value()[0] if rm.n > 0 else float("nan")
        pbar.set_postfix_str(f"L̄≈{meanL:.4f}")
        pbar.update(1)

    # Remaining steps
    for t in range(1, n_steps):
        out = step_fn(keys[t], state)
        if step_returns_info:
            state, info = out
        else:
            state, info = out, None
        record_if_needed(t)
        if info is not None:
            ctx = {"put_extra": put_extra}
            for h in info_hooks:
                h(info, ctx)
        if use_tqdm and (t % progress_update_every == 0 or t == n_steps - 1):
            meanL = rm.value()[0] if rm.n > 0 else float("nan")
            pbar.set_postfix_str(f"L̄≈{meanL:.4f}")
        if use_tqdm:
            pbar.update(1)

    if use_tqdm and hasattr(pbar, "close"):
        pbar.close()

    m, v, n = rm.value()
    # Build extras dict -> arrays
    extras: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in extras_acc.items()}

    # Shape for empty kept tensor
    kdim = kept[0].shape[-1] if kept else 0
    kept_arr = np.stack(kept, 0) if kept else np.empty((0, kdim))
    return ChainResult(
        kept=kept_arr,
        L_hist=np.asarray(Lhist),
        extras=extras,
        mean_L=m,
        var_L=v,
        n_L=n,
        eval_time_seconds=float(eval_time),
    )


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


def make_tiny_store(dim: int, config) -> tuple[Callable, Any]:
    """
    Create a function to store subset or projection of parameters.

    Returns:
        tiny_store_fn: Function that extracts subset/projection
        targets: The target indices or projection matrix
    """
    if hasattr(config, "track_subset_indices") and config.track_subset_indices:
        # Track specific parameter indices
        indices = config.track_subset_indices
        if max(indices) >= dim:
            indices = list(range(min(10, dim)))  # Fallback to first 10
        targets = jnp.array(indices)

        def tiny_store_fn(theta):
            return theta[targets]

        return tiny_store_fn, targets

    elif hasattr(config, "track_projection_dim") and config.track_projection_dim:
        # Random projection
        proj_dim = config.track_projection_dim
        key = jax.random.PRNGKey(config.seed)
        projection = jax.random.normal(key, (proj_dim, dim)) / jnp.sqrt(proj_dim)
        targets = projection

        def tiny_store_fn(theta):
            return projection @ theta

        return tiny_store_fn, targets

    else:
        # No tracking
        return lambda x: None, None


def drive_chains_batched(
    *,
    rng_keys: Array,                     # (T, C) or (T, C, ...), one key per step & chain
    init_state: Any,                     # pytree with leading chain axis: (C, …)
    step_fn_vmapped: Callable[[Array, Any], Tuple[Any, Any]],  # (keys[C], state[C]) -> (state[C], info[C] or None)
    n_steps: int,
    warmup: int = 0,
    eval_every: int = 1,
    thin: int = 1,
    position_fn: Callable[[Any], Array], # state[C,…] -> theta[C, d]
    Ln_eval_f64_vmapped: Callable[[Array], Array],  # theta64[C,d] -> Ln[C]
    tiny_store_fn: Callable[[Array], Array] | None = None,  # theta[C,d] -> tiny[C,k]
    info_extractors: Dict[str, Callable[[Any], Array]] | None = None,  # info[C] -> scalar/vec per chain
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
    K = jnp.maximum(0, (T - warmup + (thin - 1)) // thin)              # tiny keep points

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
    L_hist0   = jnp.zeros((C, M), dtype=jnp.float64)
    kept0     = jnp.zeros((C, K, tiny_dim), dtype=tiny_dtype)  # Match the tiny dtype

    # Running mean/var counters per chain (Welford) - use float64 for Ln statistics
    mean0 = jnp.zeros((C,), dtype=jnp.float64)
    M20   = jnp.zeros((C,), dtype=jnp.float64)
    n0    = jnp.zeros((C,), dtype=jnp.int32)

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
            theta = position_fn(st)                          # (C, d)
            Ln    = Ln_eval_f64_vmapped(theta.astype(jnp.float64))  # (C,)

            # Welford update per chain
            n_new    = n + 1
            delta    = Ln - mean
            mean_new = mean + delta / n_new
            M2_new   = M2 + delta * (Ln - mean_new)

            # Write into L_hist[:, idx_eval]
            L_hist = L_hist.at[:, idx_eval].set(Ln)

            # Write extras at eval index
            for name, fn in info_extractors.items():
                v = fn(info)  # (C,) or (C, p)
                # If shape is (C,), write column; if (C,p), you might want a separate array per p.
                # For simplicity, assume (C,) scalar.
                E = extras[name]
                E = E.at[:, idx_eval].set(v)
                extras[name] = E

            return (st, L_hist, mean_new, M2_new, n_new, idx_eval + 1, extras)

        def skip_eval(args):
            st, L_hist, mean, M2, n, idx_eval, extras = args
            return (st, L_hist, mean, M2, n, idx_eval, extras)

        (st_new, L_hist, mean, M2, n, idx_eval, extras) = lax.cond(
            at_eval, do_eval, skip_eval,
            operand=(st_new, L_hist, mean, M2, n, idx_eval, extras)
        )

        # Record tiny θ when at_keep
        def do_keep(args):
            st, kept, idx_keep = args
            if tiny_store_fn is not None:
                th = position_fn(st)                     # (C, d)
                tiny = tiny_store_fn(th)                 # (C, k)
                kept = kept.at[:, idx_keep, :].set(tiny)
            return (kept, idx_keep + 1)

        def skip_keep(args):
            st, kept, idx_keep = args
            return (kept, idx_keep)

        kept, idx_keep = lax.cond(
            at_keep, do_keep, skip_keep,
            operand=(st_new, kept, idx_keep)
        )

        return (st_new, L_hist, kept, mean, M2, n, idx_eval, idx_keep, extras), None

    # Keys per step already include per-chain split: rng_keys[t] has shape (C, ...)
    carry0 = (init_state, L_hist0, kept0, mean0, M20, n0,
              jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32), extras0)

    (state_T, L_hist, kept, mean, M2, n, idx_eval, idx_keep, extras), _ = lax.scan(
        body, carry0, jnp.arange(T)
    )

    var = M2 / jnp.maximum(1, n - 1)
    return BatchedResult(
        kept=kept, L_hist=L_hist, extras=extras,
        mean_L=mean, var_L=var, n_L=n, eval_time_seconds=0.0
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
        info_extractors=spec.info_extractors or {}
    )
