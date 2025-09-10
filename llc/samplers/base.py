# llc/samplers/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Optional, Dict, List, Tuple
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

Array = jnp.ndarray

@dataclass
class ChainResult:
    kept: np.ndarray               # (draws, k) tiny theta traces (or (0,k) if none)
    L_hist: np.ndarray             # (draws_L,) Ln evaluations
    extras: Dict[str, np.ndarray]  # e.g. {"accept": (draws,), "energy": (draws,)}
    mean_L: float
    var_L: float
    n_L: int
    eval_time_seconds: float       # time spent in L_n evaluations

@dataclass
class RunSummary:
    chains: List[ChainResult]
    kept_stacked: np.ndarray       # (C, draws, k) or (C, 0, k) if no storage
    L_hist_stacked: np.ndarray     # (C, draws_L)
    extras: Dict[str, List[np.ndarray]]  # per-chain lists
    eval_time_seconds: float       # time spent in Ln() (so you can subtract from sampling time)

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

def stack_thinned_per_chain(ch_kept: List[List[np.ndarray]], empty_shape: Tuple[int, ...]) -> np.ndarray:
    """Return (C, draws, k) without NaN padding (truncate to per-chain length)."""
    out = []
    for kept in ch_kept:
        out.append(np.stack(kept, 0) if kept else np.empty(empty_shape))
    # Truncate to common draw count across chains
    if not out:
        return np.empty((0, 0, 0))
    m = min(k.shape[0] for k in out)
    if m == 0:
        # preserve k
        kdim = out[0].shape[-1] if out[0].ndim == 2 else 0
        return np.empty((len(out), 0, kdim))
    return np.stack([k[:m] for k in out], axis=0)

def default_tiny_store(vec: np.ndarray, diag_dims=None, Rproj=None) -> Optional[np.ndarray]:
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
    pbar = tqdm(rng, total=n_steps, desc=progress_label, leave=False) if use_tqdm else rng

    # Helper: record Ln and tiny θ after warmup only
    def record_if_needed(t: int):
        nonlocal eval_time
        if t < warmup:
            return
        # L_n
        if ((t - warmup) % eval_every) == 0:
            t0 = time.time()
            Ln = float(jax.device_get(Ln_eval_f64(position_fn(state).astype(jnp.float64))))
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
        meanL = rm.value()[0] if rm.n > 0 else float('nan')
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