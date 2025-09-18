# llc/convert.py
from __future__ import annotations
from typing import List, Optional
import numpy as np


def _az():
    import arviz as az

    return az


def stack_ragged_2d(arrs: List[np.ndarray]) -> Optional[np.ndarray]:
    """Truncate ragged list of (T_i, D) to (C, T, D). Returns None if empty."""
    arrs = [
        np.asarray(a)
        for a in arrs
        if a is not None and a.ndim == 2 and a.shape[0] >= 2 and a.shape[1] >= 1
    ]
    if not arrs:
        return None
    t = min(a.shape[0] for a in arrs)
    d = arrs[0].shape[1]
    if t < 2 or d < 1:
        return None
    return np.stack([a[:t, :d] for a in arrs], axis=0)  # (C, T, D)


def stack_ragged_1d(arrs: List[np.ndarray]) -> Optional[np.ndarray]:
    """Truncate ragged list of (T_i,) to (C, T)."""
    arrs = [
        np.asarray(a).reshape(-1)
        for a in arrs
        if a is not None and np.asarray(a).size >= 2
    ]
    if not arrs:
        return None
    t = min(a.shape[0] for a in arrs)
    if t < 2:
        return None
    return np.stack([a[:t] for a in arrs], axis=0)  # (C, T)


def to_idata(
    *,
    Ln_histories: List[np.ndarray],
    theta_thin: Optional[List[np.ndarray] | np.ndarray],
    acceptance: Optional[List[np.ndarray]],
    energy: Optional[List[np.ndarray]],
    n: int,
    beta: float,
    L0: float,
    max_theta_dims: int = 8,
) -> "az.InferenceData":
    """Build a single ArviZ InferenceData with:
    posterior: llc (C,T), L (C,T), optional theta (C,T,d')
    sample_stats: acceptance_rate (C,T), energy (C,T)
    """
    az = _az()

    # L traces (C,T)
    if not Ln_histories or all(len(h) == 0 for h in Ln_histories):
        raise ValueError("Ln_histories is empty; cannot create InferenceData.")
    H = stack_ragged_1d([np.asarray(h) for h in Ln_histories])
    if H is None:
        raise ValueError("Ln_histories too short or invalid.")

    C, T = H.shape
    L = H.copy()
    llc = n * float(beta) * (L - L0)

    # Optional theta (C,T,d')
    theta = None
    theta_dims = None
    if theta_thin is not None:
        if isinstance(theta_thin, np.ndarray):
            # Accept (C,T,D) or (T,D)
            S = np.asarray(theta_thin)
            if S.ndim == 2 and S.shape[0] >= 2 and S.shape[1] >= 1:
                S = S[None, ...]
            if S.ndim == 3 and S.shape[0] >= 1 and S.shape[1] >= 2 and S.shape[2] >= 1:
                # Truncate draws to T and dims to max_theta_dims
                d = min(S.shape[2], max_theta_dims)
                t = min(S.shape[1], T)
                theta = S[:, :t, :d]
                C = min(C, theta.shape[0])
                T = min(T, theta.shape[1])
                theta = theta[:C, :T, :d]
                theta_dims = np.arange(d, dtype=int)
        else:
            S = stack_ragged_2d(list(theta_thin))  # (C,T,D)
            if S is not None:
                d = min(S.shape[2], max_theta_dims)
                t = min(S.shape[1], T)
                theta = S[:, :t, :d]
                C = min(C, theta.shape[0])
                T = min(T, theta.shape[1])
                theta = theta[:C, :T, :d]
                theta_dims = np.arange(d, dtype=int)

    # Align L/llc to (C,T) in case theta shortened T
    L = L[:C, :T]
    llc = llc[:C, :T]

    # sample_stats: acceptance & energy (C,T)
    sstats = {}
    if acceptance:
        A = stack_ragged_1d(list(acceptance))
        if A is not None:
            t = min(T, A.shape[1])
            A = A[:C, :t]
            L = L[:C, :t]
            llc = llc[:C, :t]
            if theta is not None:
                theta = theta[:C, :t, :]
            sstats["acceptance_rate"] = A
    if energy:
        E = stack_ragged_1d(list(energy))
        if E is not None:
            t = min(T, E.shape[1])
            E = E[:C, :t]
            L = L[:C, :t]
            llc = llc[:C, :t]
            if theta is not None:
                theta = theta[:C, :t, :]
            sstats["energy"] = E

    data = {
        "posterior": {"llc": llc, "L": L},
        "coords": {"chain": np.arange(C), "draw": np.arange(T)},
        "dims": {"llc": ["chain", "draw"], "L": ["chain", "draw"]},
        "sample_stats": sstats if sstats else None,
    }
    if theta is not None:
        data["posterior"]["theta"] = theta
        data["coords"]["theta_dim"] = theta_dims
        data["dims"]["theta"] = ["chain", "draw", "theta_dim"]

    return az.from_dict(**data)
