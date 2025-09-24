import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

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

def to_idata(*, Ln_histories, theta_thin, acceptance, energy, n, beta, L0, max_theta_dims=8):
    """Ragged → (C,T) stack + build InferenceData with posterior['llc','L'] and sample_stats."""

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
    if theta_thin is not None:
        if isinstance(theta_thin, np.ndarray):
            # Accept (C,T,D)
            S = np.asarray(theta_thin)
            if S.ndim == 3 and S.shape[0] >= 1 and S.shape[1] >= 2 and S.shape[2] >= 1:
                d = min(S.shape[2], max_theta_dims)
                t = min(S.shape[1], T)
                theta = S[:, :t, :d]
                C = min(C, theta.shape[0])
                T = min(T, theta.shape[1])
                theta = theta[:C, :T, :d]
        else:
            S = stack_ragged_2d(list(theta_thin))  # (C,T,D)
            if S is not None:
                d = min(S.shape[2], max_theta_dims)
                t = min(S.shape[1], T)
                theta = S[:, :t, :d]
                C = min(C, theta.shape[0])
                T = min(T, theta.shape[1])
                theta = theta[:C, :T, :d]

    # Align L/llc to (C,T)
    L = L[:C, :T]
    llc = llc[:C, :T]

    # Sample stats: acceptance & energy (C,T)
    sstats = {}
    if acceptance is not None:
        A = stack_ragged_1d(list(acceptance))
        if A is not None:
            t_new = min(T, A.shape[1])
            if t_new >= 4:
                A = A[:C, :t_new]
                L = L[:C, :t_new]
                llc = llc[:C, :t_new]
                if theta is not None:
                    theta = theta[:C, :t_new, :]
                T = t_new
                sstats["acceptance_rate"] = A

    if energy is not None:
        E = stack_ragged_1d(list(energy))
        if E is not None:
            t_new = min(T, E.shape[1])
            if t_new >= 4:
                E = E[:C, :t_new]
                L = L[:C, :t_new]
                llc = llc[:C, :t_new]
                if theta is not None:
                    theta = theta[:C, :t_new, :]
                T = t_new
                sstats["energy"] = E

    data = {
        "posterior": {"llc": llc, "L": L},
        "coords": {"chain": np.arange(C), "draw": np.arange(T)},
        "dims": {"llc": ["chain", "draw"], "L": ["chain", "draw"]},
        "sample_stats": sstats if sstats else None,
    }

    if theta is not None:
        # Split theta into separate scalar variables for better ArviZ plots
        d = theta.shape[2]
        for i in range(d):
            data["posterior"][f"theta_{i}"] = theta[:, :, i]
            data["dims"][f"theta_{i}"] = ["chain", "draw"]

    return az.from_dict(**data)

def llc_point_se(idata):
    mu = float(idata.posterior["llc"].values.mean())
    summ = az.summary(idata, var_names=["llc"])
    out = {"llc_mean": mu}
    if not summ.empty:
        ess = float(summ.get("ess_bulk", np.nan).iloc[0])
        sd = float(summ.get("sd", np.nan).iloc[0]) if "sd" in summ.columns else np.nan
        rhat = (
            float(summ.get("r_hat", np.nan).iloc[0])
            if "r_hat" in summ.columns
            else np.nan
        )
        se = sd / np.sqrt(max(1.0, ess)) if np.isfinite(sd) and ess > 0 else np.nan
        out.update({"llc_se": se, "ess_bulk": ess, "rhat": rhat})
    return out

def efficiency_metrics(*, idata, timings, work, n_data, sgld_batch=None):
    """Compute ESS/sec, ESS/FDE, WNV (time/FDE)."""
    summ = az.summary(idata, var_names=["llc"])
    if summ.empty:
        return {
            "ess": np.nan,
            "ess_per_sec": np.nan,
            "ess_per_fde": np.nan,
            "wnv_time": np.nan,
            "wnv_fde": np.nan,
        }

    ess = float(summ["ess_bulk"].iloc[0])
    sd = float(summ["sd"].iloc[0])
    t_sampling = float(timings.get("sampling", np.nan))

    # FDE accounting
    full_loss = float(work.get("n_full_loss", 0.0))
    mb_grads = float(work.get("n_minibatch_grads", 0.0))
    b = float(sgld_batch or 0.0)
    fde = full_loss + (mb_grads * (b / float(n_data))) if n_data > 0 else np.nan

    ess_sec = ess / t_sampling if np.isfinite(t_sampling) and t_sampling > 0 else np.nan
    ess_fde = ess / fde if np.isfinite(fde) and fde > 0 else np.nan
    wnv_time = (
        (sd * sd / ess) * t_sampling
        if ess > 0 and np.isfinite(sd) and np.isfinite(t_sampling)
        else np.nan
    )
    wnv_fde = (
        (sd * sd / ess) * fde
        if ess > 0 and np.isfinite(sd) and np.isfinite(fde)
        else np.nan
    )

    return {
        "ess": ess,
        "ess_per_sec": ess_sec,
        "ess_per_fde": ess_fde,
        "wnv_time": wnv_time,
        "wnv_fde": wnv_fde,
    }

def fig_running_llc(idata, n, beta, L0, title):
    """Pooled running-mean fix → stable running LLC figure."""
    L = idata.posterior.get("L")
    if L is None:
        raise ValueError("posterior['L'] missing; cannot draw running LLC.")
    L = L.values  # (C, T)
    C, T = L.shape

    # Per-chain running means and LLC
    cmean = np.cumsum(L, axis=1) / np.arange(1, T + 1)[None, :]
    lam = n * float(beta) * (cmean - L0)

    # Fixed pooled running mean across chains
    mean_over_chains = np.mean(L, axis=0)  # (T,)
    cumsum_pooled = np.cumsum(mean_over_chains)  # (T,)
    pooled = cumsum_pooled / np.arange(1, T + 1)  # (T,)
    lam_pooled = n * float(beta) * (pooled - L0)  # (T,)

    # mean ± 2·SE band
    summ = az.summary(idata, var_names=["llc"])
    mu, se = float(idata.posterior["llc"].values.mean()), np.nan
    if not summ.empty:
        ess = float(summ["ess_bulk"].iloc[0])
        sd = float(summ["sd"].iloc[0])
        se = sd / np.sqrt(max(1.0, ess))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i in range(C):
        ax.plot(lam[i], alpha=0.7, label=f"Chain {i}")
    ax.plot(lam_pooled, "k-", lw=2, label="Pooled")
    if np.isfinite(se):
        ax.axhline(mu, ls="--", lw=1)
        ax.fill_between(np.arange(T), mu - 2 * se, mu + 2 * se, alpha=0.1)

    ax.set_xlabel("Evaluation")
    ax.set_ylabel("LLC = n·β·(E[Lₙ] − L₀)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig