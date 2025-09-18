# llc/diagnostics.py
"""Diagnostic and plotting utilities for LLC analysis (lean + robust).

Note on ESS (Effective Sample Size):
For full-data MCMC samplers like HMC, ESS provides a relatively rigorous estimate of statistical efficiency.
For stochastic gradient samples SGLD and MCLMC, ESS is even more heuristic.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Any, Sequence
import numpy as np
import matplotlib.pyplot as plt
# Do not import heavy optional deps at module import time (keeps Modal happy)
# pandas and scipy are imported lazily in the functions that need them.

ESS_METHOD = "bulk"  # ArviZ bulk ESS (rank-normalized)

__all__ = [
    "llc_mean_and_se_from_histories",
    "llc_ci_from_histories",
    "plot_diagnostics",
    "create_summary_dataframe",
]


def _az():
    # Lazy import to avoid pulling heavy deps when not plotting/estimating
    import arviz as az

    return az


def llc_mean_and_se_from_histories(
    Ln_histories: List[np.ndarray], n: int, beta: float, L0: float
) -> Tuple[float, float, int]:
    """
    Compute LLC mean and standard error from L_n evaluation histories using ESS.

    This is the primary method for LLC estimation, using ArviZ bulk ESS for
    uncertainty quantification that attempts to account for autocorrelation.

    Args:
        Ln_histories: Per-chain histories of L_n evaluations
        n: Number of data points
        beta: Inverse temperature
        L0: Loss at empirical minimizer

    Returns:
        Tuple of (llc_mean, standard_error, effective_sample_size)
    """
    H = _stack_histories(Ln_histories)
    if H is None:
        return 0.0, np.nan, 0

    # Transform to LLC
    llc_vals = n * beta * (H - L0)  # (chains, evals)

    # Create ArviZ InferenceData
    az = _az()
    idata = az.from_dict(
        posterior={"llc": llc_vals},
        coords={
            "chain": np.arange(llc_vals.shape[0]),
            "draw": np.arange(llc_vals.shape[1]),
        },
        dims={"llc": ["chain", "draw"]},
    )

    # Compute ESS and statistics
    ess = az.ess(idata, method=ESS_METHOD)["llc"].values
    eff_sample_size = float(np.mean(ess)) if not np.isnan(ess).all() else 1.0

    # Pool across chains for final estimate
    pooled = llc_vals.flatten()
    mean_llc = float(np.mean(pooled))

    # Standard error accounting for ESS
    if len(pooled) > 1 and eff_sample_size > 1:
        se_llc = float(np.std(pooled, ddof=1) / np.sqrt(eff_sample_size))
    else:
        se_llc = np.nan

    return mean_llc, se_llc, int(eff_sample_size)


def llc_ci_from_histories(
    Ln_histories: List[np.ndarray], n: int, beta: float, L0: float, alpha: float = 0.05
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute LLC with confidence interval using ESS-based standard error.

    Uses the same ESS methodology as llc_mean_and_se_from_histories for consistency.

    Args:
        Ln_histories: Per-chain histories of L_n evaluations
        n: Number of data points
        beta: Inverse temperature
        L0: Loss at empirical minimizer
        alpha: Significance level for CI (default 0.05 for 95% CI)

    Returns:
        Tuple of (llc_mean, (ci_lower, ci_upper))
    """
    # Pack ragged histories to a rectangular array by truncating to min length
    m = min(len(h) for h in Ln_histories if len(h) > 0)
    if m == 0:
        return 0.0, (0.0, 0.0)
    H = np.stack([np.asarray(h[:m]) for h in Ln_histories], axis=0)  # (chains, m)

    lam = n * beta * (H - L0)
    az = _az()
    from scipy.stats import norm  # type: ignore

    idata = az.from_dict(
        posterior={"llc": lam},
        coords={"chain": np.arange(lam.shape[0]), "draw": np.arange(lam.shape[1])},
        dims={"llc": ["chain", "draw"]},
    )
    ess_llc = az.ess(idata, method=ESS_METHOD)["llc"].values
    eff_sample_size = float(np.mean(ess_llc)) if not np.isnan(ess_llc).all() else 1.0
    pooled = lam.flatten()
    mean_val = float(np.mean(pooled))

    if len(pooled) > 1 and eff_sample_size > 1:
        se_val = float(np.std(pooled, ddof=1) / np.sqrt(eff_sample_size))
        z = float(norm.ppf(1 - alpha / 2))
        ci = (mean_val - z * se_val, mean_val + z * se_val)
    else:
        ci = (mean_val, mean_val)

    return mean_val, ci


def _stack_histories(Ln_histories: List[np.ndarray]) -> Optional[np.ndarray]:
    """Stack ragged L_n histories into rectangular array"""
    if not Ln_histories or all(len(h) == 0 for h in Ln_histories):
        return None

    # Truncate to minimum length to avoid NaN padding
    min_len = min(len(h) for h in Ln_histories if len(h) > 0)
    if min_len == 0:
        return None

    return np.stack([h[:min_len] for h in Ln_histories], axis=0)


def _idata_from_L(Ln_histories: List[np.ndarray]) -> Tuple[Optional[Any], int]:
    """ArviZ InferenceData for L_n traces (posterior group; dims chain,draw)."""
    H = _stack_histories(Ln_histories)
    if H is None:
        return None, 0
    az = _az()
    idata = az.from_dict(
        posterior={"L": H},
        coords={"chain": np.arange(H.shape[0]), "draw": np.arange(H.shape[1])},
        dims={"L": ["chain", "draw"]},
    )
    return idata, H.shape[1]


def _idata_from_llc(
    Ln_histories: List[np.ndarray],
    n: int,
    beta: float,
    L0: float,
    acceptance_rates: Optional[List[np.ndarray]] = None,
    energies: Optional[List[np.ndarray]] = None,
) -> Optional[Any]:
    """InferenceData with `llc` + optional sample_stats (acceptance, energy)."""
    H = _stack_histories(Ln_histories)
    if H is None:
        return None
    llc = n * float(beta) * (H - L0)  # (chain, draw)

    # Align ragged stats to draws via truncation to min length
    sample_stats = {}

    # Find the minimum length across all arrays that need to be aligned
    min_len = llc.shape[1]
    if acceptance_rates is not None and any(len(a) > 0 for a in acceptance_rates):
        min_len = min(min_len, min(len(a) for a in acceptance_rates if len(a) > 0))
    if energies is not None and any(len(e) > 0 for e in energies):
        min_len = min(min_len, min(len(e) for e in energies if len(e) > 0))

    # Truncate all arrays to the same length
    if min_len > 0:
        llc = llc[:, :min_len]

        if acceptance_rates is not None and any(len(a) > 0 for a in acceptance_rates):
            acc = np.stack([np.asarray(a[:min_len]) for a in acceptance_rates], axis=0)
            sample_stats["acceptance_rate"] = acc

        if energies is not None and any(len(e) > 0 for e in energies):
            en = np.stack([np.asarray(e[:min_len]) for e in energies], axis=0)
            sample_stats["energy"] = en

    az = _az()
    idata = az.from_dict(
        posterior={"llc": llc},
        sample_stats=sample_stats if sample_stats else None,
        coords={"chain": np.arange(llc.shape[0]), "draw": np.arange(llc.shape[1])},
        dims={
            "llc": ["chain", "draw"],
            **({k: ["chain", "draw"] for k in sample_stats} if sample_stats else {}),
        },
    )
    return idata


def _idata_from_theta(
    samples_thin: Sequence[np.ndarray] | np.ndarray, max_dims: int = 8
) -> Tuple[Optional[Any], List[int]]:
    """
    Build ArviZ InferenceData for theta from either:
      • list/tuple of (draws, dims) arrays (ragged allowed), or
      • ndarray shaped (chains, draws, dims) or (draws, dims).
    Robust to ragged chains: truncates to the shortest non-empty chain.
    """
    # Normalise inputs
    chains: list[np.ndarray] = []
    if isinstance(samples_thin, np.ndarray):
        S = samples_thin
        if S.size == 0:
            return None, []
        if S.ndim == 2:  # (draws, dims) -> add singleton chain
            if S.shape[0] < 2 or S.shape[1] < 1:
                return None, []
            chains = [S]
        elif S.ndim == 3:  # (chains, draws, dims)
            for c in range(S.shape[0]):
                Si = S[c]
                if Si.ndim == 2 and Si.shape[0] >= 2 and Si.shape[1] >= 1:
                    chains.append(Si)
        else:
            return None, []
    else:
        # Sequence of arrays
        for arr in samples_thin:
            if arr is None:
                continue
            A = np.asarray(arr)
            if A.ndim == 2 and A.shape[0] >= 2 and A.shape[1] >= 1:
                chains.append(A)

    # Guard: need at least one non-empty chain with ≥2 draws
    if not chains:
        return None, []

    # Truncate to common length (min draws) to get rectangular (C, T, D)
    min_len = min(c.shape[0] for c in chains)
    if min_len < 2:
        return None, []
    # Use the same dims across chains (assume consistent model)
    D = chains[0].shape[1]
    if D < 1:
        return None, []

    # Downselect theta dims BEFORE stacking to keep memory bounded
    idx = list(range(min(D, max_dims)))
    stacked = np.stack([c[:min_len, :][:, idx] for c in chains], axis=0)  # (C, T, d')

    az = _az()
    idata = az.from_dict(
        posterior={"theta": stacked},
        coords={
            "chain": np.arange(stacked.shape[0]),
            "draw": np.arange(stacked.shape[1]),
            "theta_dim": np.array(idx, dtype=int),
        },
        dims={"theta": ["chain", "draw", "theta_dim"]},
    )
    return idata, idx


def _running_llc(
    Ln_histories: List[np.ndarray], n: int, beta: float, L0: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute running LLC estimates"""
    H = _stack_histories(Ln_histories)
    if H is None:
        return None, None
    cmean = np.cumsum(H, 1) / np.arange(1, H.shape[1] + 1)[None, :]
    lam = n * float(beta) * (cmean - L0)
    pooled = (
        np.sum(H, 0) / H.shape[0] / np.arange(1, H.shape[1] + 1)
    )  # pooled running mean
    lam_pooled = n * float(beta) * (pooled - L0)
    return lam, lam_pooled


def plot_diagnostics(
    run_dir: str,
    sampler_name: str,
    Ln_histories: List[np.ndarray],
    samples_thin: np.ndarray,
    acceptance_rates: Optional[List[np.ndarray]] = None,
    energy_deltas: Optional[List[np.ndarray]] = None,
    energies: Optional[List[np.ndarray]] = None,
    n: int = 1000,
    beta: float = 1.0,
    L0: float = 0.0,
    save_plots: bool = True,
) -> None:
    """Generate diagnostics for a sampler, aligned with ArviZ best practice."""
    # Build idata objects
    idata_llc = _idata_from_llc(
        Ln_histories, n, beta, L0, acceptance_rates=acceptance_rates, energies=energies
    )

    # 1) L_n trace plots (raw and centered)
    if Ln_histories and any(len(h) > 0 for h in Ln_histories):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for i, hist in enumerate(Ln_histories):
            if len(hist) > 0:
                ax.plot(hist, alpha=0.7, label=f"Chain {i}")
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("L_n")
        ax.set_title(f"{sampler_name} L_n Traces")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save_plots:
            _finalize_figure(fig, f"{run_dir}/{sampler_name}_Ln_trace.png")
        plt.close(fig)

        # Centered L_n - L0
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for i, hist in enumerate(Ln_histories):
            if len(hist) > 0:
                ax.plot(np.asarray(hist) - L0, alpha=0.7, label=f"Chain {i}")
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("L_n - L_0")
        ax.set_title(f"{sampler_name} Centered L_n")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save_plots:
            _finalize_figure(fig, f"{run_dir}/{sampler_name}_Ln_centered.png")
        plt.close(fig)

    # 2) Running LLC plot (per-chain + pooled, with ±2·SE band)
    lam, lam_pooled = _running_llc(Ln_histories, n, beta, L0)
    if lam is not None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for i in range(lam.shape[0]):
            ax.plot(lam[i], alpha=0.7, label=f"Chain {i}")
        if lam_pooled is not None:
            ax.plot(lam_pooled, "k-", linewidth=2, label="Pooled")
        # final mean±2SE band using ESS-based estimator
        mean_llc, se_llc, _ess = llc_mean_and_se_from_histories(
            Ln_histories, n, beta, L0
        )
        if np.isfinite(se_llc):
            ax.axhline(mean_llc, linestyle="--", linewidth=1)
            ax.fill_between(
                np.arange(len(lam_pooled)),
                mean_llc - 2 * se_llc,
                mean_llc + 2 * se_llc,
                alpha=0.1,
            )
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("LLC = n·β·(E[Lₙ] − L₀)")
        ax.set_title(f"{sampler_name} Running LLC")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save_plots:
            _finalize_figure(fig, f"{run_dir}/{sampler_name}_running_llc.png")
        plt.close(fig)

    # Theta trace plots (if available)
    idata_theta, theta_idx = _idata_from_theta(samples_thin)
    if idata_theta is not None and theta_idx:
        n_dims_to_plot = min(4, len(theta_idx))
        sel = {"theta_dim": theta_idx[:n_dims_to_plot]}
        fig, axes = plt.subplots(2, n_dims_to_plot, figsize=(12, 6), squeeze=False)
        # One call: ArviZ fills the 2×N grid itself
        _az().plot_trace(
            idata_theta,
            var_names=["theta"],
            coords=sel,
            axes=axes,
            backend_kwargs={"constrained_layout": True},  # nicer layout
        )
        plt.suptitle(f"{sampler_name} Parameter Traces")
        plt.tight_layout()
        if save_plots:
            _finalize_figure(fig, f"{run_dir}/{sampler_name}_theta_trace.png")
        plt.close(fig)

    # Acceptance rate plot (HMC only)
    if acceptance_rates is not None and any(len(acc) > 0 for acc in acceptance_rates):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for i, acc in enumerate(acceptance_rates):
            if len(acc) > 0:
                ax.plot(acc, alpha=0.7, label=f"Chain {i}")
        ax.set_xlabel("Draw")
        ax.set_ylabel("Acceptance Rate")
        ax.set_title(f"{sampler_name} Acceptance Rate")
        # reference targets from Stan (~0.8) and ChEES (~0.651)
        ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="Target 0.8")
        ax.axhline(0.651, color="gray", linestyle="--", alpha=0.5, label="Target 0.651")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save_plots:
            _finalize_figure(fig, f"{run_dir}/{sampler_name}_acceptance.png")
        plt.close(fig)

    # Energy delta histogram (MCLMC only)
    if energy_deltas is not None and any(len(e) > 0 for e in energy_deltas):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        all_deltas = np.concatenate([e for e in energy_deltas if len(e) > 0])
        ax.hist(all_deltas, bins=50, alpha=0.7, density=True)
        ax.set_xlabel("Energy Change")
        ax.set_ylabel("Density")
        ax.set_title(f"{sampler_name} Energy Changes")
        ax.grid(True, alpha=0.3)
        if save_plots:
            _finalize_figure(fig, f"{run_dir}/{sampler_name}_energy_hist.png")
        plt.close(fig)

    # 6) ArviZ-first plots for llc: rank, autocorr, ESS, and summary text
    if idata_llc is not None:
        try:
            # rank plot
            axes = _az().plot_rank(idata_llc, var_names=["llc"])
            if save_plots:
                if hasattr(axes, "figure"):
                    _finalize_figure(
                        axes.figure, f"{run_dir}/{sampler_name}_llc_rank.png"
                    )
                    plt.close(axes.figure)
                elif isinstance(axes, np.ndarray) and hasattr(axes.flat[0], "figure"):
                    _finalize_figure(
                        axes.flat[0].figure, f"{run_dir}/{sampler_name}_llc_rank.png"
                    )
                    plt.close(axes.flat[0].figure)

            # autocorr
            axes = _az().plot_autocorr(idata_llc, var_names=["llc"])
            if save_plots and axes is not None:
                if isinstance(axes, np.ndarray) and len(axes) > 0:
                    _finalize_figure(
                        axes[0].figure, f"{run_dir}/{sampler_name}_llc_autocorr.png"
                    )
                    plt.close(axes[0].figure)

            # ESS evolution
            axes = _az().plot_ess(idata_llc, var_names=["llc"], kind="evolution")
            if save_plots and hasattr(axes, "figure"):
                _finalize_figure(
                    axes.figure, f"{run_dir}/{sampler_name}_llc_ess_evolution.png"
                )
                plt.close(axes.figure)

            # Efficiency panel: SE vs draws (prefix-wise ESS)
            try:
                # Recompute s_hat and SE curve from the same data
                H = _stack_histories(Ln_histories)
                if H is not None and H.shape[1] > 5:
                    lam = n * float(beta) * (H - L0)  # (chains, draws)
                    pooled = lam.reshape(-1)
                    s_hat = float(np.std(pooled, ddof=1))
                    az = _az()
                    T = lam.shape[1]
                    # 50 checkpoints from 20 draws up to T
                    checkpoints = np.unique(
                        np.linspace(20, T, num=min(50, T), dtype=int)
                    )
                    ess_evo = []
                    for t in checkpoints:
                        id_pref = az.from_dict(
                            posterior={"llc": lam[:, :t]},
                            coords={
                                "chain": np.arange(lam.shape[0]),
                                "draw": np.arange(t),
                            },
                            dims={"llc": ["chain", "draw"]},
                        )
                        ess_t = az.ess(id_pref, method="bulk")["llc"].values
                        ess_evo.append(float(np.nanmean(ess_t)))
                    ess_evo = np.asarray(ess_evo)
                    xs = checkpoints.astype(float)
                    se_curve = s_hat / np.sqrt(np.maximum(1.0, ess_evo))
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                    ax.plot(xs, se_curve)
                    ax.set_xlabel("Draws")
                    ax.set_ylabel("SE(LLC)")
                    ax.set_title(f"{sampler_name} SE vs draws")
                    ax.grid(True, alpha=0.3)
                    if save_plots:
                        _finalize_figure(
                            fig, f"{run_dir}/{sampler_name}_llc_se_vs_draws.png"
                        )
                    plt.close(fig)
            except Exception:
                pass

            # ESS quantile (interval reliability)
            axes = _az().plot_ess(idata_llc, var_names=["llc"], kind="quantile")
            if save_plots and hasattr(axes, "figure"):
                _finalize_figure(
                    axes.figure, f"{run_dir}/{sampler_name}_llc_ess_quantile.png"
                )
                plt.close(axes.figure)

            # energy (HMC only; harmless no-op if not present)
            try:
                ax = _az().plot_energy(idata_llc)  # requires sample_stats.energy
                if save_plots:
                    _finalize_figure(ax.figure, f"{run_dir}/{sampler_name}_energy.png")
                    plt.close(ax.figure)
            except Exception:
                pass

            # R-hat/ESS summary table (to console)
            summ = _az().summary(idata_llc, var_names=["llc"])
            if not summ.empty:
                rhat = (
                    float(summ["r_hat"].iloc[0]) if "r_hat" in summ.columns else np.nan
                )
                ess_bulk = (
                    float(summ["ess_bulk"].iloc[0])
                    if "ess_bulk" in summ.columns
                    else np.nan
                )
                ess_tail = (
                    float(summ["ess_tail"].iloc[0])
                    if "ess_tail" in summ.columns
                    else np.nan
                )
                # Check if single chain run
                n_chains = idata_llc.posterior.dims.get("chain", 1)
                if n_chains < 2:
                    print(
                        f"[{sampler_name}] ESS_bulk = {ess_bulk:.1f}, ESS_tail = {ess_tail:.1f} (Note: only one chain — R-hat is not informative)"
                    )
                else:
                    print(
                        f"[{sampler_name}] R-hat = {rhat:.4f}, ESS_bulk = {ess_bulk:.1f}, ESS_tail = {ess_tail:.1f}"
                    )
        except Exception as e:
            # ArviZ plotting can fail with insufficient data
            print(f"[{sampler_name}] Could not generate ArviZ diagnostics: {e}")


def _finalize_figure(
    fig: plt.Figure, path: str, dpi: int = 150, bbox_inches: str = "tight"
) -> None:
    """Save and close a matplotlib figure"""
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor="white")


def create_summary_dataframe(results: dict, samplers: List[str]) -> Any:
    """Create a summary dataframe from sampling results"""
    # Lazy import so diagnostics module can import without pandas installed.
    import pandas as pd  # type: ignore
    summary_data = []

    for sampler in samplers:
        if sampler in results:
            res = results[sampler]
            summary_data.append(
                {
                    "sampler": sampler,
                    "llc_mean": res.get("llc_mean", np.nan),
                    "llc_se": res.get("llc_se", np.nan),
                    "ess": res.get("ess", np.nan),
                    "wnv_time": res.get("wnv_time", np.nan),
                    "wnv_fde": res.get("wnv_fde", np.nan),
                    "acceptance": res.get("acceptance", np.nan)
                    if sampler == "hmc"
                    else np.nan,
                    "time_sampling": res.get("time_sampling", np.nan),
                    "work_grads": res.get("work_grads", np.nan),
                }
            )

    return pd.DataFrame(summary_data)
