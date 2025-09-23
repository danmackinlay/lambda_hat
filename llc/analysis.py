# llc/analysis.py
"""
Central analysis functions for batched LLC experiments. Converts runner outputs
to ArviZ InferenceData and provides centralized LLC, ESS, and WNV metrics
without per-step Python hooks.
"""

from __future__ import annotations
from typing import Dict, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import arviz as az
import logging

# Context manager to quiet matplotlib during plot generation
_mpl_logger = logging.getLogger("matplotlib")

class _QuietMatplotlib:
    def __enter__(self):
        self._old = _mpl_logger.level
        # Clamp to WARNING during plotting to avoid transient DEBUG from extensions/plugins
        _mpl_logger.setLevel(max(_mpl_logger.level, logging.WARNING))
        return self
    def __exit__(self, exc_type, exc, tb):
        _mpl_logger.setLevel(self._old)


# ---------- Data Conversion ----------
def idata_from_histories(
    *, Ln_histories, theta_thin, acceptance, energy, n, beta, L0, max_theta_dims=8
):
    """Convert raw sampling histories to ArviZ InferenceData with metadata."""
    from llc.convert import to_idata

    idata = to_idata(
        Ln_histories=Ln_histories,
        theta_thin=theta_thin,
        acceptance=acceptance,
        energy=energy,
        n=n,
        beta=beta,
        L0=L0,
        max_theta_dims=max_theta_dims,
    )
    # Mark attrs so figure fns don't need n,beta,L0 again
    idata.attrs.update({"n_data": int(n), "beta": float(beta), "L0": float(L0)})
    return idata


# ---------- Metrics ----------
def llc_point_se(idata) -> Dict[str, float]:
    """Return mean, SE, ESS_bulk/tail, R-hat for posterior['llc']."""
    mu = float(idata.posterior["llc"].values.mean())
    summ = az.summary(idata, var_names=["llc"])
    out = {"llc_mean": mu}
    if not summ.empty:
        ess_bulk = float(summ.get("ess_bulk", np.nan).iloc[0])
        ess_tail = (
            float(summ.get("ess_tail", np.nan).iloc[0])
            if "ess_tail" in summ.columns
            else np.nan
        )
        rhat = (
            float(summ.get("r_hat", np.nan).iloc[0])
            if "r_hat" in summ.columns
            else np.nan
        )
        sd = float(summ.get("sd", np.nan).iloc[0]) if "sd" in summ.columns else np.nan
        se = (
            float(sd / np.sqrt(ess_bulk))
            if np.isfinite(sd) and ess_bulk > 0
            else np.nan
        )
        out.update(
            {"llc_se": se, "ess_bulk": ess_bulk, "ess_tail": ess_tail, "rhat": rhat}
        )
    return out


def llc_point_se_from_histories(Ln_histories: List, n: int, beta: float, L0: float):
    """Convert histories to idata and compute LLC point estimate + SE (replaces diagnostics version)."""
    from llc.convert import to_idata

    idata = to_idata(
        Ln_histories=Ln_histories,
        theta_thin=None,
        acceptance=None,
        energy=None,
        n=n,
        beta=beta,
        L0=L0,
    )
    metrics = llc_point_se(idata)
    # Return in the format expected by existing callers: (mean, se, ess)
    return (
        metrics.get("llc_mean", float("nan")),
        metrics.get("llc_se", float("nan")),
        int(metrics.get("ess_bulk", 0)),
    )


def efficiency_metrics(
    *, idata, timings: Dict, work: Dict, n_data: int, sgld_batch: Optional[int]
) -> Dict[str, float]:
    """Compute ESS/sec, ESS/FDE, WNV_time, WNV_FDE (FDE = full-data-equivalent grads)."""
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
    se = sd / np.sqrt(max(1.0, ess))
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


# ---------- Figures (return plt.Figure; caller saves) ----------
def fig_running_llc(idata, n: int, beta: float, L0: float, title: str) -> plt.Figure:
    """
    Running LLC per chain + pooled.
    Pooled curve is computed as the cumulative mean of the chain-averaged L_n series:
        pooled[t] = mean_c(L[c,t])_cummean_up_to_t
    so it is directly comparable to each chain's running mean.
    """
    L = idata.posterior.get("L")
    if L is None:
        raise ValueError("posterior['L'] missing; cannot draw running LLC.")
    L = L.values  # (C, T)
    C, T = L.shape

    # Per-chain running means and LLC
    cmean = np.cumsum(L, axis=1) / np.arange(1, T + 1)[None, :]
    lam = n * float(beta) * (cmean - L0)

    # ✅ Correct pooled running mean across chains (fixes the old "double / t" bug)
    # Old (buggy) code divided by t twice: np.mean(L, 0) / np.arange(1, T+1)
    # New: average across chains per step, then cumulative mean over time.
    mean_over_chains = np.mean(L, axis=0)  # (T,)
    cumsum_pooled = np.cumsum(mean_over_chains)  # (T,)
    pooled = cumsum_pooled / np.arange(1, T + 1)  # (T,)
    lam_pooled = n * float(beta) * (pooled - L0)  # (T,)

    # mean ± 2·SE band from ESS
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


def fig_rank_llc(idata) -> plt.Figure:
    axes = az.plot_rank(idata, var_names=["llc"])
    if isinstance(axes, (list, tuple, np.ndarray)):
        return axes.flat[0].figure
    return axes.figure


def fig_autocorr_llc(idata) -> plt.Figure:
    axes = az.plot_autocorr(idata, var_names=["llc"])
    return (
        axes[0].figure if isinstance(axes, (list, tuple, np.ndarray)) else axes.figure
    )


def fig_ess_evolution(idata) -> plt.Figure:
    axes = az.plot_ess(idata, var_names=["llc"], kind="evolution")
    return axes.figure


def fig_ess_quantile(idata) -> plt.Figure:
    axes = az.plot_ess(idata, var_names=["llc"], kind="quantile")
    return axes.figure


def fig_energy(idata) -> plt.Figure:
    ax = az.plot_energy(idata)
    return ax.figure


def fig_theta_trace(idata, dims: int = 4) -> plt.Figure:
    # Check if we have scalar theta variables (new approach)
    theta_vars = [v for v in idata.posterior.data_vars if v.startswith("theta_")]

    if theta_vars:
        # Use scalar theta variables - ArviZ will naturally create one row per variable
        # Sort numerically to ensure proper ordering (theta_0, theta_1, theta_2, ...)
        theta_vars_sorted = sorted(theta_vars, key=lambda x: int(x.split("_")[1]))
        n_vars = min(dims, len(theta_vars_sorted))
        var_names = theta_vars_sorted[:n_vars]
        axes = az.plot_trace(
            idata, var_names=var_names, backend_kwargs={"constrained_layout": True}
        )
        # Return figure from axes grid
        if isinstance(axes, (list, tuple, np.ndarray)):
            return axes.flat[0].figure
        return axes.figure
    else:
        # Fallback to old approach with multi-dimensional theta
        if "theta" not in idata.posterior:
            raise ValueError("Neither scalar theta_* nor multi-dim theta found.")
        nd = idata.posterior["theta"].shape[-1]
        n_vars = min(dims, nd)
        sel = {"theta_dim": list(range(n_vars))}
        fig, axes = plt.subplots(n_vars, 2, figsize=(12, 3 * n_vars), squeeze=False)
        az.plot_trace(
            idata,
            var_names=["theta"],
            coords=sel,
            axes=axes,
            backend_kwargs={"constrained_layout": True},
        )
        return fig


# ---------- Diagnostic Generation ----------
def generate_diagnostics(
    idata, sampler_name: str, out_dir: str, overwrite=False, max_theta_dims=8
):
    """Generate all diagnostic plots for a sampler using ArviZ InferenceData."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Extract metadata from idata attrs
    n = int(idata.attrs.get("n_data", 0))
    beta = float(idata.attrs.get("beta", 1.0))
    L0 = float(idata.attrs.get("L0", 0.0))

    # Generate figures with error handling (quiet matplotlib during plotting)
    with _QuietMatplotlib():
        figs = []
        try:
            figs.append(
            (
                f"{sampler_name}_running_llc.png",
                fig_running_llc(idata, n, beta, L0, f"{sampler_name} Running LLC"),
            )
        )
        except Exception:
            pass
        try:
            figs.append((f"{sampler_name}_llc_rank.png", fig_rank_llc(idata)))
        except Exception:
            pass
        try:
            figs.append((f"{sampler_name}_llc_ess_evolution.png", fig_ess_evolution(idata)))
        except Exception:
            pass
        try:
            figs.append((f"{sampler_name}_llc_ess_quantile.png", fig_ess_quantile(idata)))
        except Exception:
            pass
        try:
            figs.append((f"{sampler_name}_llc_autocorr.png", fig_autocorr_llc(idata)))
        except Exception:
            pass
        try:
            figs.append((f"{sampler_name}_energy.png", fig_energy(idata)))
        except Exception:
            pass
        try:
            # Check for scalar theta variables first
            theta_scalar = [v for v in idata.posterior.data_vars if v.startswith("theta_")]
            if theta_scalar:
                theta_dims = min(max_theta_dims, len(theta_scalar))
            elif "theta" in idata.posterior:
                theta_dims = min(
                    max_theta_dims, int(idata.posterior["theta"].sizes.get("theta_dim", 0))
                )
            else:
                theta_dims = 0

            if theta_dims > 0:
                figs.append(
                    (
                        f"{sampler_name}_theta_trace.png",
                        fig_theta_trace(idata, dims=theta_dims),
                    )
                )
        except Exception:
            pass

        # Save figures
        for name, fig in figs:
            p = out / name
            if (not p.exists()) or overwrite:
                fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
