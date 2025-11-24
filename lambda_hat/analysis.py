# lambda_hat/analysis.py
"""Analysis functions for LLC experiments with proper visualization"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

# Use non-GUI backend and disable LaTeX BEFORE any other matplotlib imports
# This must happen before importing pyplot or arviz to avoid ParseException in parallel execution
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["mathtext.default"] = "regular"

import arviz as az  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jax import random  # noqa: E402

from .data import add_noise, build_teacher, sample_X  # noqa: E402

log = logging.getLogger(__name__)


def _debug_print_idata(idata, name: str):
    """Debug helper to identify degenerate arrays in InferenceData."""

    def _summ(v):
        arr = np.asarray(v)
        return dict(
            shape=arr.shape,
            finite=np.isfinite(arr).sum().item(),
            nans=np.isnan(arr).sum().item(),
            min=(np.nanmin(arr) if np.isfinite(arr).any() else np.nan),
            max=(np.nanmax(arr) if np.isfinite(arr).any() else np.nan),
            unique=(len(np.unique(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0),
        )

    log.debug("=== DEBUG %s ===", name)
    if hasattr(idata, "posterior") and "llc" in idata.posterior:
        log.debug("llc: %s", _summ(idata.posterior["llc"].values))
    if hasattr(idata, "posterior") and "L" in idata.posterior:
        log.debug("L: %s", _summ(idata.posterior["L"].values))
    if hasattr(idata, "sample_stats"):
        for key in [
            "energy",
            "acceptance_rate",
            "cumulative_fge",
            "cumulative_time",
            "diverging",
            "elbo",
            "elbo_like",
            "logq",
            "resp_entropy",
            "pi_entropy",
            "grad_norm",
        ]:
            if key in idata.sample_stats:
                log.debug("%s: %s", key, _summ(idata.sample_stats[key].values))


def analyze_traces(
    traces: Dict[str, np.ndarray],
    manifest: Dict,
    mode: str = "light",
    outdir: Optional[Path] = None,
) -> Tuple[az.InferenceData, Dict[str, float]]:
    """Analyze sampling traces, compute LLC metrics, and create InferenceData.

    This is the single golden path for converting raw traces to diagnostics.
    Workers write traces_raw.npz + manifest.json; controller calls this function.

    Args:
        traces: Dictionary of trace arrays (from traces_raw.npz). Must contain 'llc' key.
        manifest: Run manifest dict (from manifest.json) with metadata needed for analysis.
        mode: Diagnostic depth: "light" (basic plots) or "full" (+ expensive WNV plots).
        outdir: If provided, writes trace.nc, analysis.json, and plots to this directory.

    Returns:
        Tuple of (ArviZ InferenceData, metrics dictionary).

    Raises:
        ValueError: If required keys are missing from traces or manifest.
    """
    # Extract metadata from manifest
    sampler_name = manifest.get("sampler", "unknown")
    warmup = manifest.get("warmup", 0)
    timings = manifest.get("timings")
    work = manifest.get("work")
    sampler_flavour = manifest.get("sampler_flavour")
    # Note: L0, n_data, beta no longer used (kept in manifest for reference only)
    # All samplers must provide pre-computed LLC
    if "llc" not in traces:
        raise ValueError(
            "traces must contain 'llc' key. "
            "All samplers should compute LLC (Local Learning Coefficient) and include it in traces."
        )

    llc_values = traces["llc"]
    if llc_values.ndim == 1:
        llc_values = llc_values[None, :]
    chains, draws = llc_values.shape

    # Also get Ln if available for posterior tracking
    Ln_values = traces.get("Ln")
    if Ln_values is not None:
        if Ln_values.ndim == 1:
            Ln_values = Ln_values[None, :]
    # Validate and apply warmup (burn-in)
    if warmup >= draws:
        warnings.warn(f"Warmup draws ({warmup}) >= total draws ({draws}). Using all samples.")
        warmup = 0

    # Select post-warmup data
    llc_post_warmup = llc_values[:, warmup:]
    draws_post_warmup = draws - warmup
    llc_values_np = np.array(llc_post_warmup)

    # Prepare posterior data
    posterior_data = {"llc": llc_values_np}
    if Ln_values is not None:
        Ln_post_warmup = Ln_values[:, warmup:]
        posterior_data["L"] = np.array(Ln_post_warmup)

    # Handle sample statistics (FGEs, acceptance, time, etc.)
    sample_stats_data = {}

    # Extract FGEs (tracked during sampling)
    if "cumulative_fge" in traces:
        # Ensure FGE data is extracted as float64 numpy array
        fge_post_warmup = traces["cumulative_fge"][:, warmup:]
        sample_stats_data["cumulative_fge"] = np.array(fge_post_warmup, dtype=np.float64)

    # Calculate cumulative time using precise timings
    if timings:
        sampling_time = timings.get("sampling", 0.0)
        adaptation_time = timings.get("adaptation", 0.0)

        if sampling_time > 0:
            # Determine how time relates to the recorded draws
            if adaptation_time == 0.0:
                # Case 1: SGLD/MCLMC (No separate adaptation). Sampling time covers all 'draws'.
                time_per_draw = sampling_time / draws if draws > 0 else 0.0
                cumulative_time_all = np.arange(1, draws + 1) * time_per_draw
                # Select post-warmup time
                cumulative_time_post_warmup = cumulative_time_all[warmup:]
            else:
                # Case 2: HMC (Separate adaptation). Sampling time covers 'draws_post_warmup'.
                # Note: For HMC, warmup should be 0 as traces start after adaptation.
                time_per_draw = sampling_time / draws_post_warmup if draws_post_warmup > 0 else 0.0
                cumulative_time_post_warmup = np.arange(1, draws_post_warmup + 1) * time_per_draw
                # Add adaptation time as offset (time starts after adaptation)
                cumulative_time_post_warmup += adaptation_time

            # Replicate across chains (C, T)
            if cumulative_time_post_warmup.size > 0:
                cumulative_time_data = np.tile(cumulative_time_post_warmup, (chains, 1))
                sample_stats_data["cumulative_time"] = cumulative_time_data

    # Extract other diagnostics
    for key in ["acceptance_rate", "energy", "is_divergent"]:
        if key in traces:
            stat_trace = traces[key]
            # Ensure trace length matches total draws before applying warmup slicing
            if stat_trace.shape[1] == draws:
                # Standardize key for ArviZ if necessary
                output_key = "diverging" if key == "is_divergent" else key
                sample_stats_data[output_key] = np.array(stat_trace[:, warmup:])

    # Extract VI-specific diagnostics (Stage 2)
    vi_keys = [
        "elbo",
        "elbo_like",
        "logq",
        "resp_entropy",
        "pi_min",
        "pi_max",
        "pi_entropy",
        "D_sqrt_min",
        "D_sqrt_max",
        "D_sqrt_med",
        "grad_norm",
        "A_col_norm_max",
    ]
    for key in vi_keys:
        if key in traces:
            stat_trace = traces[key]
            if stat_trace.shape[1] == draws:
                sample_stats_data[key] = np.array(stat_trace[:, warmup:])

    # Create ArviZ InferenceData (contains only post-warmup data)
    data = {
        "posterior": posterior_data,
        "sample_stats": sample_stats_data if sample_stats_data else None,
        "coords": {"chain": np.arange(chains), "draw": np.arange(draws_post_warmup)},
        "dims": {"llc": ["chain", "draw"], "L": ["chain", "draw"]},
    }
    idata = az.from_dict(**data)

    # Default to markov if not specified
    if sampler_flavour is None:
        sampler_flavour = "markov"

    # Compute metrics
    metrics = _compute_metrics_from_idata(idata, llc_values_np, timings, work, sampler_flavour)

    # Write outputs if outdir provided (golden path for diagnostics)
    if outdir is not None:
        import json

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Write trace.nc (ArviZ InferenceData cache)
        idata.to_netcdf(outdir / "trace.nc")
        log.info("[analysis] Wrote trace.nc to %s", outdir)

        # Write analysis.json (metrics cache)
        (outdir / "analysis.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
        log.info("[analysis] Wrote analysis.json to %s", outdir)

        # Generate plots based on mode
        # For plotting functions, wrap idata in dict with sampler name as key
        idata_dict = {sampler_name: idata}

        # Always generate basic plots
        create_arviz_diagnostics(idata_dict, outdir)
        create_combined_convergence_plot(idata_dict, outdir)

        # Generate expensive plots only in "full" mode
        if mode == "full":
            # WNV plot is expensive; only generate if work metrics available
            if work is not None:
                create_work_normalized_variance_plot(idata_dict, outdir)

    return idata, metrics


def _compute_metrics_from_idata(
    idata: az.InferenceData,
    llc_values_np: np.ndarray,
    timings: Optional[Dict[str, float]],
    work: Optional[Dict[str, float]] = None,
    sampler_flavour: Optional[str] = None,
) -> Dict[str, float]:
    """Helper to compute metrics from InferenceData.

    Args:
        idata: ArviZ InferenceData object
        llc_values_np: LLC values as numpy array
        timings: Timing information
        work: Work tracking information
        sampler_flavour: "iid" for independent draws, "markov" for MCMC (default)
    """
    metrics = {
        "llc_mean": float(llc_values_np.mean()),
        "llc_std": float(llc_values_np.std()),  # Standard deviation of the samples
        "llc_min": float(llc_values_np.min()),
        "llc_max": float(llc_values_np.max()),
    }

    # Extract Total Work (FGEs and Time)
    total_fge = 0.0
    if hasattr(idata, "sample_stats") and "cumulative_fge" in idata.sample_stats:
        fges = idata.sample_stats["cumulative_fge"].values
        if fges.size > 0:
            # Use max across chains to represent the total work done
            total_fge = float(np.max(fges))
    metrics["total_fge"] = total_fge

    total_time = timings.get("total", 0.0) if timings else 0.0
    metrics["elapsed_time"] = total_time

    # Sizes for simple guardrails
    C = idata.posterior["llc"].sizes.get("chain", 1)
    T = idata.posterior["llc"].sizes.get("draw", 0)

    # Defaults
    metrics["ess_bulk"] = np.nan
    metrics["ess_tail"] = np.nan
    metrics["r_hat"] = np.nan
    metrics["ess"] = np.nan
    metrics["llc_sem"] = np.nan

    # Sampler-specific metrics based on draw independence
    if sampler_flavour == "iid":
        # IID draws (e.g., VI) - ESS = total draws, R-hat undefined
        metrics["ess"] = float(C * T)
        metrics["ess_bulk"] = float(C * T)
        metrics["ess_tail"] = float(C * T)
        metrics["r_hat"] = float("nan")  # Undefined for IID samples

        # Variance estimates for IID samples
        if metrics["ess"] > 0:
            variance_estimate = metrics["llc_std"] ** 2 / metrics["ess"]
            metrics["llc_sem"] = float(np.sqrt(variance_estimate))
    else:
        # Markov chain path (default) - use ArviZ diagnostics
        # ESS is available even for 1 chain, but can be noisy for tiny T.
        MIN_DRAWS_FOR_ESS = 10
        if T >= MIN_DRAWS_FOR_ESS:
            try:
                bulk = az.ess(idata, var_names=["llc"], method="bulk").to_array().values
                tail = az.ess(idata, var_names=["llc"], method="tail").to_array().values
                metrics["ess_bulk"] = float(bulk.flatten()[0]) if bulk.size else np.nan
                metrics["ess_tail"] = float(tail.flatten()[0]) if tail.size else np.nan
            except Exception:
                pass

        # r-hat is meaningful only with multiple chains and enough draws
        MIN_DRAWS_FOR_RHAT = 20
        if C >= 2 and T >= MIN_DRAWS_FOR_RHAT:
            try:
                rhat = az.rhat(idata, var_names=["llc"]).to_array().values
                metrics["r_hat"] = float(rhat.flatten()[0]) if rhat.size else np.nan
            except Exception:
                pass

        # Consolidated ESS for MCMC
        ess = min(metrics["ess_bulk"], metrics["ess_tail"])
        if not np.isnan(ess) and ess > 0:
            metrics["ess"] = ess
            # Variance of the estimate (SEM^2) = Var(samples) / ESS
            variance_estimate = metrics["llc_std"] ** 2 / ess
            metrics["llc_sem"] = float(np.sqrt(variance_estimate))

    # Work-normalized variance (all samplers)
    if work is not None and not np.isnan(metrics["ess"]) and metrics["ess"] > 0:
        variance_estimate = metrics["llc_std"] ** 2 / metrics["ess"]
        total_work = work.get("n_full_loss", 0.0) + work.get("n_minibatch_grads", 0.0)
        if total_work > 0:
            metrics["wnv"] = float(variance_estimate * total_work)
            metrics["efficiency_work"] = float(metrics["ess"] / total_work)

    return metrics


def create_arviz_diagnostics(inference_data: Dict[str, az.InferenceData], output_dir: Path) -> None:
    """Create ArviZ diagnostic plots (Trace, Rank, Energy)."""
    # Create a subdirectory for detailed plots
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    def _has_var(idata, var):
        """Check if variable has >1 finite value and is not constant."""
        if hasattr(idata, "posterior") and var in idata.posterior:
            vals = np.asarray(idata.posterior[var].values)
            vals = vals[np.isfinite(vals)]
            return vals.size > 1 and (np.nanmax(vals) != np.nanmin(vals))
        return False

    for sampler_name, idata in inference_data.items():
        # 1. Trace Plot (Trace + Posterior Density), only if we have >1 finite value
        try:
            vars_to_plot = [v for v in ["llc", "L"] if _has_var(idata, v)]
            if vars_to_plot:
                az.plot_trace(idata, var_names=vars_to_plot, figsize=(12, 8), compact=False)
                plt.suptitle(f"{sampler_name.upper()} Trace Plot", y=1.02)
                plt.tight_layout()
                plt.savefig(diag_dir / "trace.png", dpi=150, bbox_inches="tight")
                plt.close()
            else:
                warnings.warn(
                    f"{sampler_name}: skipped trace plot (degenerate or no finite values)."
                )
        except Exception:
            warnings.warn(f"Failed to create trace plot for {sampler_name}")

        # 2. Rank Plot (Convergence check)
        try:
            if _has_var(idata, "llc"):
                az.plot_rank(idata, var_names=["llc"], figsize=(12, 5))
                plt.suptitle(f"{sampler_name.upper()} Rank Plot", y=1.02)
                plt.tight_layout()
                plt.savefig(diag_dir / "rank.png", dpi=150, bbox_inches="tight")
                plt.close()
            else:
                warnings.warn(f"{sampler_name}: skipped rank plot (llc degenerate).")
        except Exception:
            warnings.warn(f"Failed to create rank plot for {sampler_name}")

        # 3. Energy Plot (if available, useful for HMC/MCLMC)
        if hasattr(idata, "sample_stats") and "energy" in idata.sample_stats:
            try:
                az.plot_energy(idata, figsize=(12, 6))
                plt.suptitle(f"{sampler_name.upper()} Energy Plot", y=1.02)
                plt.tight_layout()
                plt.savefig(
                    diag_dir / "energy.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()
            except Exception:
                warnings.warn(f"Failed to create energy plot for {sampler_name}")


def create_combined_convergence_plot(
    inference_data: Dict[str, az.InferenceData], output_dir: Path
) -> None:
    """Create combined LLC convergence plots vs FGEs and Time, including 95% CI."""
    if not inference_data:
        return

    # Create diagnostics subdirectory
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    # Setup figure with two subplots (FGEs and Time), sharing the Y-axis
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    ax_fge, ax_time = axes

    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(inference_data)))

    for i, (sampler_name, idata) in enumerate(inference_data.items()):
        color = colors[i % len(colors)]
        llc_traces = idata.posterior["llc"].values  # (C, T)
        C, T = llc_traces.shape

        if T < 2 or C < 1:
            continue

        # Calculate running mean for each chain
        running_means = np.cumsum(llc_traces, axis=1) / np.arange(1, T + 1)  # (C, T)

        # Calculate overall running mean (mean across chains)
        mean_running_mean = running_means.mean(axis=0)  # (T,)

        # Calculate Standard Error of the Mean (SEM) using cross-chain variance
        # Std across chains (use ddof=1 if C>1 for sample standard deviation)
        ddof = 1 if C > 1 else 0
        running_std_chains = running_means.std(axis=0, ddof=ddof)  # (T,)
        # SEM = Std / sqrt(C)
        running_sem = running_std_chains / np.sqrt(C)  # (T,)

        # Calculate 95% CI bounds (Mean +/- 1.96 * SEM)
        ci_lower = mean_running_mean - 1.96 * running_sem
        ci_upper = mean_running_mean + 1.96 * running_sem

        # Define a starting index to avoid plotting noisy CI early on
        start_idx = min(10, T - 1)

        # --- Plot vs FGEs ---
        if hasattr(idata, "sample_stats") and "cumulative_fge" in idata.sample_stats:
            # Use mean across chains for the X axis
            fges = idata.sample_stats["cumulative_fge"].values.mean(axis=0)
            if fges.shape[0] == T:
                ax_fge.plot(fges, mean_running_mean, label=sampler_name.upper(), color=color)
                ax_fge.fill_between(
                    fges[start_idx:],
                    ci_lower[start_idx:],
                    ci_upper[start_idx:],
                    color=color,
                    alpha=0.2,
                )

        # --- Plot vs Time ---
        if hasattr(idata, "sample_stats") and "cumulative_time" in idata.sample_stats:
            # Use mean across chains for the X axis
            times = idata.sample_stats["cumulative_time"].values.mean(axis=0)
            if times.shape[0] == T:
                ax_time.plot(times, mean_running_mean, label=sampler_name.upper(), color=color)
                ax_time.fill_between(
                    times[start_idx:],
                    ci_lower[start_idx:],
                    ci_upper[start_idx:],
                    color=color,
                    alpha=0.2,
                )

    # Finalize plots
    ax_fge.set_title("LLC Convergence vs. Computational Work (FGEs)")
    ax_fge.set_xlabel("Full-Data Gradient Evaluations (FGEs)")
    ax_fge.set_ylabel("LLC Estimate (Running Mean ± 95% CI)")
    ax_fge.legend()
    ax_fge.grid(True, alpha=0.3)
    ax_fge.set_xscale("log")  # Log scale for FGEs

    ax_time.set_title("LLC Convergence vs. Wall-clock Time")
    ax_time.set_xlabel("Time (seconds)")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xscale("log")  # Log scale for Time

    plt.tight_layout()
    plt.savefig(diag_dir / "llc_convergence_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_work_normalized_variance_plot(
    inference_data: Dict[str, az.InferenceData], output_dir: Path
) -> None:
    """Create plots of Work-Normalized Variance (WNV) vs Work."""
    if not inference_data:
        return

    # Create diagnostics subdirectory
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    ax_fge, ax_time = axes
    colors = plt.cm.tab10(np.linspace(0, 1, len(inference_data)))

    for i, (sampler_name, idata) in enumerate(inference_data.items()):
        color = colors[i % len(colors)]
        llc_traces = idata.posterior["llc"].values  # (C, T)
        T = llc_traces.shape[1]
        C = llc_traces.shape[0]

        if T < 5 or C < 2:
            warnings.warn(
                f"Skipping WNV for {sampler_name}: "
                f"Requires T>=5 and C>=2 for reliable variance estimate."
            )
            continue

        # Estimate the variance of the estimator using cross-chain variance
        running_means = np.cumsum(llc_traces, axis=1) / np.arange(1, T + 1)  # (C, T)
        # Var(lambda_hat_t) ≈ Var_chains(running_means_t) / C
        # Use ddof=1 for sample variance
        variance_estimate = running_means.var(axis=0, ddof=1) / C  # (T,)

        # Smooth the variance estimate as it can be noisy
        if T > 20:
            window_size = max(5, int(T * 0.1))  # 10% window size
            # Use pandas rolling mean for smoothing
            variance_estimate = (
                pd.Series(variance_estimate)
                .rolling(window=window_size, min_periods=5)
                .mean()
                .values
            )

        # --- Plot vs FGEs ---
        if hasattr(idata, "sample_stats") and "cumulative_fge" in idata.sample_stats:
            fges = idata.sample_stats["cumulative_fge"].values.mean(axis=0)
            if fges.shape[0] == T:
                # WNV = Variance * Work
                wnv_fge = variance_estimate * fges
                ax_fge.plot(fges, wnv_fge, label=sampler_name.upper(), color=color)

        # --- Plot vs Time ---
        if hasattr(idata, "sample_stats") and "cumulative_time" in idata.sample_stats:
            times = idata.sample_stats["cumulative_time"].values.mean(axis=0)
            if times.shape[0] == T:
                wnv_time = variance_estimate * times
                ax_time.plot(times, wnv_time, label=sampler_name.upper(), color=color)

    # Finalize plots
    ax_fge.set_title("Work-Normalized Variance (WNV) vs. FGEs (Smoothed)")
    ax_fge.set_xlabel("FGEs")
    ax_fge.set_ylabel("WNV (Variance × Work)")
    ax_fge.legend()
    ax_fge.grid(True, alpha=0.3)
    ax_fge.set_xscale("log")
    ax_fge.set_yscale("log")  # WNV on log scale

    ax_time.set_title("WNV vs. Time (Smoothed)")
    ax_time.set_xlabel("Time (seconds)")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xscale("log")

    plt.tight_layout()
    plt.savefig(diag_dir / "llc_wnv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_comparison_plot(analysis_results: Dict[str, Dict], output_dir: Path) -> None:
    """Create comparison bar chart of LLC estimates across samplers."""
    if not analysis_results:
        return

    samplers = list(analysis_results.keys())
    llc_means = [analysis_results[s].get("llc_mean", 0.0) for s in samplers]
    llc_sems = [analysis_results[s].get("llc_sem", 0.0) for s in samplers]

    # Create bar plot with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(samplers, llc_means, yerr=llc_sems, capsize=5, alpha=0.7)

    # Color bars differently
    colors = plt.cm.tab10(np.linspace(0, 1, len(samplers)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_ylabel("LLC Estimate (Mean ± SEM)")
    ax.set_title("LLC Estimates Across Samplers")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "llc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_summary_table(analysis_results: Dict[str, Dict], output_dir: Path) -> None:
    """Create and save summary table of results."""
    if not analysis_results:
        return

    # Create DataFrame
    df = pd.DataFrame(analysis_results).T
    df.to_csv(output_dir / "metrics.csv")

    # Create formatted string summary
    summary_text = "LLC Experiment Summary\n"
    summary_text += "=" * 50 + "\n\n"

    for sampler_name, metrics in analysis_results.items():
        summary_text += f"{sampler_name.upper()}:\n"
        summary_text += f"  LLC Mean: {metrics.get('llc_mean', 0.0):.6f}\n"
        summary_text += f"  LLC SEM:  {metrics.get('llc_sem', 0.0):.6f}\n"  # Standard Error of Mean
        summary_text += f"  R-hat:    {metrics.get('r_hat', 1.0):.4f}\n"
        summary_text += "-" * 20 + "\n"
        summary_text += f"  ESS:      {metrics.get('ess', 0.0):.1f}\n"
        summary_text += f"  Total FGEs: {metrics.get('total_fge', 0.0):.1f}\n"
        summary_text += f"  Time (s):   {metrics.get('elapsed_time', 0.0):.2f}\n"
        # WNV (Work-Normalized Variance)
        summary_text += f"  WNV (FGE):  {metrics.get('wnv_fge', np.nan):.4f}\n"
        summary_text += f"  WNV (Time): {metrics.get('wnv_time', np.nan):.4f}\n"
        # Efficiency (ESS / Work)
        summary_text += f"  Eff (FGE):  {metrics.get('efficiency_fge', np.nan):.4f}\n"
        summary_text += f"  Eff (Time): {metrics.get('efficiency_time', np.nan):.4f}\n\n"

    # Save summary text
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary_text)


def compute_target_diagnostics(
    cfg,
    X: np.ndarray,
    Y: np.ndarray,
    model,
    outdir: Path,
) -> Dict[str, float]:
    """Write simple problem-level diagnostics and return scalar metrics.

    Args:
        cfg: Full OmegaConf/Config object used for Stage A
            (must have .data, .teacher, .target.seed).
        X, Y: Training data arrays from the target artifact.
        model: Trained Equinox model (ERM solution).
        outdir: Directory to write PNGs into (created if needed).

    Returns:
        Dict with scalar metrics (MSEs) to embed into meta.json.
    """
    import time

    import jax

    t0 = time.time()
    log_local = logging.getLogger(__name__)
    log_local.info("[diagnostics] starting; X=%s Y=%s", X.shape, Y.shape)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_local.info("[diagnostics] created outdir in %.3fs", time.time() - t0)
    t0 = time.time()

    # Rebuild teacher with the same RNG convention as make_dataset
    seed = int(cfg.target.seed)
    key = jax.random.PRNGKey(seed)
    kx_train, k_teacher, k_noise = random.split(key, 3)

    teacher_params, teacher_forward = build_teacher(k_teacher, cfg)
    log_local.info("[diagnostics] built teacher in %.3fs", time.time() - t0)
    t0 = time.time()

    X_train = jnp.asarray(X)
    Y_train = jnp.asarray(Y)

    # Teacher predictions on training inputs (ignore dropout for smooth reference)
    y_teacher_train = teacher_forward(X_train)
    log_local.info("[diagnostics] teacher(train) in %.3fs", time.time() - t0)
    t0 = time.time()

    y_pred_train = model(X_train)
    log_local.info("[diagnostics] student(train) in %.3fs", time.time() - t0)
    t0 = time.time()

    mse_train_noise = float(jnp.mean((y_pred_train - Y_train) ** 2))
    mse_train_teacher = float(jnp.mean((y_pred_train - y_teacher_train) ** 2))

    # Fresh test set from same x_dist + noise model
    n_test = int(getattr(cfg.data, "n_test", min(int(X_train.shape[0]), 2000)))
    kx_test, k_noise_test = random.split(k_noise)
    X_test = sample_X(kx_test, cfg, n_test, cfg.model.in_dim)
    log_local.info("[diagnostics] sample_X test in %.3fs (n_test=%d)", time.time() - t0, n_test)
    t0 = time.time()

    y_teacher_test = teacher_forward(X_test)
    log_local.info("[diagnostics] teacher(test) in %.3fs", time.time() - t0)
    t0 = time.time()

    Y_test = add_noise(k_noise_test, y_teacher_test, cfg, X_test)
    log_local.info("[diagnostics] add_noise in %.3fs", time.time() - t0)
    t0 = time.time()

    y_pred_test = model(X_test)
    log_local.info("[diagnostics] student(test) in %.3fs", time.time() - t0)
    t0 = time.time()

    mse_test_noise = float(jnp.mean((y_pred_test - Y_test) ** 2))
    mse_test_teacher = float(jnp.mean((y_pred_test - y_teacher_test) ** 2))

    # 1) Train vs test bar plot
    fig, ax = plt.subplots()
    ax.bar(
        ["train/noise", "test/noise", "train/teacher", "test/teacher"],
        [mse_train_noise, mse_test_noise, mse_train_teacher, mse_test_teacher],
    )
    ax.set_ylabel("MSE")
    ax.set_title("Train vs test MSE (student vs noisy labels / teacher)")
    fig.tight_layout()
    fig.savefig(outdir / "target_train_test_loss.png", dpi=150)
    plt.close(fig)
    log_local.info("[diagnostics] wrote bar plot in %.3fs", time.time() - t0)
    t0 = time.time()

    # 2-3) Pred vs teacher scatter plots
    def _scatter(true_vals, pred_vals, fname: str, title: str):
        true_np = np.asarray(true_vals).ravel()
        pred_np = np.asarray(pred_vals).ravel()
        fig, ax = plt.subplots()
        ax.scatter(true_np, pred_np, s=5, alpha=0.4)
        lo = float(min(true_np.min(), pred_np.min()))
        hi = float(max(true_np.max(), pred_np.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("teacher(x)")
        ax.set_ylabel("student(x)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)

    _scatter(
        y_teacher_train,
        y_pred_train,
        "target_pred_vs_teacher_train.png",
        "Train: student vs teacher",
    )
    log_local.info("[diagnostics] wrote train scatter in %.3fs", time.time() - t0)
    t0 = time.time()

    _scatter(
        y_teacher_test,
        y_pred_test,
        "target_pred_vs_teacher_test.png",
        "Test: student vs teacher",
    )
    log_local.info("[diagnostics] wrote test scatter in %.3fs", time.time() - t0)
    log_local.info("[diagnostics] COMPLETE")

    return {
        "train_mse_noise": mse_train_noise,
        "test_mse_noise": mse_test_noise,
        "train_mse_teacher": mse_train_teacher,
        "test_mse_teacher": mse_test_teacher,
    }
