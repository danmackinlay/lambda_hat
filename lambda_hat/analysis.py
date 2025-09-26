# llc/analysis.py
"""Simplified analysis functions for Hydra-based LLC experiments"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import warnings


def analyze_traces(
    traces: Dict[str, jnp.ndarray],
    L0: float,
    n_data: int,
    beta: float,
    warmup: int = 0,
    timings: Dict[str, float] = None,
) -> Tuple[Dict[str, float], az.InferenceData]:
    """Analyze sampling traces, compute LLC metrics, and create InferenceData.

    Args:
        traces: Dictionary of traces (vmapped: C, T).
        L0, n_data, beta: Parameters for LLC calculation.
        warmup: Number of recorded draws to discard (burn-in).
        timings: Dictionary of timing information.

    Returns:
        Tuple of (metrics dictionary, ArviZ InferenceData).
    """
    if "Ln" not in traces:
        raise ValueError("Traces must contain 'Ln' key.")

    # Handle input shapes (Ensure C, T)
    Ln_values = traces["Ln"]
    if Ln_values.ndim == 1:
        Ln_values = Ln_values[None, :]
    chains, draws = Ln_values.shape

    # Validate and apply warmup (burn-in)
    if warmup >= draws:
        warnings.warn(
            f"Warmup draws ({warmup}) >= total draws ({draws}). Using all samples."
        )
        warmup = 0

    # Select post-warmup data
    Ln_post_warmup = Ln_values[:, warmup:]
    draws_post_warmup = draws - warmup

    # Compute LLC values: n * beta * (L_n(w) - L0)
    llc_values = float(n_data) * float(beta) * (Ln_post_warmup - L0)
    llc_values_np = np.array(llc_values)

    # Prepare posterior data
    posterior_data = {"llc": llc_values_np, "L": np.array(Ln_post_warmup)}

    # Handle sample statistics (FGEs, acceptance, time, etc.)
    sample_stats_data = {}

    # Extract FGEs (tracked during sampling)
    if "cumulative_fge" in traces:
        # Ensure FGE data is extracted as float64 numpy array
        fge_post_warmup = traces["cumulative_fge"][:, warmup:]
        sample_stats_data["cumulative_fge"] = np.array(
            fge_post_warmup, dtype=np.float64
        )

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
                time_per_draw = (
                    sampling_time / draws_post_warmup if draws_post_warmup > 0 else 0.0
                )
                cumulative_time_post_warmup = (
                    np.arange(1, draws_post_warmup + 1) * time_per_draw
                )
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

    # Create ArviZ InferenceData (contains only post-warmup data)
    data = {
        "posterior": posterior_data,
        "sample_stats": sample_stats_data if sample_stats_data else None,
        "coords": {"chain": np.arange(chains), "draw": np.arange(draws_post_warmup)},
        "dims": {"llc": ["chain", "draw"], "L": ["chain", "draw"]},
    }
    idata = az.from_dict(**data)

    # Compute metrics
    metrics = _compute_metrics_from_idata(idata, llc_values_np, timings)
    return metrics, idata


def _compute_metrics_from_idata(
    idata: az.InferenceData,
    llc_values_np: np.ndarray,
    timings: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Helper to compute metrics from InferenceData."""
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

    # Use ArviZ summary for diagnostics
    summary = az.summary(idata, var_names=["llc"])
    if not summary.empty:
        metrics["ess_bulk"] = float(summary.get("ess_bulk", np.nan).iloc[0])
        metrics["ess_tail"] = float(summary.get("ess_tail", np.nan).iloc[0])
        metrics["r_hat"] = float(summary.get("r_hat", np.nan).iloc[0])
        ess = min(metrics.get("ess_bulk", np.nan), metrics.get("ess_tail", np.nan))
        metrics["ess"] = ess if not np.isnan(ess) else np.nan

        # Calculate Efficiency and WNV
        if not np.isnan(ess) and ess > 0:
            # Variance of the estimate (SEM squared) = Var(samples) / ESS
            variance_estimate = metrics["llc_std"] ** 2 / ess
            metrics["llc_sem"] = np.sqrt(
                variance_estimate
            )  # Standard Error of the Mean

            if total_fge > 0:
                # Efficiency = ESS / Work
                metrics["efficiency_fge"] = ess / total_fge
                # WNV = Variance(Estimate) * Work
                metrics["wnv_fge"] = variance_estimate * total_fge

            if total_time > 0:
                metrics["efficiency_time"] = ess / total_time
                metrics["wnv_time"] = variance_estimate * total_time

    return metrics


def create_arviz_diagnostics(
    inference_data: Dict[str, az.InferenceData], output_dir: Path
) -> None:
    """Create ArviZ diagnostic plots (Trace, Rank, Energy)."""
    # Create a subdirectory for detailed plots
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    for sampler_name, idata in inference_data.items():
        # 1. Trace Plot (Trace + Posterior Density)
        try:
            az.plot_trace(idata, var_names=["llc", "L"], figsize=(12, 8), compact=False)
            plt.suptitle(f"{sampler_name.upper()} Trace Plot", y=1.02)
            plt.tight_layout()
            plt.savefig(
                diag_dir / f"{sampler_name}_trace.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
        except Exception:
            warnings.warn(f"Failed to create trace plot for {sampler_name}")

        # 2. Rank Plot (Convergence check)
        try:
            az.plot_rank(idata, var_names=["llc"], figsize=(12, 5))
            plt.suptitle(f"{sampler_name.upper()} Rank Plot", y=1.02)
            plt.tight_layout()
            plt.savefig(
                diag_dir / f"{sampler_name}_rank.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
        except Exception:
            warnings.warn(f"Failed to create rank plot for {sampler_name}")

        # 3. Energy Plot (if available, useful for HMC/MCLMC)
        if hasattr(idata, "sample_stats") and "energy" in idata.sample_stats:
            try:
                az.plot_energy(idata, figsize=(12, 6))
                plt.suptitle(f"{sampler_name.upper()} Energy Plot", y=1.02)
                plt.tight_layout()
                plt.savefig(
                    diag_dir / f"{sampler_name}_energy.png",
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
                ax_fge.plot(
                    fges, mean_running_mean, label=sampler_name.upper(), color=color
                )
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
                ax_time.plot(
                    times, mean_running_mean, label=sampler_name.upper(), color=color
                )
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
    plt.savefig(
        output_dir / "llc_convergence_combined.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def create_work_normalized_variance_plot(
    inference_data: Dict[str, az.InferenceData], output_dir: Path
) -> None:
    """Create plots of Work-Normalized Variance (WNV) vs Work."""
    if not inference_data:
        return

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
                f"Skipping WNV for {sampler_name}: Requires T>=5 and C>=2 for reliable variance estimate."
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
    plt.savefig(output_dir / "llc_wnv.png", dpi=150, bbox_inches="tight")
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
        summary_text += (
            f"  LLC SEM:  {metrics.get('llc_sem', 0.0):.6f}\n"  # Standard Error of Mean
        )
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
        summary_text += (
            f"  Eff (Time): {metrics.get('efficiency_time', np.nan):.4f}\n\n"
        )

    # Save summary text
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary_text)
