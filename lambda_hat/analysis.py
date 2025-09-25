# llc/analysis.py
"""Enhanced analysis functions for Hydra-based LLC experiments with proper visualization"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import jax.numpy as jnp
import numpy as np
import arviz as az
from pathlib import Path
import warnings
import pandas as pd
import matplotlib.pyplot as plt


def analyze_traces(
    traces: Dict[str, jnp.ndarray],
    L0: float,
    n_data: int,
    beta: float,
    warmup: int = 0,
) -> Tuple[Dict[str, float], az.InferenceData]:
    """Analyze sampling traces, compute LLC metrics, and create InferenceData.

    Args:
        traces: Dictionary of traces (vmapped: C, T).
        L0, n_data, beta: Parameters for LLC calculation.
        warmup: Number of recorded draws to discard (burn-in).

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
        warnings.warn(f"Warmup draws ({warmup}) >= total draws ({draws}). Using all samples.")
        warmup = 0

    # Select post-warmup data
    Ln_post_warmup = Ln_values[:, warmup:]
    draws_post_warmup = draws - warmup

    # Compute LLC values: n * beta * (L_n(w) - L0)
    llc_values = float(n_data) * float(beta) * (Ln_post_warmup - L0)
    llc_values_np = np.array(llc_values)

    # Prepare posterior data
    posterior_data = {"llc": llc_values_np, "L": np.array(Ln_post_warmup)}

    # Handle sample statistics (FGEs, acceptance, etc.)
    sample_stats_data = {}

    # Extract FGEs (tracked during sampling)
    if "cumulative_fge" in traces:
        # Ensure FGE data is extracted as float64 numpy array
        fge_post_warmup = traces["cumulative_fge"][:, warmup:]
        sample_stats_data["cumulative_fge"] = np.array(fge_post_warmup, dtype=np.float64)

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
    metrics = _compute_metrics_from_idata(idata, llc_values_np)
    return metrics, idata

def _compute_metrics_from_idata(idata: az.InferenceData, llc_values_np: np.ndarray) -> Dict[str, float]:
    """Helper to compute metrics from InferenceData."""
    metrics = {
        "llc_mean": float(llc_values_np.mean()),
        "llc_std": float(llc_values_np.std()),
        "llc_min": float(llc_values_np.min()),
        "llc_max": float(llc_values_np.max()),
    }
    # Use ArviZ summary for diagnostics
    summary = az.summary(idata, var_names=["llc"])
    if not summary.empty:
        metrics["ess_bulk"] = float(summary.get("ess_bulk", np.nan).iloc[0])
        metrics["ess_tail"] = float(summary.get("ess_tail", np.nan).iloc[0])
        metrics["r_hat"] = float(summary.get("r_hat", np.nan).iloc[0])
        # Use minimum of bulk and tail ESS
        ess = min(metrics.get("ess_bulk", np.nan), metrics.get("ess_tail", np.nan))
        metrics["ess"] = ess if not np.isnan(ess) else np.nan
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
            plt.savefig(diag_dir / f"{sampler_name}_trace.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            warnings.warn(f"Failed to create trace plot for {sampler_name}")

        # 2. Rank Plot (Convergence check)
        try:
            az.plot_rank(idata, var_names=["llc"], figsize=(12, 5))
            plt.suptitle(f"{sampler_name.upper()} Rank Plot", y=1.02)
            plt.tight_layout()
            plt.savefig(diag_dir / f"{sampler_name}_rank.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
             warnings.warn(f"Failed to create rank plot for {sampler_name}")

        # 3. Energy Plot (if available, useful for HMC/MCLMC)
        if hasattr(idata, 'sample_stats') and 'energy' in idata.sample_stats:
            try:
                az.plot_energy(idata, figsize=(12, 6))
                plt.suptitle(f"{sampler_name.upper()} Energy Plot", y=1.02)
                plt.tight_layout()
                plt.savefig(diag_dir / f"{sampler_name}_energy.png", dpi=150, bbox_inches="tight")
                plt.close()
            except Exception:
                warnings.warn(f"Failed to create energy plot for {sampler_name}")

def create_convergence_plots(
    inference_data: Dict[str, az.InferenceData], output_dir: Path
) -> None:
    """Create LLC convergence plots (Running Mean) vs FGEs."""
    num_samplers = len(inference_data)
    if num_samplers == 0:
        return

    # Create a combined figure
    fig, axes = plt.subplots(num_samplers, 1, figsize=(12, 4 * num_samplers), squeeze=False)
    axes = axes.flatten()

    for i, (sampler_name, idata) in enumerate(inference_data.items()):
        ax = axes[i]
        llc_traces = idata.posterior["llc"].values # (C, T)
        T = llc_traces.shape[1]

        # Calculate running mean for each chain
        running_means = np.cumsum(llc_traces, axis=1) / np.arange(1, T + 1)

        # Determine X-axis (FGEs or Draws)
        # FGEs are stored in sample_stats, shape (C, T)
        if hasattr(idata, 'sample_stats') and 'cumulative_fge' in idata.sample_stats:
            fges = idata.sample_stats["cumulative_fge"].values
            xlabel = "Full-Data Gradient Evaluations (FGEs)"
            use_fge = True
        else:
            x_axis_draws = np.arange(1, T + 1)
            xlabel = "Draws"
            use_fge = False

        # Plot running means
        for chain_idx in range(running_means.shape[0]):
            if use_fge:
                # Ensure shapes match if using FGEs (they should)
                if fges.shape[1] == T:
                    ax.plot(fges[chain_idx], running_means[chain_idx], alpha=0.7)
                else:
                     warnings.warn(f"FGE shape mismatch for {sampler_name}. Falling back to Draws.")
                     ax.plot(x_axis_draws, running_means[chain_idx], alpha=0.7)
                     xlabel = "Draws"
            else:
                ax.plot(x_axis_draws, running_means[chain_idx], alpha=0.7)


        # Plot the final mean (the target)
        final_mean = llc_traces.mean()
        ax.axhline(final_mean, color='k', linestyle='--', label=f"Target (Final Mean): {final_mean:.4f}")

        ax.set_title(f"{sampler_name.upper()} LLC Convergence")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("LLC Estimate (Running Mean)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Use log scale for X axis if the range is large
        # Check the last value of the first chain (or the draws axis)
        if use_fge and fges.shape[1] > 0 and fges[0, -1] > 5000:
            ax.set_xscale('log')
        elif not use_fge and T > 5000:
            ax.set_xscale('log')

    plt.tight_layout()
    # Save the combined convergence plot (replaces the old uninformative llc_traces.png)
    plt.savefig(output_dir / "llc_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_comparison_plot(analysis_results: Dict[str, Dict], output_dir: Path) -> None:
    """Create comparison plot of LLC means across samplers

    Args:
        analysis_results: Analysis results with metrics
        output_dir: Directory to save plots
    """
    samplers = list(analysis_results.keys())
    if not samplers:
        return

    means = [analysis_results[s].get("llc_mean", 0.0) for s in samplers]
    stds = [analysis_results[s].get("llc_std", 0.0) for s in samplers]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(samplers, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_ylabel("LLC Mean ± Std")
    ax.set_title("LLC Comparison Across Samplers")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "llc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_summary_table(analysis_results: Dict[str, Dict], output_dir: Path) -> None:
    """Create summary table of metrics

    Args:
        analysis_results: Analysis results with metrics
        output_dir: Directory to save table
    """

    if not analysis_results:
        return

    # Convert to DataFrame
    df = pd.DataFrame(analysis_results).T

    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(6)

    # Save as CSV
    df.to_csv(output_dir / "metrics_summary.csv")

    # Create formatted string summary
    summary_text = "LLC Experiment Summary\n"
    summary_text += "=" * 50 + "\n\n"

    for sampler_name, metrics in analysis_results.items():
        summary_text += f"{sampler_name.upper()}:\n"
        summary_text += f"  LLC Mean: {metrics.get('llc_mean', 0.0):.6f}\n"
        summary_text += f"  LLC Std:  {metrics.get('llc_std', 0.0):.6f}\n"
        summary_text += f"  ESS:      {metrics.get('ess', 0.0):.1f}\n"
        summary_text += f"  R-hat:    {metrics.get('r_hat', 1.0):.4f}\n\n"

    # Save summary text
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary_text)
