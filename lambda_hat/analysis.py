# llc/analysis.py
"""Simplified analysis functions for Hydra-based LLC experiments"""

from __future__ import annotations
from typing import Dict, Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
from pathlib import Path
# Updated to use new JAX tree API (jax>=0.4.28)
import warnings


def compute_llc_metrics(
    Ln_values: jnp.ndarray,
    L0: float,
    n_data: int,
    beta: float,
    warmup: int = 0,
) -> Dict[str, float]:
    """Compute LLC metrics from pre-computed Ln values (MEMORY EFFICIENT).

    Implements the estimator: hat{lambda} = n * beta * (E[L_n(w)] - L0)

    Args:
        Ln_values: Pre-computed loss values, shape (chains, draws) or (draws,) for single chain
        L0: Reference loss value (loss at ERM solution)
        n_data: Dataset size (n)
        beta: Inverse temperature (beta)
        warmup: Number of warmup samples to discard from the beginning

    Returns:
        Dictionary with LLC statistics (hat{lambda})
    """
    # Handle both (chains, draws) and (draws,) shapes
    if Ln_values.ndim == 1:
        # Single chain: reshape to (1, draws)
        Ln_values = Ln_values[None, :]
        chains, draws = 1, Ln_values.shape[1]
    else:
        chains, draws = Ln_values.shape

    # Apply warmup: discard initial samples efficiently
    if warmup >= draws:
        # If warmup is too large, use all draws but warn
        warnings.warn(
            f"Warmup ({warmup}) >= total draws ({draws}). Using all samples without warmup."
        )
        warmup = 0

    if warmup > 0:
        Ln_values = Ln_values[:, warmup:]
        draws = draws - warmup

    # Compute LLC values directly from Ln: hat{lambda} = n * beta * (L_n(w) - L0)
    llc_values = float(n_data) * float(beta) * (Ln_values - L0)

    # Convert to numpy for ArviZ
    llc_values_np = np.array(llc_values)

    # Create ArviZ InferenceData for ESS computation
    idata = az.convert_to_inference_data(
        {
            "llc": llc_values_np[..., None]  # Add dummy dimension for ArviZ
        }
    )

    # Compute summary statistics
    summary = az.summary(idata, var_names=["llc"])

    # Extract metrics
    metrics = {
        "llc_mean": float(llc_values_np.mean()),
        "llc_std": float(llc_values_np.std()),
        "llc_min": float(llc_values_np.min()),
        "llc_max": float(llc_values_np.max()),
    }

    # Add ESS and R-hat if available
    if not summary.empty:
        metrics["ess_bulk"] = float(summary.get("ess_bulk", np.nan).iloc[0])
        metrics["ess_tail"] = float(summary.get("ess_tail", np.nan).iloc[0])
        metrics["r_hat"] = float(summary.get("r_hat", np.nan).iloc[0])

        # Use minimum of bulk and tail ESS as overall ESS, handling potential NaNs
        ess_bulk = metrics["ess_bulk"]
        ess_tail = metrics["ess_tail"]
        if np.isnan(ess_bulk) or np.isnan(ess_tail):
            metrics["ess"] = np.nan
        else:
            metrics["ess"] = min(ess_bulk, ess_tail)

    return metrics


# Legacy compute_llc_metrics function deleted - now using memory-efficient compute_llc_metrics (renamed from compute_llc_from_Ln)


def analyze_from_Ln_dict(
    Ln_histories: Dict[str, jnp.ndarray],
    L0: float,
    n_data: int,
    beta: float,
    warmup: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Analyze multiple samplers from their pre-computed Ln histories (MEMORY EFFICIENT).

    Args:
        Ln_histories: Dict[sampler_name, Ln_values] where Ln_values has shape (chains, draws)
        L0: Reference loss value
        n_data: Dataset size
        beta: Inverse temperature
        warmup: Number of warmup samples to discard

    Returns:
        Dict[sampler_name, metrics_dict]
    """
    analysis_results = {}
    for sampler_name, Ln_values in Ln_histories.items():
        analysis_results[sampler_name] = compute_llc_metrics(
            Ln_values, L0, n_data, beta, warmup
        )
    return analysis_results


def create_trace_plots(
    Ln_histories: Dict[str, jnp.ndarray],
    analysis_results: Dict[str, Dict],
    output_dir: Path,
    L0: float,
    n_data: int,
    beta: float,
) -> None:
    """Create trace plots for LLC values from pre-computed Ln histories (MEMORY EFFICIENT).

    Args:
        Ln_histories: Dict[sampler_name, Ln_values] where Ln_values has shape (chains, draws)
        analysis_results: Analysis results with metrics
        output_dir: Directory to save plots
        L0: Reference loss value
        n_data: Dataset size
        beta: Inverse temperature
    """
    if not Ln_histories:
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(Ln_histories), 1, figsize=(10, 3 * len(Ln_histories)))
    if len(Ln_histories) == 1:
        axes = [axes]

    for i, (sampler_name, Ln_values) in enumerate(Ln_histories.items()):
        ax = axes[i]

        # Convert Ln to LLC values: llc = n * beta * (Ln - L0)
        llc_values = float(n_data) * float(beta) * (Ln_values - L0)
        llc_values_np = np.array(llc_values)

        # Get chains and draws
        if llc_values_np.ndim == 1:
            chains, draws = 1, llc_values_np.shape[0]
            llc_values_np = llc_values_np[None, :]  # Add chain dimension
        else:
            chains, draws = llc_values_np.shape

        # Plot traces
        for chain in range(chains):
            ax.plot(
                llc_values_np[chain],
                alpha=0.7,
                linewidth=0.8,
                label=f"Chain {chain + 1}",
            )

        # Get metrics from analysis results
        metrics = analysis_results.get(sampler_name, {})
        llc_mean = metrics.get("llc_mean", 0.0)
        ess = metrics.get("ess", 0.0)

        ax.axhline(
            llc_mean,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Mean: {llc_mean:.4f}",
        )
        ax.set_title(f"{sampler_name.upper()} LLC Traces (ESS: {ess:.1f})")
        ax.set_xlabel("Draw")
        ax.set_ylabel("LLC")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "llc_traces.png", dpi=150, bbox_inches="tight")
    plt.close()


# Legacy create_trace_plots function deleted - now using memory-efficient create_trace_plots (renamed from create_trace_plots_from_Ln)


def create_comparison_plot(analysis_results: Dict[str, Dict], output_dir: Path) -> None:
    """Create comparison plot of LLC means across samplers

    Args:
        analysis_results: Analysis results with metrics
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt

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
    import pandas as pd

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
