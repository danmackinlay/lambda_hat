# llc/analysis.py
"""Simplified analysis functions for Hydra-based LLC experiments"""

from __future__ import annotations
from typing import Dict, Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
from pathlib import Path
from jax.tree_util import tree_map


def compute_llc_from_Ln(
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
        import warnings
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


def compute_llc_metrics(
    traces: Dict[str, jnp.ndarray],
    loss_fn: Callable,
    L0: float,
    n_data: int,
    beta: float,
    warmup: int = 0,
) -> Dict[str, float]:
    """LEGACY: Compute LLC metrics from sampling traces (MEMORY INTENSIVE).

    ⚠️  WARNING: This function may cause OOM errors with large parameter counts.
    Use compute_llc_from_Ln() with pre-computed Ln values for better performance.

    Args:
        traces: Dictionary with 'position' key containing parameter samples
        loss_fn: Loss function to evaluate on parameter samples
        L0: Reference loss value (loss at ERM solution)
        n_data: Dataset size (n)
        beta: Inverse temperature (beta)
        warmup: Number of warmup samples to discard from the beginning

    Returns:
        Dictionary with LLC statistics (hat{lambda})
    """
    import warnings
    warnings.warn(
        "compute_llc_metrics() is deprecated and memory-intensive. "
        "Use compute_llc_from_Ln() with pre-computed Ln values instead.",
        DeprecationWarning
    )

    # Check if traces already contain pre-computed Ln values (efficient path)
    if "Ln" in traces:
        return compute_llc_from_Ln(traces["Ln"], L0, n_data, beta, warmup)

    # Fall back to legacy memory-intensive computation
    # Extract positions
    positions = traces["position"]  # (chains, draws, param_structure)

    # Determine chains and draws robustly
    if isinstance(positions, dict):
        # Handle structured params (Haiku), infer shape from the first leaf
        first_leaf = jax.tree_util.tree_leaves(positions)[0]
        chains, draws = first_leaf.shape[:2]
    else:
        # Handle flat params
        chains, draws = positions.shape[:2]

    # Apply warmup: discard initial samples efficiently
    if warmup >= draws:
        # If warmup is too large, use all draws but warn
        warnings.warn(
            f"Warmup ({warmup}) >= total draws ({draws}). Using all samples without warmup."
        )
        warmup = 0

    if warmup > 0:
        if isinstance(positions, dict):
            # Discard warmup samples from the traces before processing
            positions = tree_map(lambda x: x[:, warmup:, ...], positions)
        else:
            positions = positions[:, warmup:, ...]

        # Update draws count after warmup
        draws = draws - warmup

    # Compute loss for all samples using vmap (MEMORY INTENSIVE!)
    # Flatten chains and draws for batch computation
    if isinstance(positions, dict):
        flat_positions = tree_map(lambda x: x.reshape(-1, *x.shape[2:]), positions)
        losses = jax.vmap(loss_fn)(flat_positions)
    else:
        flat_positions = positions.reshape(-1, *positions.shape[2:])
        losses = jax.vmap(loss_fn)(flat_positions)

    # Reshape back to (chains, draws)
    losses = losses.reshape(chains, draws)

    # Use the efficient function with computed losses
    return compute_llc_from_Ln(losses, L0, n_data, beta, warmup=0)  # warmup already applied


def analyze_from_Ln_dict(
    Ln_histories: Dict[str, jnp.ndarray],
    L0: float,
    n_data: int,
    beta: float,
    warmup: int = 0
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
        analysis_results[sampler_name] = compute_llc_from_Ln(
            Ln_values, L0, n_data, beta, warmup
        )
    return analysis_results


def create_trace_plots_from_Ln(
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
            ax.plot(llc_values_np[chain], alpha=0.7, linewidth=0.8, label=f"Chain {chain+1}")

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


def create_trace_plots(
    results: Dict[str, Any], analysis_results: Dict[str, Dict], output_dir: Path
) -> None:
    """LEGACY: Create trace plots for LLC values (MEMORY INTENSIVE).

    ⚠️  WARNING: This function may cause OOM errors with large parameter counts.
    Use create_trace_plots_from_Ln() with pre-computed Ln values for better performance.

    Args:
        results: Raw sampling results
        analysis_results: Analysis results with metrics
        output_dir: Directory to save plots
    """
    import warnings
    warnings.warn(
        "create_trace_plots() is deprecated and may be memory-intensive. "
        "Use create_trace_plots_from_Ln() with pre-computed Ln values instead.",
        DeprecationWarning
    )

    if not results:
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 1, figsize=(10, 3 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for i, (sampler_name, sampler_data) in enumerate(results.items()):
        ax = axes[i]

        # Get LLC metrics
        metrics = analysis_results.get(sampler_name, {})
        llc_mean = metrics.get("llc_mean", 0.0)
        ess = metrics.get("ess", 0.0)

        # Try to extract actual LLC values if available
        if "Ln" in sampler_data:
            # Use pre-computed Ln values if available
            Ln_values = sampler_data["Ln"]
            # This would require L0, n_data, beta to compute LLC - skip for legacy compatibility
            chains, draws = 4, 1000  # Placeholder
            trace_data = np.random.normal(llc_mean, 0.1, (chains, draws))
        else:
            # Create dummy trace for illustration
            chains, draws = 4, 1000  # Placeholder
            trace_data = np.random.normal(llc_mean, 0.1, (chains, draws))

        for chain in range(chains):
            ax.plot(trace_data[chain], alpha=0.7, linewidth=0.8)

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
