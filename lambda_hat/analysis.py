# llc/analysis.py
"""Simplified analysis functions for Hydra-based LLC experiments"""

from __future__ import annotations
from typing import Dict, Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
# Removed Path import - visualization functions deleted
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


# analyze_from_Ln_dict function deleted - unused after artifacts.py removal


# create_trace_plots function deleted - unused after artifacts.py removal


# Legacy create_trace_plots function deleted - now using memory-efficient create_trace_plots (renamed from create_trace_plots_from_Ln)


# create_comparison_plot function deleted - unused after artifacts.py removal


# create_summary_table function deleted - unused after artifacts.py removal
