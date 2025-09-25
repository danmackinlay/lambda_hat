# llc/artifacts.py
"""Artifact saving for Hydra-based experiments"""

from typing import Dict, Any
from pathlib import Path
import json
from omegaconf import OmegaConf

from .analysis import (
    create_trace_plots, create_comparison_plot, create_summary_table,
    create_trace_plots_from_Ln, analyze_from_Ln_dict
)


def save_run_artifacts(
    results: Dict[str, Any],
    analysis_results: Dict[str, Dict],
    target: Any,
    cfg: Any,
    output_dir: Path,
) -> None:
    """Save all run artifacts including plots, metrics, and configuration

    Args:
        results: Raw sampling results
        analysis_results: Analysis results with metrics
        target: Target object with model info
        cfg: Configuration object
        output_dir: Directory to save artifacts (managed by Hydra)
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Save target info
    target_info = {
        "target_type": cfg.target,
        "dimension": target.d,
        "L0_reference": float(target.L0),
        "n_data": cfg.data.n_data,
        "model_params": target.d if hasattr(target, "d") else None,
    }

    with open(output_dir / "target_info.json", "w") as f:
        json.dump(target_info, f, indent=2)

    # Save timing and config info
    run_info = {
        "samplers_run": list(results.keys()),
        "total_chains": cfg.sampler.chains,
        "seed": cfg.seed,
        "elapsed_times": {
            name: data.get("elapsed_time", 0.0) for name, data in results.items()
        },
    }

    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    # Create plots
    if cfg.output.save_plots and results:
        create_trace_plots(results, analysis_results, output_dir)
        create_comparison_plot(analysis_results, output_dir)

    # Save summary table and metrics
    create_summary_table(analysis_results, output_dir)

    print(f"Artifacts saved to: {output_dir}")


def save_run_artifacts_from_Ln(
    Ln_histories: Dict[str, Any],
    target: Any,
    cfg: Any,
    output_dir: Path,
    beta: float,
    warmup: int = 0,
) -> None:
    """Save all run artifacts using pre-computed Ln values (MEMORY EFFICIENT).

    Args:
        Ln_histories: Dict[sampler_name, {"Ln": Ln_values, ...}] where Ln_values has shape (chains, draws)
        target: Target object with model info
        cfg: Configuration object
        output_dir: Directory to save artifacts (managed by Hydra)
        beta: Inverse temperature
        warmup: Number of warmup samples to discard
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract Ln values and determine n_data
    Ln_only = {}
    n_data = None

    # Get n_data from target or cfg
    if hasattr(target, 'X_f32') and target.X_f32 is not None:
        n_data = target.X_f32.shape[0]
    elif hasattr(target, 'X_f64') and target.X_f64 is not None:
        n_data = target.X_f64.shape[0]
    elif hasattr(cfg, 'data') and hasattr(cfg.data, 'n_data'):
        n_data = cfg.data.n_data
    else:
        raise ValueError("Cannot determine n_data from target or configuration")

    # Extract just the Ln values
    for sampler_name, sampler_data in Ln_histories.items():
        if isinstance(sampler_data, dict) and "Ln" in sampler_data:
            Ln_only[sampler_name] = sampler_data["Ln"]
        else:
            raise ValueError(f"Expected dict with 'Ln' key for sampler {sampler_name}, got {type(sampler_data)}")

    # Run efficient analysis
    analysis_results = analyze_from_Ln_dict(
        Ln_only, target.L0, n_data, beta, warmup
    )

    # Save configuration
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Save target info
    target_info = {
        "target_type": getattr(cfg, 'target', 'unknown'),
        "dimension": target.d,
        "L0_reference": float(target.L0),
        "n_data": n_data,
        "model_params": target.d if hasattr(target, "d") else None,
    }

    with open(output_dir / "target_info.json", "w") as f:
        json.dump(target_info, f, indent=2)

    # Save timing and config info
    run_info = {
        "samplers_run": list(Ln_histories.keys()),
        "total_chains": cfg.sampler.chains if hasattr(cfg, 'sampler') else 'unknown',
        "seed": getattr(cfg, 'seed', 'unknown'),
        "elapsed_times": {
            name: data.get("elapsed_time", 0.0)
            for name, data in Ln_histories.items()
            if isinstance(data, dict)
        },
    }

    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    # Create plots using efficient method
    if cfg.output.save_plots and Ln_only:
        create_trace_plots_from_Ln(
            Ln_only, analysis_results, output_dir,
            target.L0, n_data, beta
        )
        create_comparison_plot(analysis_results, output_dir)

    # Save summary table and metrics
    create_summary_table(analysis_results, output_dir)

    print(f"Artifacts saved to: {output_dir} (using efficient Ln-based analysis)")
