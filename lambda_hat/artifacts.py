# llc/artifacts.py
"""Artifact saving for Hydra-based experiments"""

from typing import Dict, Any
from pathlib import Path
import json
from omegaconf import OmegaConf

from .analysis import create_trace_plots, create_comparison_plot, create_summary_table


def save_run_artifacts(
    results: Dict[str, Any],
    analysis_results: Dict[str, Dict],
    target: Any,
    cfg: Any,
    output_dir: Path
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
    with open(output_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    # Save target info
    target_info = {
        'target_type': cfg.target,
        'dimension': target.d,
        'L0_reference': float(target.L0),
        'n_data': cfg.data.n_data,
        'model_params': target.d if hasattr(target, 'd') else None,
    }

    with open(output_dir / 'target_info.json', 'w') as f:
        json.dump(target_info, f, indent=2)

    # Save timing and config info
    run_info = {
        'samplers_run': list(results.keys()),
        'total_chains': cfg.sampler.chains,
        'seed': cfg.seed,
        'elapsed_times': {
            name: data.get('elapsed_time', 0.0)
            for name, data in results.items()
        }
    }

    with open(output_dir / 'run_info.json', 'w') as f:
        json.dump(run_info, f, indent=2)

    # Create plots
    if cfg.output.save_plots:
        create_trace_plots(results, analysis_results, output_dir)
        create_comparison_plot(analysis_results, output_dir)

    # Save summary table and metrics
    create_summary_table(analysis_results, output_dir)

    print(f"Artifacts saved to: {output_dir}")