# lambda_hat/commands/diagnose_cmd.py
"""Diagnose command: Generate offline diagnostics for sampling runs (Stage C).

This command generates plots and analysis from completed sampling runs.
It decouples expensive diagnostics (matplotlib/arviz) from the sampling stage,
allowing sampling to run fast with analysis_mode="none".

Usage:
    # Single run
    lambda-hat diagnose --run-dir artifacts/experiments/smoke/runs/20251120...

    # All runs in experiment
    lambda-hat diagnose-experiment --experiment smoke --mode light
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import arviz as az
import numpy as np

from lambda_hat.artifacts import Paths
from lambda_hat.logging_config import configure_logging

log = logging.getLogger(__name__)


def diagnose_entry(
    run_dir: str,
    mode: str = "light",
) -> Dict:
    """Generate offline diagnostics for a single completed run.

    Args:
        run_dir: Path to run directory containing trace.nc or traces_raw.json
        mode: Diagnostic depth - "light" (basic plots) or "full" (+ expensive plots)

    Returns:
        dict: Diagnostic results with keys:
            - run_dir: Path to run directory
            - diagnostics_dir: Path to diagnostics output
            - plots_generated: List of plot filenames created
            - mode: Diagnostic mode used

    Raises:
        FileNotFoundError: If run_dir doesn't exist or lacks trace data
        ValueError: If invalid mode specified
    """
    configure_logging()

    if mode not in ("light", "full"):
        raise ValueError(f"Invalid mode '{mode}', must be 'light' or 'full'")

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    log.info("[diagnose] Processing run: %s (mode=%s)", run_dir.name, mode)

    # Load manifest to get sampler name and metadata
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {run_dir}")

    manifest = json.loads(manifest_path.read_text())
    sampler_name = manifest.get("sampler", "unknown")

    # Load InferenceData (or create from raw traces)
    trace_nc = run_dir / "trace.nc"
    traces_raw_json = run_dir / "traces_raw.json"

    if trace_nc.exists():
        log.info("[diagnose] Loading trace.nc")
        idata = az.from_netcdf(trace_nc)
    elif traces_raw_json.exists():
        log.info("[diagnose] Reconstructing InferenceData from traces_raw.json")
        idata = _reconstruct_idata_from_json(traces_raw_json, manifest)
        # Save for future use
        idata.to_netcdf(trace_nc)
        log.info("[diagnose] Wrote trace.nc")
    else:
        raise FileNotFoundError(
            f"No trace data found in {run_dir} (expected trace.nc or traces_raw.json)"
        )

    # Create diagnostics directory
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)

    # Import analysis functions (lazy to avoid loading in workers)
    from lambda_hat.analysis import (
        create_arviz_diagnostics,
        create_combined_convergence_plot,
        create_work_normalized_variance_plot,
    )

    plots_generated = []

    # Basic diagnostics (always in light/full mode)
    log.info("[diagnose] Generating ArviZ diagnostics")
    create_arviz_diagnostics({sampler_name: idata}, run_dir)
    plots_generated.extend(
        ["diagnostics/trace.png", "diagnostics/rank.png", "diagnostics/energy.png"]
    )

    log.info("[diagnose] Generating convergence plot")
    create_combined_convergence_plot({sampler_name: idata}, run_dir)
    plots_generated.append("diagnostics/llc_convergence_combined.png")

    # Full diagnostics (expensive)
    if mode == "full":
        log.info("[diagnose] Generating work-normalized variance plot")
        create_work_normalized_variance_plot({sampler_name: idata}, run_dir)
        plots_generated.append("diagnostics/wnv.png")

    log.info("[diagnose] ✓ Generated %d plots in %s", len(plots_generated), diagnostics_dir)

    return {
        "run_dir": str(run_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "plots_generated": plots_generated,
        "mode": mode,
    }


def diagnose_experiment_entry(
    experiment: str,
    mode: str = "light",
    samplers: Optional[List[str]] = None,
) -> Dict:
    """Generate diagnostics for all runs in an experiment.

    Args:
        experiment: Experiment name (e.g., "smoke", "dev")
        mode: Diagnostic depth - "light" or "full"
        samplers: Optional list of sampler names to process (default: all)

    Returns:
        dict: Summary with keys:
            - experiment: Experiment name
            - num_runs: Number of runs processed
            - num_success: Number of successful diagnostics
            - num_failed: Number of failures
            - failed_runs: List of failed run directories
            - mode: Diagnostic mode used
    """
    configure_logging()

    if mode not in ("light", "full"):
        raise ValueError(f"Invalid mode '{mode}', must be 'light' or 'full'")

    # Find all runs in experiment
    paths = Paths.from_env()
    experiment_runs_dir = paths.experiments / experiment / "runs"

    if not experiment_runs_dir.exists():
        raise FileNotFoundError(f"Experiment runs directory not found: {experiment_runs_dir}")

    # Find all run directories
    run_dirs = sorted([d for d in experiment_runs_dir.iterdir() if d.is_dir()])

    if samplers:
        # Filter by sampler name (run dirs contain sampler in name)
        run_dirs = [d for d in run_dirs if any(s in d.name for s in samplers)]

    log.info(
        "[diagnose-experiment] Processing %d runs in experiment '%s' (mode=%s)",
        len(run_dirs),
        experiment,
        mode,
    )

    num_success = 0
    num_failed = 0
    failed_runs = []

    for i, run_dir in enumerate(run_dirs, 1):
        try:
            log.info("[diagnose-experiment] [%d/%d] %s", i, len(run_dirs), run_dir.name)
            diagnose_entry(str(run_dir), mode=mode)
            num_success += 1
        except Exception as e:
            log.warning(
                "[diagnose-experiment] [%d/%d] FAILED: %s - %s", i, len(run_dirs), run_dir.name, e
            )
            num_failed += 1
            failed_runs.append(str(run_dir))

    log.info(
        "[diagnose-experiment] ✓ Completed: %d success, %d failed out of %d runs",
        num_success,
        num_failed,
        len(run_dirs),
    )

    return {
        "experiment": experiment,
        "num_runs": len(run_dirs),
        "num_success": num_success,
        "num_failed": num_failed,
        "failed_runs": failed_runs,
        "mode": mode,
    }


def _reconstruct_idata_from_json(traces_json_path: Path, manifest: Dict) -> az.InferenceData:
    """Reconstruct ArviZ InferenceData from raw traces JSON.

    This is used when a run was created with analysis_mode="none" and only
    has traces_raw.json. We reconstruct the InferenceData so diagnostics
    can be generated later.

    Args:
        traces_json_path: Path to traces_raw.json
        manifest: Run manifest with metadata

    Returns:
        ArviZ InferenceData object
    """
    traces_raw = json.loads(traces_json_path.read_text())

    # Convert lists back to numpy arrays
    traces = {}
    for key, value in traces_raw.items():
        if isinstance(value, list):
            traces[key] = np.array(value)
        else:
            traces[key] = value

    # Extract metadata from manifest
    sampler_name = manifest.get("sampler", "unknown")
    warmup_steps = manifest.get("warmup_steps", 0)

    # Determine chain dimensions
    # traces["samples"] has shape (n_samples, n_params)
    # ArviZ expects (chain, draw, *shape)
    samples = traces.get("samples")
    if samples is None:
        raise ValueError("traces_raw.json missing 'samples' key")

    n_samples, n_params = samples.shape

    # Reshape to single chain
    samples_reshaped = samples[np.newaxis, :, :]  # (1, n_samples, n_params)

    # Create posterior group
    posterior_dict = {"theta": samples_reshaped}

    # Add scalar traces if they exist
    for key in ["llc", "grad_norm", "accept_prob"]:
        if key in traces:
            value = traces[key]
            if hasattr(value, "shape"):
                # Reshape to (chain, draw) if needed
                if value.ndim == 1:
                    posterior_dict[key] = value[np.newaxis, :]
                else:
                    posterior_dict[key] = value

    # Create InferenceData
    idata = az.from_dict(
        posterior=posterior_dict,
        attrs={
            "sampler": sampler_name,
            "warmup_steps": warmup_steps,
        },
    )

    return idata
