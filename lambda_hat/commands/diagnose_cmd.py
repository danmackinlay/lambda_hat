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


def _load_traces_raw_json(path: Path) -> Dict[str, np.ndarray]:
    """Load traces from traces_raw.json and convert lists to numpy arrays.

    Args:
        path: Path to traces_raw.json

    Returns:
        Dict mapping trace names to numpy arrays
    """
    traces_raw = json.loads(path.read_text())

    traces = {}
    for key, value in traces_raw.items():
        if isinstance(value, list):
            traces[key] = np.array(value)
        else:
            traces[key] = value

    return traces


def _load_manifest(path: Path) -> Dict:
    """Load manifest.json.

    Args:
        path: Path to manifest.json

    Returns:
        Manifest dict
    """
    return json.loads(path.read_text())


def diagnose_entry(
    run_dir: str,
    mode: str = "light",
    force: bool = False,
) -> Dict:
    """Generate offline diagnostics for a single completed run.

    Uses the single golden path: loads raw traces → calls analyze_traces() → generates outputs.
    Supports smart caching: if trace.nc and analysis.json exist and are fresh, reuses them.

    Args:
        run_dir: Path to run directory containing traces_raw.json
        mode: Diagnostic depth - "light" (basic plots) or "full" (+ expensive plots)
        force: If True, bypass cache and recompute analysis even if fresh

    Returns:
        dict: Diagnostic results with keys:
            - run_dir: Path to run directory
            - diagnostics_dir: Path to diagnostics output
            - plots_generated: List of plot filenames created
            - mode: Diagnostic mode used

    Raises:
        FileNotFoundError: If run_dir doesn't exist or lacks raw traces
        ValueError: If invalid mode specified
    """
    configure_logging()

    if mode not in ("light", "full"):
        raise ValueError(f"Invalid mode '{mode}', must be 'light' or 'full'")

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    log.info("[diagnose] Processing run: %s (mode=%s, force=%s)", run_dir.name, mode, force)

    # Required paths
    traces_raw_path = run_dir / "traces_raw.json"
    manifest_path = run_dir / "manifest.json"
    trace_nc_path = run_dir / "trace.nc"
    analysis_json_path = run_dir / "analysis.json"

    if not traces_raw_path.exists():
        raise FileNotFoundError(f"Missing traces_raw.json in {run_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {run_dir}")

    # Load manifest
    manifest = _load_manifest(manifest_path)

    # Smart caching: use cached analysis if fresh and not forcing
    if (
        not force
        and trace_nc_path.exists()
        and analysis_json_path.exists()
        and trace_nc_path.stat().st_mtime >= traces_raw_path.stat().st_mtime
    ):
        log.info("[diagnose] Using cached trace.nc and analysis.json (fresh)")
        idata = az.from_netcdf(trace_nc_path)
        metrics = json.loads(analysis_json_path.read_text())
    else:
        # Golden path: load raw traces and analyze
        log.info("[diagnose] Loading traces_raw.json and running analysis")
        traces = _load_traces_raw_json(traces_raw_path)

        # Import analyze_traces (lazy to avoid loading in workers)
        from lambda_hat.analysis import analyze_traces

        # analyze_traces writes trace.nc, analysis.json, and plots to outdir
        idata, metrics = analyze_traces(traces, manifest, mode=mode, outdir=run_dir)
        log.info("[diagnose] ✓ Analysis complete, wrote trace.nc and analysis.json")

    # Diagnostics directory (created by analyze_traces if it ran, otherwise may not exist)
    diagnostics_dir = run_dir / "diagnostics"

    # Collect plot filenames
    plots_generated = []
    if diagnostics_dir.exists():
        for plot_file in diagnostics_dir.glob("*.png"):
            plots_generated.append(f"diagnostics/{plot_file.name}")

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
