# lambda_hat/commands/diagnose_target_cmd.py
"""Diagnose Target command: Generate target diagnostics (teacher comparison plots).

This command regenerates teacher diagnostic plots for a built target.
Useful for on-demand regeneration when diagnostics were skipped during build
(e.g., LAMBDA_HAT_SKIP_DIAGNOSTICS=1 in Parsl workflows).

Usage:
    # Single target
    lambda-hat diagnose-target --target-id tgt_abc123 --experiment dev

    # All targets in experiment
    lambda-hat diagnose-targets --experiment dev
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from lambda_hat.artifacts import Paths
from lambda_hat.logging_config import configure_logging

log = logging.getLogger(__name__)


def diagnose_target_entry(
    target_id: str,
    experiment: str,
) -> Dict:
    """Generate target diagnostics for a built target.

    Args:
        target_id: Target ID (e.g., "tgt_abc123")
        experiment: Experiment name (e.g., "dev")

    Returns:
        dict: Diagnostic results with keys:
            - target_id: Target ID
            - target_dir: Path to target directory
            - diagnostics_dir: Path to diagnostics output
            - plots_generated: List of plot filenames created

    Raises:
        FileNotFoundError: If target directory or required files don't exist
        ValueError: If target lacks teacher (teacher: _null)
    """
    configure_logging()

    # Find target directory
    paths = Paths.from_env()

    # Try legacy path first (runs/targets/), then new path (experiments/{exp}/targets/)
    legacy_target_dir = Path("runs") / "targets" / target_id
    new_target_dir = paths.experiments / experiment / "targets" / target_id

    if new_target_dir.exists():
        target_dir = new_target_dir
    elif legacy_target_dir.exists():
        target_dir = legacy_target_dir
    else:
        raise FileNotFoundError(
            f"Target directory not found: {new_target_dir} (or legacy {legacy_target_dir})"
        )

    log.info("[diagnose-target] Processing target: %s", target_id)

    # Load target metadata
    meta_path = target_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {target_dir}")

    meta = json.loads(meta_path.read_text())

    # Check if target has a teacher
    teacher = meta.get("target", {}).get("teacher")
    if teacher is None or teacher == "_null":
        raise ValueError(
            f"Target {target_id} has no teacher (teacher: _null) - cannot generate teacher diagnostics"
        )

    # Load target parameters and data
    params_path = target_dir / "params.npz"
    data_path = target_dir / "data.npz"

    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.npz in {target_dir}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.npz in {target_dir}")

    log.info("[diagnose-target] Loading target artifacts")
    params_data = np.load(params_path)
    data_npz = np.load(data_path)

    # Create diagnostics directory
    diagnostics_dir = target_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)

    # Import analysis function (lazy to avoid loading matplotlib in workers)
    from lambda_hat.analysis import compute_target_diagnostics

    log.info("[diagnose-target] Generating teacher comparison plots")

    # Reconstruct necessary context for compute_target_diagnostics
    # The function expects: params, data, meta, target_dir
    compute_target_diagnostics(
        params=dict(params_data),
        data=data_npz,
        meta=meta,
        target_dir=target_dir,
    )

    # Expected plots
    expected_plots = [
        "target_train_test_loss.png",
        "target_pred_vs_teacher_train.png",
        "target_pred_vs_teacher_test.png",
    ]

    plots_generated = []
    for plot_name in expected_plots:
        plot_path = diagnostics_dir / plot_name
        if plot_path.exists():
            plots_generated.append(plot_name)

    log.info("[diagnose-target] ✓ Generated %d plots in %s", len(plots_generated), diagnostics_dir)

    return {
        "target_id": target_id,
        "target_dir": str(target_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "plots_generated": plots_generated,
    }


def diagnose_targets_entry(
    experiment: str,
) -> Dict:
    """Generate target diagnostics for all targets in an experiment.

    Args:
        experiment: Experiment name (e.g., "dev", "smoke")

    Returns:
        dict: Summary with keys:
            - experiment: Experiment name
            - num_targets: Number of targets processed
            - num_success: Number of successful diagnostics
            - num_failed: Number of failures
            - failed_targets: List of failed target IDs
    """
    configure_logging()

    # Find all targets in experiment
    paths = Paths.from_env()

    # Try new path first (experiments/{exp}/targets/), then legacy (runs/targets/)
    new_targets_dir = paths.experiments / experiment / "targets"
    legacy_targets_dir = Path("runs") / "targets"

    if new_targets_dir.exists():
        targets_dir = new_targets_dir
    elif legacy_targets_dir.exists():
        targets_dir = legacy_targets_dir
    else:
        raise FileNotFoundError(
            f"Targets directory not found: {new_targets_dir} (or legacy {legacy_targets_dir})"
        )

    # Find all target directories (skip catalog and other files)
    target_dirs = sorted(
        [d for d in targets_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    )
    target_ids = [d.name for d in target_dirs]

    log.info(
        "[diagnose-targets] Processing %d targets in experiment '%s'",
        len(target_ids),
        experiment,
    )

    num_success = 0
    num_failed = 0
    failed_targets = []

    for i, target_id in enumerate(target_ids, 1):
        try:
            log.info("[diagnose-targets] [%d/%d] %s", i, len(target_ids), target_id)
            diagnose_target_entry(target_id, experiment)
            num_success += 1
        except Exception as e:
            log.warning(
                "[diagnose-targets] [%d/%d] FAILED: %s - %s", i, len(target_ids), target_id, e
            )
            num_failed += 1
            failed_targets.append(target_id)

    log.info(
        "[diagnose-targets] ✓ Completed: %d success, %d failed out of %d targets",
        num_success,
        num_failed,
        len(target_ids),
    )

    return {
        "experiment": experiment,
        "num_targets": len(target_ids),
        "num_success": num_success,
        "num_failed": num_failed,
        "failed_targets": failed_targets,
    }
