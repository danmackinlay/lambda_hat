"""Image promotion utilities for README examples."""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


# Which files to copy (source key -> stable asset name).
# We prefer exact filename matches; else substring fallback.
DEFAULT_SELECTION = [
    ("sgld_running_llc.png", "sgld_llc_running.png"),
    ("hmc_running_llc.png",  "hmc_llc_running.png"),
    ("mclmc_running_llc.png","mclmc_llc_running.png"),

    # rank / ess / acf
    ("hmc_llc_rank.png",         "llc_rank.png"),
    ("hmc_llc_ess_evolution.png","llc_ess_evolution.png"),
    ("hmc_llc_autocorr.png",     "llc_autocorr.png"),

    # energy panels we still generate
    ("hmc_energy.png",   "hmc_energy.png"),
    ("mclmc_energy.png", "mclmc_energy.png"),

    # (optional) theta traces for the paper
    ("hmc_theta_trace.png",   "hmc_theta_trace.png"),
    ("sgld_theta_trace.png",  "sgld_theta_trace.png"),
    ("mclmc_theta_trace.png", "mclmc_theta_trace.png"),
]

def latest_run_dir(root_dir: Path, selection: List[Tuple[str, str]] = None) -> Path:
    """Pick the newest completed run from canonical runs/ directory."""
    if selection is None:
        selection = DEFAULT_SELECTION

    runs_dir = root_dir / "runs"

    candidates = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Consider a run completed if it has metrics.json
        if not (run_dir / "metrics.json").exists():
            continue

        # Sort by modification time of metrics.json (indicates completion time)
        start_time = (run_dir / "metrics.json").stat().st_mtime

        candidates.append((start_time, run_dir))

    if not candidates:
        raise RuntimeError(
            "No completed runs found in runs/ (looking for metrics.json)"
        )

    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def promote_images(
    run_dir: Path,
    assets_dir: Path,
    selection: List[Tuple[str, str]] = None,
    root_dir: Path = None,
) -> int:
    """
    Promote diagnostic images from run_dir to assets_dir.

    Returns:
        Number of images copied.
    """
    if selection is None:
        selection = DEFAULT_SELECTION

    assets_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        raise RuntimeError(f"Run dir not found: {run_dir}")

    print(f"Promoting images from: {run_dir}")
    copied = 0

    for key, outname in selection:
        # Look in analysis/ subfolder first (new location), fallback to run root
        analysis_src = run_dir / "analysis" / key
        root_src = run_dir / key

        if analysis_src.exists():
            src = analysis_src
        elif root_src.exists():
            src = root_src
        else:
            print(f"  [skip] no match for '{key}'")
            continue

        dst = assets_dir / outname
        shutil.copy2(src, dst)
        if root_dir:
            print(f"  copied {src.name} -> {dst.relative_to(root_dir)}")
        else:
            print(f"  copied {src.name} -> {dst}")
        copied += 1

    if copied == 0:
        print(
            "No images copied. Did this run save plots? (save_plots=True) "
            "Or are you running an old diagnostics set?"
        )
    else:
        print(f"Done. Copied {copied} images.")
        print(
            "Commit updated assets/readme/*.png and refresh README references if needed."
        )

    return copied
