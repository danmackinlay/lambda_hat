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
    ("hmc_running_llc.png", "hmc_llc_running.png"),
    ("mclmc_running_llc.png", "mclmc_llc_running.png"),
    ("hmc_acceptance.png", "hmc_acceptance.png"),
    ("hmc_energy.png", "hmc_energy.png"),
    ("hmc_llc_rank.png", "llc_rank.png"),
    ("hmc_llc_ess_evolution.png", "llc_ess_evolution.png"),
    ("hmc_Ln_centered.png", "Ln_centered.png"),
    ("mclmc_energy_hist.png", "mclmc_energy_hist.png"),
]


def _has_needed_artifacts(p: Path, selection: List[Tuple[str, str]]) -> bool:
    """Valid if it contains metrics.json OR at least one of the expected PNGs."""
    if (p / "metrics.json").exists():
        return True
    pngs = [q.name for q in p.glob("*.png")]
    return any(key in name for key, _ in selection for name in pngs)


def _latest_from_artifacts(artifacts_dir: Path, selection: List[Tuple[str, str]]) -> Path:
    """Fallback: pick newest run from artifacts/ using old logic."""
    candidates = []
    for p in artifacts_dir.iterdir():
        if not p.is_dir():
            continue
        # Follow symlink where possible
        try:
            q = p.resolve()
        except Exception:
            q = p
        if not _has_needed_artifacts(q, selection):
            continue  # skip empty/aborted dirs

        # Prefer parsed timestamp when name is YYYYMMDD-HHMMSS; else use newest file mtime
        ts = None
        if re.fullmatch(r"\d{8}-\d{6}", p.name):
            try:
                ts = datetime.strptime(p.name, "%Y%m%d-%H%M%S").timestamp()
            except ValueError:
                ts = None
        if ts is None:
            mtimes = [f.stat().st_mtime for f in q.glob("*.png")]
            if (q / "metrics.json").exists():
                mtimes.append((q / "metrics.json").stat().st_mtime)
            ts = max(mtimes) if mtimes else q.stat().st_mtime
        candidates.append((ts, p))

    if not candidates:
        raise RuntimeError("No runs with artifacts found under artifacts/")
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def latest_run_dir(root_dir: Path, selection: List[Tuple[str, str]] = None) -> Path:
    """Pick the newest completed run from canonical runs/ directory."""
    if selection is None:
        selection = DEFAULT_SELECTION

    runs_dir = root_dir / "runs"

    # Import here to avoid circular dependencies
    from llc.manifest import is_run_completed, get_run_start_time

    candidates = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Only consider completed runs
        if not is_run_completed(run_dir):
            continue

        # Get start time for sorting
        start_time = get_run_start_time(run_dir)
        if start_time is None:
            start_time = run_dir.stat().st_mtime  # fallback to dir mtime

        candidates.append((start_time, run_dir))

    if not candidates:
        raise RuntimeError("No completed runs found in runs/")

    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]





def promote_images(
    run_dir: Path,
    assets_dir: Path,
    selection: List[Tuple[str, str]] = None,
    root_dir: Path = None
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
        src = (run_dir.resolve() if run_dir.exists() else run_dir) / key
        if not src:
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