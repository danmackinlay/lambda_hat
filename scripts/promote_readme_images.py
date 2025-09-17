#!/usr/bin/env python3
"""
Promote diagnostic plots from artifacts/ to assets/readme/ for README examples.

Usage:
    uv run python scripts/promote_readme_images.py                    # promote from latest run
    uv run python scripts/promote_readme_images.py artifacts/run-dir  # promote from specific run
"""

from pathlib import Path
import shutil
import sys
import re
from datetime import datetime

# Add parent directory to path for llc module imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from llc.manifest import is_run_completed, get_run_start_time

# Which files to copy (source key -> stable asset name).
# We prefer exact filename matches; else substring fallback.
SELECT = [
    # Running LLC (one per sampler)
    ("sgld_running_llc.png", "sgld_llc_running.png"),
    ("hmc_running_llc.png", "hmc_llc_running.png"),
    ("mclmc_running_llc.png", "mclmc_llc_running.png"),
    # HMC diagnostics
    ("hmc_acceptance.png", "hmc_acceptance.png"),
    ("hmc_energy.png", "hmc_energy.png"),
    # ArviZ-first LLC diagnostics
    ("hmc_llc_rank.png", "llc_rank.png"),
    ("hmc_llc_ess_evolution.png", "llc_ess_evolution.png"),
    # Optionally include one centered L_n for pedagogy
    ("hmc_Ln_centered.png", "Ln_centered.png"),
    # If you still want an energy histogram for MCLMC, keep it:
    ("mclmc_energy_hist.png", "mclmc_energy_hist.png"),
]


def _has_needed_artifacts(p: Path) -> bool:
    """Valid if it contains metrics.json OR at least one of the expected PNGs."""
    if (p / "metrics.json").exists():
        return True
    pngs = [q.name for q in p.glob("*.png")]
    return any(key in name for key, _ in SELECT for name in pngs)


def latest_run_dir(root_dir: Path) -> Path:
    """Pick the newest completed run from canonical runs/ directory."""
    runs_dir = root_dir / "runs"

    if not runs_dir.exists():
        # Fallback to old artifacts/ scanning for backward compatibility
        artifacts_dir = root_dir / "artifacts"
        if not artifacts_dir.exists():
            raise SystemExit("Neither runs/ nor artifacts/ directory found")
        return _latest_from_artifacts(artifacts_dir)

    candidates: list[tuple[float, Path]] = []
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
        raise SystemExit("No completed runs found in runs/")

    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def _latest_from_artifacts(artifacts_dir: Path) -> Path:
    """Fallback: pick newest run from artifacts/ using old logic."""
    candidates: list[tuple[float, Path]] = []
    for p in artifacts_dir.iterdir():
        if not p.is_dir():
            continue
        # Follow symlink where possible
        try:
            q = p.resolve()
        except Exception:
            q = p
        if not _has_needed_artifacts(q):
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
        raise SystemExit("No runs with artifacts found under artifacts/")
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def find_first_match(run_dir: Path, key: str) -> Path | None:
    """
    Prefer exact filename match; fall back to substring.
    """
    base = run_dir.resolve() if run_dir.exists() else run_dir
    exact = base / key
    if exact.exists():
        return exact
    # substring fallback for backward compatibility
    for p in sorted(base.glob("*.png")):
        if key in p.name:
            return p
    return None


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets" / "readme"
    assets.mkdir(parents=True, exist_ok=True)

    # Allow overriding the source run dir on the CLI
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1]).resolve()
    else:
        run_dir = latest_run_dir(root)

    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    print(f"Promoting images from: {run_dir}")
    copied = 0

    for key, outname in SELECT:
        src = find_first_match(run_dir, key)
        if not src:
            print(f"  [skip] no match for '{key}'")
            continue

        dst = assets / outname
        shutil.copy2(src, dst)
        print(f"  copied {src.name} -> {dst.relative_to(root)}")
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


if __name__ == "__main__":
    main()
