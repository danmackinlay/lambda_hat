from __future__ import annotations
import shutil
from pathlib import Path
import os
from typing import Dict


def gather_latest_runs(
    runs_root: Path, samplers: list[str], plot_name: str
) -> Dict[str, Path]:
    """Pick the most recent run dir for each sampler type."""
    result: Dict[str, Path] = {}
    for sampler in samplers:
        # Look for per-sampler file e.g. "{sampler}_trace.png"
        per_sampler = f"{sampler}_{plot_name}"
        pattern = (
            runs_root
            / "samples"
            / "*"
            / sampler
            / "run_*"
            / "diagnostics"
            / per_sampler
        )
        files = list(pattern.parent.glob(f"**/{per_sampler}"))
        # Promote the parent of "diagnostics" → the run_* directory
        candidates = [f.parent.parent for f in files]
        if not candidates:
            raise RuntimeError(f"No runs found for sampler {sampler}")
        # pick newest by mtime
        newest = max(candidates, key=os.path.getmtime)
        result[sampler] = newest
    return result


def promote(
    runs_root: Path, samplers: list[str], outdir: Path, plot_name: str = "trace.png"
) -> None:
    """
    For each sampler, copy the chosen analysis plot to assets/<sampler>.png.

    Args:
        runs_root: directory containing runs/<id>/
        samplers: list of sampler names to promote
        outdir: destination assets directory
        plot_name: which plot to copy from analysis/ (default: 'trace.png')
    """
    outdir.mkdir(parents=True, exist_ok=True)
    latest_runs = gather_latest_runs(runs_root, samplers, plot_name)

    for sampler, run_dir in latest_runs.items():
        src = run_dir / "diagnostics" / f"{sampler}_{plot_name}"
        if not src.exists():
            raise RuntimeError(f"Expected plot {src} not found")
        dst = outdir / f"{sampler}.png"
        shutil.copyfile(src, dst)
        print(f"Promoted {src} -> {dst}")
