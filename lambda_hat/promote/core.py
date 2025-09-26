from __future__ import annotations
import shutil
from pathlib import Path
import os
from typing import Dict


def gather_latest_runs(runs_root: Path, samplers: list[str]) -> Dict[str, Path]:
    """Pick the most recent run dir for each sampler type."""
    result: Dict[str, Path] = {}
    for sampler in samplers:
        candidates = []
        for run_dir in runs_root.glob("*/"):
            analysis_dir = run_dir / "diagnostics"
            if not analysis_dir.exists():
                continue
            # heuristic: check for an image file for this sampler
            if any(
                f.name.startswith(sampler) and f.suffix in (".png", ".jpg")
                for f in analysis_dir.iterdir()
            ):
                candidates.append(run_dir)
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
    latest_runs = gather_latest_runs(runs_root, samplers)

    for sampler, run_dir in latest_runs.items():
        src = run_dir / "diagnostics" / plot_name
        if not src.exists():
            raise RuntimeError(f"Expected plot {src} not found")
        dst = outdir / f"{sampler}.png"
        shutil.copyfile(src, dst)
        print(f"Promoted {src} -> {dst}")
