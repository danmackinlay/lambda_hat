from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

DEFAULT_FILENAME = "llc_convergence_combined.png"


def find_run_dirs(runs_root: Path, target_id: str | None, sampler: str | None) -> List[Path]:
    """Find run directories based on target_id and/or sampler filtering."""
    base = runs_root / "targets"
    if target_id:
        base = base / target_id
        pat = "run_*" if sampler is None else f"run_{sampler}_*"
        return sorted([p for p in base.glob(pat) if p.is_dir()])
    else:
        # Search across all targets
        pat = "*/run_*" if sampler is None else f"*/run_{sampler}_*"
        return sorted([p for p in base.glob(pat) if p.is_dir()])


def find_plot_files(runs_root: Path, sampler: str, plot_name: str) -> List[Path]:
    """Find plot files across all targets for a given sampler."""
    return list(runs_root.glob(f"targets/*/run_{sampler}_*/diagnostics/{plot_name}"))


def _find_plot_files(runs_root: Path, sampler: str, plot_name: str) -> List[Path]:
    """
    Return list of plot files matching:
      runs/targets/*/run_<sampler>_*/diagnostics/<plot_name>
    """
    return list(runs_root.glob(f"targets/*/run_{sampler}_*/diagnostics/{plot_name}"))


def _run_dir_from_plot(plot_file: Path) -> Path:
    """Given .../run_*/diagnostics/<plot>, return the run_* directory."""
    return plot_file.parent.parent


def _analysis_json(run_dir: Path) -> Path:
    return run_dir / "analysis.json"


def _load_metrics(run_dir: Path) -> Dict:
    p = _analysis_json(run_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def gather_latest_runs(runs_root: Path, samplers: List[str], plot_name: str) -> Dict[str, Path]:
    """Pick the most recent run_* dir per sampler, assuming generic plot names."""
    result: Dict[str, Path] = {}

    for sampler in samplers:
        files = _find_plot_files(runs_root, sampler, plot_name)
        if not files:
            raise RuntimeError(f"No runs found for sampler {sampler}")
        # newest by mtime of the plot file
        newest_plot = max(files, key=os.path.getmtime)
        result[sampler] = _run_dir_from_plot(newest_plot)
    return result


def promote(
    runs_root: Path, samplers: List[str], outdir: Path, plot_name: str = "trace.png"
) -> None:
    """
    Copy diagnostics/<plot_name> from newest run_* per sampler to assets/<sampler>.png
    """
    outdir.mkdir(parents=True, exist_ok=True)
    latest_runs = gather_latest_runs(runs_root, samplers, plot_name)

    for sampler, run_dir in latest_runs.items():
        src = run_dir / "diagnostics" / plot_name
        if not src.exists():
            # Try with sampler prefix if exact filename not found
            src_alt = run_dir / "diagnostics" / f"{sampler}_{plot_name}"
            if src_alt.exists():
                src = src_alt
            else:
                raise RuntimeError(f"Expected plot {src} not found (also tried {src_alt})")

        dst = outdir / f"{sampler}.png"
        shutil.copyfile(src, dst)
        log.info("Promoted %s -> %s", src, dst)


def promote_gallery(
    runs_root: Path,
    samplers: List[str],
    outdir: Path,
    plot_name: str = "trace.png",
    md_snippet_out: Path | None = None,
) -> List[Tuple[str, Path, Dict]]:
    """
    1) Promote newest run per sampler -> assets/<sampler>.png
    2) Optionally write a README-ready markdown snippet showing a simple gallery.
    Returns a list of (sampler, asset_path, metrics_dict)
    """
    outdir.mkdir(parents=True, exist_ok=True)
    latest_runs = gather_latest_runs(runs_root, samplers, plot_name)

    rows: List[Tuple[str, Path, Dict]] = []

    for sampler, run_dir in latest_runs.items():
        src = run_dir / "diagnostics" / plot_name
        if not src.exists():
            raise RuntimeError(f"Expected plot {src} not found")
        dst = outdir / f"{sampler}.png"
        shutil.copyfile(src, dst)

        metrics = _load_metrics(run_dir)
        rows.append((sampler, dst, metrics))
        log.info("[gallery] %s: %s -> %s", sampler, src, dst)

    if md_snippet_out is not None:
        # Write a very small, README-friendly HTML snippet for a 3-across gallery
        # with alt text showing sampler name and (if present) llc_mean and r_hat.
        lines = []
        lines.append("<!-- auto-generated: begin gallery -->")
        lines.append('<p align="center">')
        for sampler, asset, metrics in rows:
            alt = f"{sampler.upper()}"
            if "llc_mean" in metrics:
                alt += f" | mean={metrics['llc_mean']:.3f}"
            if "r_hat" in metrics:
                alt += f" | R̂={metrics['r_hat']:.3f}"
            lines.append(f'  <img src="{asset.as_posix()}" alt="{alt}" width="30%"/>')
        lines.append("</p>")
        lines.append("<!-- auto-generated: end gallery -->\n")
        md_snippet_out.write_text("\n".join(lines))
        log.info("[gallery] Wrote README snippet -> %s", md_snippet_out)

    return rows
