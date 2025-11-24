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
    """Find sampler run directories in artifact system.

    Args:
        runs_root: Path to experiments/{exp}/runs/ directory (artifact system)
        target_id: Optional target filter (short ID, e.g., tgt_abc123 or tgt_abc123[:8])
        sampler: Optional sampler filter (e.g., "hmc", "vi")

    Returns:
        List of Path objects matching filters

    Note: Artifact system uses flat structure: {timestamp}-{sampler}-{tag}-{id}/
          Tag typically contains target short ID (first 8 chars)
    """
    runs_root = Path(runs_root)

    # Artifact system uses flat structure
    all_runs = sorted(runs_root.glob("*"))

    # Filter by manifest presence (sampler runs only, excludes orchestration/build runs)
    sampler_runs = [d for d in all_runs if d.is_dir() and (d / "manifest.json").exists()]

    # Apply optional sampler filter
    if sampler:
        sampler_runs = [d for d in sampler_runs if f"-{sampler}-" in d.name]

    # Apply optional target_id filter
    if target_id:
        # Target ID (short form) appears in tag portion of run directory name
        target_short = target_id[:8] if target_id.startswith("tgt_") else target_id
        sampler_runs = [d for d in sampler_runs if target_short in d.name]

    return sampler_runs


def find_plot_files(runs_root: Path, sampler: str, plot_name: str) -> List[Path]:
    """Find plot files across all runs for a given sampler (artifact system)."""
    runs_root = Path(runs_root)
    pat = f"*-{sampler}-*/diagnostics/{plot_name}"
    return list(runs_root.glob(pat))


def _find_plot_files(runs_root: Path, sampler: str, plot_name: str) -> List[Path]:
    """Find plot files in artifact system's flat structure.

    Args:
        runs_root: Path to experiments/{exp}/runs/ directory
        sampler: Sampler name (e.g., "hmc", "vi")
        plot_name: Plot filename (e.g., "trace.png")

    Returns:
        List of plot file paths matching:
          {runs_root}/{timestamp}-{sampler}-{tag}-{id}/diagnostics/{plot_name}
    """
    runs_root = Path(runs_root)
    pat = f"*-{sampler}-*/diagnostics/{plot_name}"
    return list(runs_root.glob(pat))


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
    """Pick the most recent run_* dir per sampler, assuming generic plot names.

    Gracefully skips samplers with no runs containing the requested plot.
    """
    result: Dict[str, Path] = {}

    for sampler in samplers:
        files = _find_plot_files(runs_root, sampler, plot_name)
        if not files:
            log.warning(
                "No runs with plot '%s' found for sampler '%s' - skipping from promotion",
                plot_name,
                sampler,
            )
            continue
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


def promote_target_diagnostics(
    runs_root: Path,
    outdir: Path,
) -> List[Tuple[str, List[Path]]]:
    """DEPRECATED: Promote target diagnostics to promotion gallery.

    This function is obsolete with the artifact system. Target diagnostics
    already exist at the correct location and don't need promotion:
    artifacts/experiments/{exp}/targets/{id}/diagnostics/

    Args:
        runs_root: Root directory containing targets/ subdirectory
            (e.g., artifacts/experiments/dev/runs)
        outdir: Output directory for promoted plots (creates targets/ subdirectory)

    Returns:
        List of (target_id, list_of_promoted_plot_paths) tuples (always empty)
    """
    # Function deprecated - target diagnostics no longer need promotion
    log.warning(
        "promote_target_diagnostics() is deprecated and no longer needed with the artifact system"
    )
    return []

    # Legacy code below (unreachable)
    targets_dir = runs_root / "targets"
    if not targets_dir.exists():
        log.warning("No targets directory found at %s - skipping target promotion", targets_dir)
        return []

    promoted: List[Tuple[str, List[Path]]] = []

    for target_dir in sorted(targets_dir.iterdir()):
        if not target_dir.is_dir() or target_dir.name.startswith("_"):
            continue

        target_id = target_dir.name
        diagnostics_dir = target_dir / "diagnostics"

        if not diagnostics_dir.exists():
            log.debug("No diagnostics directory for target %s - skipping", target_id)
            continue

        # Find all PNG files in the diagnostics directory
        plot_files = list(diagnostics_dir.glob("*.png"))
        if not plot_files:
            log.debug("No diagnostic plots for target %s - skipping", target_id)
            continue

        # Create output directory for this target
        target_outdir = outdir / "targets" / target_id
        target_outdir.mkdir(parents=True, exist_ok=True)

        promoted_plots: List[Path] = []
        for src in plot_files:
            dst = target_outdir / src.name
            shutil.copyfile(src, dst)
            promoted_plots.append(dst)
            log.info("[target] %s: %s -> %s", target_id, src, dst)

        promoted.append((target_id, promoted_plots))

    return promoted


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

    if not latest_runs:
        log.warning(
            "No runs with plot '%s' found for any sampler - promotion skipped",
            plot_name,
        )
        return []

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
