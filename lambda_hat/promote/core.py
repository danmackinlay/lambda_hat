from __future__ import annotations
import shutil
from pathlib import Path
import os
from typing import Dict, Iterable
import time
import logging

log = logging.getLogger("lambda_hat.promote")


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    """Iterate over run directories, supporting both legacy runs/ and current outputs/ structure."""
    if not root.exists():
        return

    # Support outputs/YYYY-MM-DD/HH-MM-SS/ structure
    if root.name == "outputs" or "outputs" in str(root):
        # Check if we have date directories
        try:
            for day_entry in os.scandir(root):
                if not day_entry.is_dir(follow_symlinks=False):
                    continue
                # Check if this looks like a date directory (YYYY-MM-DD pattern)
                if len(day_entry.name) == 10 and day_entry.name[4] == '-' and day_entry.name[7] == '-':
                    # It's a date directory, scan its children
                    for run_entry in os.scandir(day_entry.path):
                        if run_entry.is_dir(follow_symlinks=False):
                            yield Path(run_entry.path)
                else:
                    # Not a date directory, might be a direct run directory
                    yield Path(day_entry.path)
        except OSError:
            # Directory became unreadable
            return
    else:
        # Legacy runs/ structure or other flat structure
        try:
            for entry in os.scandir(root):
                if entry.is_dir(follow_symlinks=False):
                    yield Path(entry.path)
        except OSError:
            return


def gather_latest_runs(runs_root: Path, samplers: list[str],
                      max_dirs: int = 5000, verbose: bool = False) -> Dict[str, Path]:
    """Pick the most recent run dir for each sampler type."""
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    result: Dict[str, Path] = {}
    total_scanned = 0

    for sampler in samplers:
        candidates: list[Path] = []
        start_time = time.time()
        dirs_checked = 0

        for run_dir in _iter_run_dirs(runs_root):
            dirs_checked += 1
            total_scanned += 1

            if total_scanned > max_dirs:
                if verbose:
                    log.warning(f"Reached max_dirs limit ({max_dirs}), stopping scan")
                break

            diagnostics_dir = run_dir / "diagnostics"
            if not diagnostics_dir.exists():
                continue

            try:
                # Use os.scandir for faster directory iteration
                found_sampler_plot = False
                for entry in os.scandir(diagnostics_dir):
                    if (entry.is_file() and
                        entry.name.startswith(sampler) and
                        entry.name.lower().endswith(('.png', '.jpg'))):
                        found_sampler_plot = True
                        break

                if found_sampler_plot:
                    candidates.append(run_dir)
            except OSError:
                # Unreadable directory, skip
                continue

        elapsed = time.time() - start_time
        if verbose:
            log.info(f"[{sampler}] Scanned {dirs_checked} dirs in {elapsed:.2f}s, found {len(candidates)} candidates")

        if not candidates:
            raise RuntimeError(f"No runs found for sampler {sampler} in {runs_root}")

        # Pick newest by modification time
        newest = max(candidates, key=lambda p: os.path.getmtime(p))
        result[sampler] = newest

        if verbose:
            log.info(f"[{sampler}] Selected: {newest}")

    return result


def promote(
    runs_root: Path, samplers: list[str], outdir: Path,
    plot_name: str = "trace.png", max_dirs: int = 5000, verbose: bool = False
) -> None:
    """
    For each sampler, copy the chosen analysis plot to assets/<sampler>.png.

    Args:
        runs_root: directory containing outputs/YYYY-MM-DD/HH-MM-SS/ or legacy runs/<id>/
        samplers: list of sampler names to promote
        outdir: destination assets directory
        plot_name: which plot to copy from diagnostics/ (default: 'trace.png')
        max_dirs: maximum number of directories to scan
        verbose: enable verbose logging
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    outdir.mkdir(parents=True, exist_ok=True)
    latest_runs = gather_latest_runs(runs_root, samplers, max_dirs=max_dirs, verbose=verbose)

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
        print(f"Promoted {src} -> {dst}")
