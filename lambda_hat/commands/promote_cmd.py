# lambda_hat/commands/promote_cmd.py
"""Promote commands - Stage C: Copy plots to galleries."""

from pathlib import Path
from typing import List, Optional

from lambda_hat.promote.core import promote, promote_gallery


def promote_single_entry(
    runs_root: str, samplers: List[str], outdir: str, plot_name: str = "trace.png"
) -> None:
    """Promote plots for a single target.

    Args:
        runs_root: Root directory containing run subdirectories
        samplers: List of sampler names to include
        outdir: Output directory for promoted plots
        plot_name: Name of plot file to copy (default: trace.png)
    """
    runs_root = Path(runs_root)
    outdir = Path(outdir)
    promote(runs_root, samplers, outdir, plot_name=plot_name)
    print(f"[promote] single target plots → {outdir}")


def promote_gallery_entry(
    runs_root: str,
    samplers: List[str],
    outdir: str = "runs/promotion",
    plot_name: str = "trace.png",
    snippet_out: Optional[str] = None,
) -> None:
    """Generate gallery HTML of all targets.

    Args:
        runs_root: Root directory containing run subdirectories
        samplers: List of sampler names to include
        outdir: Output directory for gallery (default: runs/promotion)
        plot_name: Name of plot file to copy (default: trace.png)
        snippet_out: Optional path to write HTML snippet
    """
    runs_root = Path(runs_root)
    outdir = Path(outdir)
    snippet_path = Path(snippet_out) if snippet_out else None

    promote_gallery(runs_root, samplers, outdir, plot_name=plot_name, snippet_out=snippet_path)
    print(f"[promote] gallery → {outdir}")
    if snippet_path:
        print(f"[promote] HTML snippet → {snippet_path}")
