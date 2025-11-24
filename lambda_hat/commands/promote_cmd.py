# lambda_hat/commands/promote_cmd.py
"""Promote commands - Stage C: Copy plots to galleries."""

import logging
from pathlib import Path
from typing import List, Optional

from lambda_hat.logging_config import configure_logging
from lambda_hat.promote.core import promote, promote_gallery

log = logging.getLogger(__name__)


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
    configure_logging()
    runs_root = Path(runs_root)
    outdir = Path(outdir)
    promote(runs_root, samplers, outdir, plot_name=plot_name)
    log.info("[promote] single target plots → %s", outdir)


def promote_gallery_entry(
    runs_root: str,
    samplers: List[str],
    outdir: str,
    plot_name: str = "trace.png",
    snippet_out: Optional[str] = None,
) -> Optional[str]:
    """Generate gallery HTML from sampler runs (artifact system).

    Args:
        runs_root: Path to experiments/{exp}/runs/ directory (artifact system)
        samplers: List[str] of sampler names to include
        outdir: Output directory for gallery (required)
        plot_name: Name of plot file to copy (default: trace.png)
        snippet_out: Optional path to write HTML snippet

    Returns:
        Optional[str]: Path to generated HTML snippet if snippet_out provided, else None

    Note: Artifact system uses flat run structure at experiments/{exp}/runs/
          Each run directory: {timestamp}-{sampler}-{tag}-{id}/
    """
    configure_logging()
    runs_root = Path(runs_root)
    outdir = Path(outdir)
    snippet_path = Path(snippet_out) if snippet_out else None

    promote_gallery(runs_root, samplers, outdir, plot_name=plot_name, md_snippet_out=snippet_path)
    log.info("[promote] gallery → %s", outdir)
    if snippet_path:
        log.info("[promote] HTML snippet → %s", snippet_path)
        return str(snippet_path)
    return None
