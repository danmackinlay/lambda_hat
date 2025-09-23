"""Promote README images command implementation."""

from pathlib import Path
from typing import List, Tuple, Union
import logging
import os
import shutil

from llc.util.promote_utils import latest_run_dir, promote_images

logger = logging.getLogger(__name__)


def promote_readme_images_entry(
    runs: Union[str, List[Tuple[str, str]]] = None, root: str = "."
) -> None:
    """
    Entry point for promote-readme-images command.

    Args:
        runs: Either a single run_dir string (legacy) or list of (sampler, run_dir) tuples
        root: Repository root directory
    """
    root = Path(root).resolve()
    assets = root / "assets" / "readme"

    # Handle legacy single run_dir case
    if isinstance(runs, str) or runs is None:
        single_run_dir = runs
        if single_run_dir:
            run_dir = Path(single_run_dir).resolve()
        else:
            run_dir = latest_run_dir(root)
        promote_images(run_dir, assets, root_dir=root)
        return

    # Handle multiple (sampler, run_dir) pairs
    assets.mkdir(parents=True, exist_ok=True)
    expected = {
        "sgld":  ("sgld_running_llc.png",  "sgld_llc_running.png"),
        "sghmc": ("sghmc_running_llc.png", "sghmc_llc_running.png"),
        "hmc":   ("hmc_running_llc.png",   "hmc_llc_running.png"),
        "mclmc": ("mclmc_running_llc.png", "mclmc_llc_running.png"),
    }
    copied = 0
    for sampler, run_dir in runs:
        if sampler not in expected:
            logger.debug("Unknown sampler %r for promotion; skipping", sampler)
            continue
        src_name, dst_name = expected[sampler]

        # Look in analysis/ subfolder first (new location), fallback to run root
        analysis_src = Path(run_dir) / "analysis" / src_name
        root_src = Path(run_dir) / src_name

        if analysis_src.exists():
            src_path = analysis_src
        elif root_src.exists():
            src_path = root_src
        else:
            logger.warning("expected image not found for %s: %s (checked analysis/ and root)", sampler, src_name)
            continue

        dst_path = assets / dst_name
        shutil.copy2(src_path, dst_path)
        logger.info("copied %s -> %s", src_path.relative_to(root), dst_path.relative_to(root))
        copied += 1

    logger.info("Promotion finished. Copied %d images.", copied)
