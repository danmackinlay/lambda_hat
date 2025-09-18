"""Promote README images command implementation."""

from pathlib import Path

from llc.util.promote_utils import latest_run_dir, promote_images


def promote_readme_images_entry(run_dir: str = None, root: str = ".") -> None:
    """Entry point for promote-readme-images command."""
    root = Path(root).resolve()
    assets = root / "assets" / "readme"

    if run_dir:
        run_dir = Path(run_dir).resolve()
    else:
        run_dir = latest_run_dir(root)

    promote_images(run_dir, assets, root_dir=root)