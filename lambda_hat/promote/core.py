from __future__ import annotations
from pathlib import Path
import shutil
import os
from typing import Iterable

# Hydra creates timestamped single-run dirs under 'outputs/YYYY-MM-DD/HH-MM-SS'
# and multirun dirs under 'multirun/...'. We search both and pick the newest file.
# See Hydra docs on working directories and multiruns.

DEFAULT_FILENAME = "llc_convergence_combined.png"

def find_latest_artifact(
    filename: str = DEFAULT_FILENAME,
    roots: Iterable[Path] | None = None,
) -> Path:
    roots = list(roots) if roots else [Path("outputs"), Path("multirun")]
    candidates: list[tuple[float, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        # Recursive search (handles outputs/YYYY/... and multirun/.../job_id)
        for p in root.rglob(filename):
            try:
                candidates.append((p.stat().st_mtime, p))
            except FileNotFoundError:
                # In case of a race or deleted file during traversal
                continue
    if not candidates:
        raise RuntimeError(
            f"No '{filename}' found under: {', '.join(str(r) for r in roots)}. "
            "Run `lambda-hat` first to generate artifacts."
        )
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def promote_latest_combined(
    assets_dir: Path,
    filename: str = DEFAULT_FILENAME,
    mode: str = "copy",  # or "link"
    roots: Iterable[Path] | None = None,
) -> Path:
    """Copy/symlink the newest combined plot into the assets dir.

    Returns the destination path.
    """
    src = find_latest_artifact(filename=filename, roots=roots)
    assets_dir.mkdir(parents=True, exist_ok=True)
    dst = assets_dir / filename
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "link":
        try:
            dst.symlink_to(src.resolve())
            print(f"Linked  {src} -> {dst}")
            return dst
        except OSError:
            # Fallback to copy on filesystems without symlink perms
            pass
    shutil.copyfile(src, dst)
    print(f"Copied  {src} -> {dst}")
    return dst
