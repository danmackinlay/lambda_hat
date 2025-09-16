"""
Manifest utilities for atomic run state management.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any


def write_manifest_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write manifest.json atomically via temporary file and rename."""
    d = path.parent
    with tempfile.NamedTemporaryFile('w', dir=d, delete=False) as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    os.replace(tmp, path)  # atomic on POSIX


def read_manifest(path: Path) -> Dict[str, Any] | None:
    """Read manifest.json, returning None if not found or invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def is_run_completed(run_dir: Path) -> bool:
    """Check if a run is completed based on manifest or heuristics."""
    manifest_path = run_dir / "manifest.json"
    manifest = read_manifest(manifest_path)

    if manifest:
        # Check for explicit completed flag (migration script format)
        if "completed" in manifest:
            return bool(manifest["completed"])

        # Check for pipeline manifest format (has results section)
        if "results" in manifest:
            return True

    # Fallback heuristics: has metrics.json or PNG files
    return bool(
        (run_dir / "metrics.json").exists() or
        any(p.suffix == ".png" for p in run_dir.glob("*.png"))
    )


def get_run_start_time(run_dir: Path) -> float | None:
    """Get run start timestamp, preferring manifest, falling back to filesystem."""
    manifest_path = run_dir / "manifest.json"
    manifest = read_manifest(manifest_path)

    if manifest:
        # Check for timestamp field (pipeline format)
        if "timestamp" in manifest:
            from datetime import datetime
            try:
                return datetime.fromisoformat(manifest["timestamp"]).timestamp()
            except Exception:
                pass

        # Check for started_at field (migration format)
        if "started_at" in manifest and manifest["started_at"]:
            from datetime import datetime
            try:
                return datetime.fromisoformat(manifest["started_at"]).timestamp()
            except Exception:
                pass

    # Fallback: use oldest file mtime
    files = list(run_dir.iterdir())
    if files:
        return min(f.stat().st_mtime for f in files if f.is_file())

    return None


def create_timestamp_symlink(artifacts_dir: Path, timestamp: str, run_id: str) -> None:
    """Create convenience timestamp symlink to canonical run."""
    symlink_path = artifacts_dir / timestamp
    if symlink_path.exists():
        return  # Don't overwrite existing symlinks

    relative_target = Path("../runs") / run_id
    symlink_path.symlink_to(relative_target)