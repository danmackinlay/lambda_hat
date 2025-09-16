#!/usr/bin/env python3
"""
Migrate existing artifacts/ structure to canonical runs/ + convenience symlinks.

This script:
1. Moves hash-named dirs from artifacts/ to runs/
2. Creates manifests for existing runs based on observed files
3. Creates timestamp symlinks in artifacts/ for completed runs
4. Cleans up empty/dangling timestamp dirs

Usage:
    uv run python scripts/migrate_to_runs.py [--dry-run]
"""

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import argparse


def write_manifest_atomic(path: Path, data: dict):
    """Write manifest.json atomically."""
    d = path.parent
    with tempfile.NamedTemporaryFile('w', dir=d, delete=False) as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    os.replace(tmp, path)


def is_hash_like(name: str) -> bool:
    """Check if directory name looks like a hash/run_id."""
    return bool(re.fullmatch(r"[0-9a-f]{8,}", name.lower()))


def is_timestamp_like(name: str) -> bool:
    """Check if directory name looks like a timestamp."""
    return bool(re.fullmatch(r"\d{8}-\d{6}", name))


def create_manifest_from_dir(run_dir: Path) -> dict:
    """Create manifest data from existing run directory contents."""
    artifacts = []

    # Collect all significant files
    for pattern in ["*.png", "*.nc", "*.json", "*.html", "*.txt"]:
        artifacts.extend([p.name for p in run_dir.glob(pattern)])

    # Try to determine start time from files
    started_at = None
    if artifacts:
        oldest_mtime = min(run_dir.glob("*"), key=lambda p: p.stat().st_mtime).stat().st_mtime
        started_at = datetime.fromtimestamp(oldest_mtime).isoformat()

    # Consider complete if has metrics.json or PNG files
    completed = bool((run_dir / "metrics.json").exists() or any(p.suffix == ".png" for p in run_dir.iterdir()))

    return {
        "run_id": run_dir.name,
        "started_at": started_at,
        "completed": completed,
        "artifacts": sorted(artifacts)
    }


def migrate_artifacts_to_runs(dry_run: bool = False):
    """Migrate artifacts/ to runs/ structure."""
    root = Path.cwd()
    artifacts_dir = root / "artifacts"
    runs_dir = root / "runs"

    if not artifacts_dir.exists():
        print("No artifacts/ directory found")
        return

    runs_dir.mkdir(exist_ok=True)

    print(f"{'[DRY RUN] ' if dry_run else ''}Migrating artifacts to canonical runs/ structure...")

    # Step 1: Move hash-named directories to runs/
    hash_dirs = [p for p in artifacts_dir.iterdir() if p.is_dir() and is_hash_like(p.name)]
    for hash_dir in hash_dirs:
        target = runs_dir / hash_dir.name
        if target.exists():
            print(f"  Skipping {hash_dir.name} (already in runs/)")
            continue

        print(f"  Moving {hash_dir.name} -> runs/{hash_dir.name}")
        if not dry_run:
            shutil.move(str(hash_dir), str(target))

            # Create manifest if it doesn't exist
            manifest_path = target / "manifest.json"
            if not manifest_path.exists():
                manifest_data = create_manifest_from_dir(target)
                write_manifest_atomic(manifest_path, manifest_data)
                print(f"    Created manifest for {hash_dir.name}")

    # Step 2: Handle timestamp directories
    timestamp_dirs = [p for p in artifacts_dir.iterdir() if p.is_dir() and is_timestamp_like(p.name)]
    for ts_dir in timestamp_dirs:
        if ts_dir.is_symlink():
            # Check if symlink is valid
            try:
                target = ts_dir.resolve()
                if not target.exists():
                    print(f"  Removing dangling symlink: {ts_dir.name}")
                    if not dry_run:
                        ts_dir.unlink()
                else:
                    print(f"  Keeping valid symlink: {ts_dir.name} -> {target}")
            except Exception:
                print(f"  Removing broken symlink: {ts_dir.name}")
                if not dry_run:
                    ts_dir.unlink()
        else:
            # Real timestamp directory - need to move to runs/ and create symlink
            # Generate a hash-like ID for it
            import hashlib
            run_id = hashlib.sha1(ts_dir.name.encode()).hexdigest()[:12]
            target = runs_dir / run_id

            if target.exists():
                print(f"  Collision for {ts_dir.name} -> {run_id}, skipping")
                continue

            print(f"  Moving {ts_dir.name} -> runs/{run_id}")
            if not dry_run:
                shutil.move(str(ts_dir), str(target))

                # Create manifest
                manifest_data = create_manifest_from_dir(target)
                manifest_data["original_timestamp_name"] = ts_dir.name
                write_manifest_atomic(target / "manifest.json", manifest_data)

                # Create symlink back if run was completed
                if manifest_data["completed"]:
                    symlink_path = artifacts_dir / ts_dir.name
                    relative_target = Path("../runs") / run_id
                    symlink_path.symlink_to(relative_target)
                    print(f"    Created symlink: {ts_dir.name} -> {relative_target}")

    # Step 3: Create manifests for any existing runs/ directories
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                manifest_data = create_manifest_from_dir(run_dir)
                print(f"  Creating manifest for existing run: {run_dir.name}")
                if not dry_run:
                    write_manifest_atomic(manifest_path, manifest_data)

    print(f"{'[DRY RUN] ' if dry_run else ''}Migration complete!")


def main():
    parser = argparse.ArgumentParser(description="Migrate to canonical runs/ structure")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    migrate_artifacts_to_runs(dry_run=args.dry_run)


if __name__ == "__main__":
    main()