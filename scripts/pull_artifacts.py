#!/usr/bin/env python
"""
Pull artifacts from the Modal volume.

Usage:
  # Pull a specific run
  uv run python scripts/pull_artifacts.py <run_id>

  # Pull the latest run (no args) or with --latest
  uv run python scripts/pull_artifacts.py
  uv run python scripts/pull_artifacts.py --latest

Options:
  --volume  Name of Modal volume (default: llc-artifacts)
  --target  Local target folder (default: ./artifacts)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import re

VOL_DEFAULT = "llc-artifacts"
ROOT_PREFIX = "/artifacts/"


def _list_volume_paths(volume: str) -> list[str]:
    """Return list of paths in the Modal volume (text-parse fallback)."""
    # We rely on `modal volume ls <volume>` (same call documented in the Makefile).
    # Output format is textual; we grab tokens that look like /artifacts/<name>
    cmd = ["modal", "volume", "ls", volume]
    out = subprocess.check_output(cmd, text=True)
    paths = []
    for line in out.splitlines():
        m = re.search(r"(/artifacts/[^\s]+)", line)
        if m:
            paths.append(m.group(1))
    return sorted(set(paths))


def _pick_latest(paths: list[str]) -> str | None:
    """Pick the most recent-looking path. Prefer timestamp symlinks if present."""
    if not paths:
        return None
    # Prefer timestamped entries like /artifacts/YYYYMMDD-HHMMSS if present
    ts = [p for p in paths if re.search(r"/artifacts/\d{8}-\d{6}$", p)]
    if ts:
        return sorted(ts)[-1]
    # Otherwise fall back to lexical max (works fine for hex run_ids too)
    candidates = [p for p in paths if p.startswith(ROOT_PREFIX)]
    return sorted(candidates)[-1] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", help="Run ID (e.g., d9c9f33dce4e)")
    ap.add_argument("--latest", action="store_true", help="Pull the latest run")
    ap.add_argument("--volume", default=VOL_DEFAULT, help="Modal volume name")
    ap.add_argument("--target", default="artifacts", help="Local target directory")
    args = ap.parse_args()

    # Decide which remote path to pull
    if args.run_id:
        remote_path = f"{ROOT_PREFIX}{args.run_id}"
    else:
        # auto-discover latest
        paths = _list_volume_paths(args.volume)
        latest = _pick_latest(paths)
        if not latest:
            sys.exit("[pull-artifacts] No runs found on the volume.")
        remote_path = latest
        print(f"[pull-artifacts] Auto-selected latest: {remote_path}")

    # Local target path (mirror remote leaf)
    run_leaf = Path(remote_path).name
    target = Path(args.target) / run_leaf
    target.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["modal", "volume", "get", args.volume, remote_path, str(target)]
    print(f"[pull-artifacts] Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"[pull-artifacts] Artifacts pulled into {target}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"[pull-artifacts] Error: {e}")


if __name__ == "__main__":
    main()
