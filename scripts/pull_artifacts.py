#!/usr/bin/env python3
"""
Pull artifacts from Modal using the SDK, not the CLI.

Usage:
  # Pull the LATEST run (auto-detected)
  uv run python scripts/pull_artifacts.py

  # Pull a specific run by ID
  uv run python scripts/pull_artifacts.py <run_id>

It writes to ./artifacts/<run_id>/ locally.
"""

import argparse
import io
import tarfile
from pathlib import Path
import modal

APP = "llc-experiments"  # app name
FN_LIST = "list_artifacts"  # server-side lister
FN_EXPORT = "export_artifacts"  # server-side exporter


def main():
    ap = argparse.ArgumentParser(description="Pull artifacts from Modal using SDK")
    ap.add_argument(
        "run_id", nargs="?", help="Run ID (e.g. f1ce73101e3d). Omit for latest."
    )
    ap.add_argument(
        "--target", default="artifacts", help="Local target root (default: ./artifacts)"
    )
    args = ap.parse_args()

    # Look up deployed functions
    list_fn = modal.Function.from_name(APP, FN_LIST)
    export_fn = modal.Function.from_name(APP, FN_EXPORT)

    # Pick run_id
    if args.run_id:
        run_id = args.run_id
        print(f"[pull-sdk] Pulling specific run: {run_id}")
    else:
        # Ask server for list; pick latest by name (timestamps sort after hex ids)
        print("[pull-sdk] Discovering latest run on server...")
        paths = list_fn.remote("/artifacts")
        if not paths:
            raise SystemExit("No remote artifacts found.")
        run_id = Path(sorted(paths)[-1]).name
        print(f"[pull-sdk] Latest on server: {run_id}")

    # Fetch tarball and extract
    print(f"[pull-sdk] Downloading and extracting {run_id}...")
    data = export_fn.remote(run_id)
    dest_root = Path(args.target)
    dest_root.mkdir(parents=True, exist_ok=True)

    # Clean any existing directory to ensure fresh extraction
    target_dir = dest_root / run_id
    if target_dir.exists():
        import shutil

        shutil.rmtree(target_dir)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(dest_root)
    print(f"[pull-sdk] Extracted into {dest_root / run_id}")


if __name__ == "__main__":
    main()
