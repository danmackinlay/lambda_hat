# lambda_hat/commands/artifacts_cmd.py
"""Artifact management commands - GC, list, and TensorBoard helpers."""

import json
import os
import shutil
import time
from typing import Dict, Optional

from lambda_hat.artifacts import ArtifactStore, Paths


def _collect_reachable(paths: Paths) -> set:
    """Collect all reachable artifact URNs from experiment manifests.

    Args:
        paths: Paths object with experiments directory

    Returns:
        set: Set of URN strings that are referenced in manifests
    """
    reachable = set()
    for exp_manifest in paths.experiments.glob("*/manifest.jsonl"):
        try:
            for line in exp_manifest.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                for key in ("inputs", "outputs"):
                    for item in rec.get(key, []):
                        urn = item.get("urn")
                        if urn:
                            reachable.add(urn)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    return reachable


def gc_entry(ttl_days: Optional[int] = None) -> Dict:
    """Garbage collect unreachable artifacts and old run directories.

    Args:
        ttl_days: Time-to-live in days (default: from LAMBDA_HAT_TTL_DAYS env or 30)

    Returns:
        dict: GC statistics with keys:
            - removed: Number of objects removed
            - ttl_days: TTL used
    """
    paths = Paths.from_env()
    paths.ensure()
    store = ArtifactStore(paths.store)

    # Get TTL from env or argument
    if ttl_days is None:
        ttl_days = int(os.environ.get("LAMBDA_HAT_TTL_DAYS", "30"))
    cutoff = time.time() - ttl_days * 86400

    # Prune run scratch/logs older than TTL
    for run_dir in paths.experiments.glob("*/runs/*"):
        try:
            mtime = run_dir.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime < cutoff and run_dir.is_dir():
            for subdir in ("scratch", "logs", "parsl"):
                p = run_dir / subdir
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)

    # Prune unreachable objects in store older than TTL
    reachable = _collect_reachable(paths)
    base = paths.store / "objects" / "sha256"
    removed = 0

    if base.exists():
        for obj_dir in base.glob("*/*/*"):
            meta_path = obj_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
                urn = f"urn:lh:{meta.get('type', 'unknown')}:sha256:{meta['hash']['hex']}"
                if urn in reachable:
                    continue
                if meta_path.stat().st_mtime < cutoff:
                    shutil.rmtree(obj_dir, ignore_errors=True)
                    removed += 1
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                continue

    print(f"GC removed {removed} unreachable objects (older than {ttl_days}d).")
    return {"removed": removed, "ttl_days": ttl_days}


def ls_entry() -> None:
    """List experiments and their runs to stdout."""
    paths = Paths.from_env()
    paths.ensure()

    experiments = sorted(d.name for d in paths.experiments.glob("*") if d.is_dir())
    if not experiments:
        print("No experiments found.")
        return

    for exp_name in experiments:
        print(f"[{exp_name}]")
        manifest = paths.experiments / exp_name / "manifest.jsonl"
        if not manifest.exists():
            print("  (no runs)")
            continue

        try:
            for line in manifest.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                run_id = rec.get("run_id", "?")
                algo = rec.get("algo", "?")
                phase = rec.get("phase", "?")
                print(f"  {run_id}  {algo}  phase={phase}")
        except (json.JSONDecodeError, FileNotFoundError):
            print("  (error reading manifest)")


def tb_entry(experiment: str) -> str:
    """Get TensorBoard logdir path for an experiment.

    Args:
        experiment: Experiment name

    Returns:
        str: Path to TensorBoard logdir
    """
    paths = Paths.from_env()
    paths.ensure()
    tb_dir = paths.experiments / experiment / "tb"
    print(f"TensorBoard logdir: {tb_dir}")
    return str(tb_dir)
