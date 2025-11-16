# lambda_hat/entrypoints/lh.py
"""Lambda-Hat artifact management utility."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from lambda_hat.artifacts import ArtifactStore, Paths


def collect_reachable(paths: Paths) -> set[str]:
    """Collect all URNs referenced in experiment manifests."""
    reachable = set()
    for exp in paths.experiments.glob("*/manifest.jsonl"):
        with exp.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for k in ("inputs", "outputs"):
                    for it in rec.get(k, []):
                        urn = it.get("urn")
                        if urn:
                            reachable.add(urn)
    return reachable


def cmd_gc(args):
    """Garbage collect unreachable objects and old scratch."""
    paths = Paths.from_env()
    paths.ensure()
    store = ArtifactStore(paths.store)
    ttl_days = int(os.environ.get("LAMBDA_HAT_TTL_DAYS", "30"))
    cutoff = time.time() - ttl_days * 86400

    # 1) Prune run scratch/logs older than TTL (but keep artifacts + manifests)
    for run_dir in paths.experiments.glob("*/runs/*"):
        try:
            mtime = run_dir.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime < cutoff and run_dir.is_dir():
            # Remove scratch; keep manifest/artifacts/tb
            for sub in ("scratch", "logs", "parsl"):
                p = run_dir / sub
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)

    # 2) Prune unreachable objects in store older than TTL
    reachable = collect_reachable(paths)
    base = paths.store / "objects" / "sha256"
    removed = 0
    for d2 in base.glob("*/*/*"):
        meta_p = d2 / "meta.json"
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        urn = f"urn:lh:{meta.get('type','unknown')}:sha256:{meta['hash']['hex']}"
        if urn in reachable:
            continue
        # Age gate
        if meta_p.stat().st_mtime < cutoff:
            shutil.rmtree(d2, ignore_errors=True)
            removed += 1
    print(f"GC removed {removed} unreachable objects (older than {ttl_days}d).")


def cmd_ls(args):
    """List experiments and runs."""
    paths = Paths.from_env()
    paths.ensure()
    for exp in sorted(d.name for d in paths.experiments.glob("*") if d.is_dir()):
        print(f"[{exp}]")
        m = paths.experiments / exp / "manifest.jsonl"
        if not m.exists():
            print("  (no runs)")
            continue
        with m.open() as f:
            for line in f:
                rec = json.loads(line)
                print(
                    f"  {rec['run_id']}  {rec.get('algo','?')}  phase={rec.get('phase','?')}"
                )


def cmd_tb(args):
    """Show TensorBoard logdir path."""
    paths = Paths.from_env()
    paths.ensure()
    tb_root = paths.experiments / args.experiment / "tb"
    print(f"TensorBoard logdir: {tb_root}")


def main():
    ap = argparse.ArgumentParser("lh", description="Lambda-Hat artifact manager")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gc", help="Garbage collect old artifacts")
    sub.add_parser("ls", help="List experiments and runs")
    tbp = sub.add_parser("tb", help="Show TensorBoard logdir")
    tbp.add_argument("experiment", help="Experiment name")
    args = ap.parse_args()

    if args.cmd == "gc":
        cmd_gc(args)
    elif args.cmd == "ls":
        cmd_ls(args)
    elif args.cmd == "tb":
        cmd_tb(args)


if __name__ == "__main__":
    main()
