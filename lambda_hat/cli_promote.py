#!/usr/bin/env python3
"""Entry point for lambda-hat-promote: pick the newest
`llc_convergence_combined.png` under outputs/ or multirun/
and copy/symlink it into an assets directory for README use."""

import argparse
from pathlib import Path

from lambda_hat.promote.core import promote_latest_combined


def main() -> None:
    p = argparse.ArgumentParser("lambda-hat-promote")
    p.add_argument(
        "--assets-dir", default="assets/readme", type=Path, help="Destination dir for README assets"
    )
    p.add_argument(
        "--filename", default="llc_convergence_combined.png", help="Artifact filename to promote"
    )
    p.add_argument(
        "--mode",
        choices=["copy", "link"],
        default="copy",
        help="Copy file (default) or create a symlink",
    )
    p.add_argument(
        "--roots",
        nargs="*",
        default=["outputs", "multirun"],
        help="Root directories to search (in order)",
    )
    args = p.parse_args()

    roots = [Path(r) for r in args.roots]
    promote_latest_combined(
        assets_dir=args.assets_dir,
        filename=args.filename,
        mode=args.mode,
        roots=roots,
    )


if __name__ == "__main__":
    main()
