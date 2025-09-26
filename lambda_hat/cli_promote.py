#!/usr/bin/env python3
"""
Entry point for lambda-hat-promote command.
"""

import argparse
from pathlib import Path
from lambda_hat.promote.core import promote


def main() -> None:
    """Entry point for lambda-hat-promote command."""
    p = argparse.ArgumentParser("lambda-hat-promote")
    p.add_argument("--runs-root", default="outputs", type=Path,
                   help="Root directory for run outputs (default: outputs)")
    p.add_argument("--samplers", default="sgld,hmc,mclmc")
    p.add_argument("--outdir", default="assets", type=Path)
    p.add_argument(
        "--plot-name",
        default="sgld_trace.png",
        help="Which plot to copy from diagnostics/ (filename only)",
    )
    p.add_argument("--max-dirs", type=int, default=5000,
                   help="Max run directories to scan before bailing (default: 5000)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose output showing scan progress")
    args = p.parse_args()

    samplers = [s.strip() for s in args.samplers.split(",") if s.strip()]
    promote(args.runs_root, samplers, args.outdir,
            plot_name=args.plot_name,
            max_dirs=args.max_dirs,
            verbose=args.verbose)


if __name__ == "__main__":
    main()
