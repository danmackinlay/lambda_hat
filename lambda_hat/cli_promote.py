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
    p.add_argument("--runs-root", default="runs", type=Path)
    p.add_argument("--samplers", default="sgld,hmc,mclmc")
    p.add_argument("--outdir", default="assets", type=Path)
    p.add_argument(
        "--plot-name",
        default="trace.png",
        help="Which plot to copy from analysis/ (filename only)",
    )
    args = p.parse_args()

    samplers = [s.strip() for s in args.samplers.split(",") if s.strip()]
    promote(args.runs_root, samplers, args.outdir, plot_name=args.plot_name)


if __name__ == "__main__":
    main()