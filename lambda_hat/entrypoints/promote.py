from __future__ import annotations

import argparse
from pathlib import Path

from lambda_hat.promote.core import promote, promote_gallery


def main():
    ap = argparse.ArgumentParser("lambda-hat-promote")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gallery", help="Promote newest run per sampler into an asset gallery")
    g.add_argument("--runs-root", required=True)
    g.add_argument("--samplers", required=True, help="Comma-separated (e.g. sgld,hmc)")
    g.add_argument("--outdir", default="runs/promotion")
    g.add_argument("--plot-name", default="trace.png")
    g.add_argument("--snippet-out", default=None)

    s = sub.add_parser("single", help="Copy newest plots per sampler (no snippet)")
    s.add_argument("--runs-root", required=True)
    s.add_argument("--samplers", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--plot-name", default="trace.png")

    args = ap.parse_args()
    runs_root = Path(getattr(args, "runs_root"))
    samplers = [s.strip() for s in getattr(args, "samplers").split(",") if s.strip()]
    outdir = Path(getattr(args, "outdir", "runs/promotion"))
    plot_name = getattr(args, "plot_name", "trace.png")

    if args.cmd == "gallery":
        snippet = Path(args.snippet_out) if args.snippet_out else None
        promote_gallery(runs_root, samplers, outdir, plot_name=plot_name, md_snippet_out=snippet)
    else:
        promote(runs_root, samplers, outdir, plot_name=plot_name)


if __name__ == "__main__":
    main()
