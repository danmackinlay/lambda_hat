from __future__ import annotations
import argparse
from pathlib import Path
from lambda_hat.promote.core import promote, promote_gallery


def main_promote():
    ap = argparse.ArgumentParser("lambda-hat-promote")
    ap.add_argument("runs_root", type=str, help="runs root directory (e.g., runs)")
    ap.add_argument(
        "--samplers",
        type=str,
        required=True,
        help="comma-separated samplers, e.g. sgld,hmc,mclmc",
    )
    ap.add_argument("--outdir", type=str, default="assets", help="output assets dir")
    ap.add_argument(
        "--plot_name", type=str, default="trace.png", help="diagnostics plot to promote"
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["single", "gallery"],
        default="single",
        help="promotion mode",
    )
    ap.add_argument(
        "--snippet_out",
        type=str,
        default="",
        help="(gallery) write README snippet to this path",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    samplers = [s.strip() for s in args.samplers.split(",") if s.strip()]

    if args.mode == "single":
        promote(runs_root, samplers, outdir, plot_name=args.plot_name)
    else:
        snippet = Path(args.snippet_out) if args.snippet_out else None
        promote_gallery(
            runs_root,
            samplers,
            outdir,
            plot_name=args.plot_name,
            md_snippet_out=snippet,
        )


if __name__ == "__main__":
    main_promote()
