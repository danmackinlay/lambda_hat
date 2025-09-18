#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main(csv="llc_sweep_results.csv", out="sweep_plots"):
    df = pd.read_csv(csv)
    Path(out).mkdir(parents=True, exist_ok=True)

    # Focus on "dim" sweep (target_params); filter rows that have that column
    if "target_params" not in df.columns:
        raise SystemExit("CSV lacks target_params; re-run sweep with that dimension.")
    # Keep rows with numeric metrics
    df = df[df["llc_mean"].notna() & df["ess"].notna()]

    # 1) ESS/sec vs target_params
    if "t_sampling" in df.columns:
        df["ess_per_sec"] = df["ess"] / df["t_sampling"].replace(0, pd.NA)
    for s in sorted(df["sampler"].unique()):
        d = df[df["sampler"] == s].groupby("target_params", as_index=False).agg(
            ess=("ess", "median"),
            ess_per_sec=("ess_per_sec", "median"),
            wnv_time=("wnv_time", "median"),
            wnv_fde=("wnv_fde", "median"),
            se=("llc_se", "median"),
        )
        if not d.empty:
            # ESS/sec
            ax = d.plot(x="target_params", y="ess_per_sec", marker="o", legend=False,
                        title=f"{s.upper()}: ESS/sec vs target_params")
            ax.set_xlabel("target_params (problem size)"); ax.set_ylabel("ESS/sec")
            ax.figure.savefig(f"{out}/{s}_ess_per_sec_vs_size.png", dpi=150, bbox_inches="tight"); plt.close(ax.figure)
            # WNV (time)
            ax = d.plot(x="target_params", y="wnv_time", marker="o", legend=False,
                        title=f"{s.upper()}: WNV_time vs target_params")
            ax.set_xlabel("target_params"); ax.set_ylabel("WNV (Var × seconds)")
            ax.figure.savefig(f"{out}/{s}_wnv_time_vs_size.png", dpi=150, bbox_inches="tight"); plt.close(ax.figure)
            # WNV (FDE)
            ax = d.plot(x="target_params", y="wnv_fde", marker="o", legend=False,
                        title=f"{s.upper()}: WNV_FDE vs target_params")
            ax.set_xlabel("target_params"); ax.set_ylabel("WNV (Var × FDE)")
            ax.figure.savefig(f"{out}/{s}_wnv_fde_vs_size.png", dpi=150, bbox_inches="tight"); plt.close(ax.figure)

    # 2) Frontier: SE vs FDE for each sampler (median over seeds)
    for s in sorted(df["sampler"].unique()):
        d = df[df["sampler"] == s].copy()
        if {"wnv_fde","llc_se","target_params"}.issubset(d.columns):
            # Given Var = SE^2, a lower WNV_FDE indicates better efficiency.
            # Plot SE vs target_params color-coded by FDE.
            ax = d.plot.scatter(x="target_params", y="llc_se", c="wnv_fde", colormap="viridis",
                                title=f"{s.upper()}: SE vs size (color=WNV_FDE)")
            ax.set_xlabel("target_params"); ax.set_ylabel("SE(LLC)")
            ax.figure.colorbar(ax.collections[0], ax=ax, label="WNV_FDE")
            ax.figure.savefig(f"{out}/{s}_se_vs_size_colored_by_wnv_fde.png", dpi=150, bbox_inches="tight"); plt.close(ax.figure)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("Usage: python plot_sweep.py [CSV_FILE] [OUTPUT_DIR]")
        print("  CSV_FILE: Path to sweep results CSV (default: llc_sweep_results.csv)")
        print("  OUTPUT_DIR: Directory for plots (default: sweep_plots)")
        sys.exit(0)

    csv = sys.argv[1] if len(sys.argv) > 1 else "llc_sweep_results.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "sweep_plots"
    main(csv, out)