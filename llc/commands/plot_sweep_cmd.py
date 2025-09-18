"""Plot sweep command implementation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_sweep_entry(
    csv_path: str,
    out_dir: str,
    size_col: str,
    samplers: str,
    filters: str,
    logx: bool,
    overwrite: bool
) -> None:
    """Entry point for plot-sweep command."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(csv_path)

    # Basic sanity
    required = {"sampler", size_col, "llc_mean", "llc_se", "ess"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Optional columns (compute if absent)
    if "t_sampling" in df.columns and "ess_per_sec" not in df.columns:
        df["ess_per_sec"] = df["ess"] / df["t_sampling"].replace({0: np.nan})
    for col in ("wnv_time", "wnv_fde"):
        if col not in df.columns:
            df[col] = np.nan  # keep pipeline robust even if WNV not computed yet

    # Filters: activation=relu,x_dist=gauss_iso
    if filters.strip():
        for clause in filters.split(","):
            if "=" not in clause:
                raise ValueError(f"Bad filter '{clause}'. Use key=value.")
            k, v = [x.strip() for x in clause.split("=", 1)]
            if k not in df.columns:
                raise ValueError(f"Unknown filter column '{k}'.")
            df = df[df[k].astype(str) == v]

    if df.empty:
        print("[plot-sweep] No rows after filtering.")
        return

    # Normalize/clean
    keep_samplers = [s.strip() for s in samplers.split(",") if s.strip()]
    df = df[df["sampler"].isin(keep_samplers)]

    # Group to medians across seeds/config duplicates
    agg = df.groupby(["sampler", size_col], as_index=False).agg(
        ess=("ess", "median"),
        ess_per_sec=("ess_per_sec", "median"),
        wnv_time=("wnv_time", "median"),
        wnv_fde=("wnv_fde", "median"),
        se=("llc_se", "median"),
    )

    if agg.empty:
        print("[plot-sweep] No rows for selected samplers/filters.")
        return

    # Helper to save-or-skip
    def save_fig(fig, path):
        path = Path(path)
        if path.exists() and not overwrite:
            print(f"[plot-sweep] exists: {path.name} (use --overwrite)")
        else:
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"[plot-sweep] saved: {path.name}")
        plt.close(fig)

    # 1) ESS/sec vs size
    for s in agg["sampler"].unique():
        d = agg[(agg["sampler"] == s) & np.isfinite(agg["ess_per_sec"])]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(d[size_col], d["ess_per_sec"], marker="o")
        ax.set_xlabel(size_col)
        ax.set_ylabel("ESS/sec")
        ax.set_title(f"{s.upper()}: ESS/sec vs {size_col}")
        if logx:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / f"{s}_ess_per_sec_vs_{size_col}.png")

    # 2) WNV (time) vs size
    for s in agg["sampler"].unique():
        d = agg[(agg["sampler"] == s) & np.isfinite(agg["wnv_time"])]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(d[size_col], d["wnv_time"], marker="o")
        ax.set_xlabel(size_col)
        ax.set_ylabel("WNV (Var × seconds)")
        ax.set_title(f"{s.upper()}: WNV_time vs {size_col}")
        if logx:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / f"{s}_wnv_time_vs_{size_col}.png")

    # 3) WNV (FDE) vs size
    for s in agg["sampler"].unique():
        d = agg[(agg["sampler"] == s) & np.isfinite(agg["wnv_fde"])]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(d[size_col], d["wnv_fde"], marker="o")
        ax.set_xlabel(size_col)
        ax.set_ylabel("WNV (Var × FDE)")
        ax.set_title(f"{s.upper()}: WNV_FDE vs {size_col}")
        if logx:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        save_fig(fig, out / f"{s}_wnv_fde_vs_{size_col}.png")

    # 4) Frontier: SE vs size colored by WNV_FDE
    for s in agg["sampler"].unique():
        d = agg[(agg["sampler"] == s) & np.isfinite(agg["se"])]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(d[size_col], d["se"], c=d["wnv_fde"], cmap="viridis")
        ax.set_xlabel(size_col)
        ax.set_ylabel("SE(LLC)")
        ax.set_title(f"{s.upper()}: SE vs {size_col} (color=WNV_FDE)")
        if logx:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("WNV_FDE")
        save_fig(fig, out / f"{s}_se_vs_{size_col}_colored_by_wnv_fde.png")