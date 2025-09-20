"""Analyze command implementation."""

import logging
from pathlib import Path


def analyze_entry(
    run_dir: str, which: str, plots: str, out: str = None, overwrite: bool = False
) -> None:
    """Entry point for analyze command."""
    from arviz import from_netcdf
    from llc.analysis import (
        llc_point_se,
        fig_running_llc,
        fig_rank_llc,
        fig_ess_evolution,
        fig_ess_quantile,
        fig_autocorr_llc,
        fig_energy,
        fig_theta_trace,
    )

    logger = logging.getLogger(__name__)
    run_dir = Path(run_dir)
    out_dir = Path(out or (run_dir / "analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    which = ["sgld", "sghmc", "hmc", "mclmc"] if which == "all" else [which]
    plots = [p.strip() for p in plots.split(",") if p.strip()]

    for s in which:
        nc = run_dir / f"{s}.nc"
        if not nc.exists():
            logger.info(f"[analyze] skip {s}: {nc.name} not found")
            continue

        try:
            idata = from_netcdf(nc)
        except Exception as e:
            logger.warning(f"[analyze] failed to load {s}: {e}")
            continue

        # metrics
        m = llc_point_se(idata)
        logger.info(
            f"[{s}] mean={m.get('llc_mean', float('nan')):.4g} se={m.get('llc_se', float('nan')):.3g} "
            f"ESS={m.get('ess_bulk', float('nan')):.1f} Rhat={m.get('rhat', float('nan')):.3f}"
        )

        # figures
        for p in plots:
            try:
                if p == "running_llc":
                    # need L0, n, beta; derive from attrs if you stored them, else skip:
                    n_attr = idata.attrs.get("n_data", None)
                    if n_attr is None:
                        logger.warning(f"[{s}] n_data missing in attrs; reading {run_dir/'config.json'}")
                        try:
                            import json
                            with open(run_dir/'config.json') as f:
                                n = int(json.load(f)["n_data"])
                        except Exception:
                            logger.error(f"[{s}] cannot determine n_data; skipping running_llc")
                            continue
                    else:
                        n = int(n_attr)
                    beta = float(idata.attrs.get("beta", 1.0))
                    L0 = float(idata.attrs.get("L0", 0.0))
                    fig = fig_running_llc(idata, n, beta, L0, f"{s} Running LLC")
                    path = out_dir / f"{s}_running_llc.png"
                elif p == "rank":
                    fig = fig_rank_llc(idata)
                    path = out_dir / f"{s}_llc_rank.png"
                elif p == "ess_evolution":
                    fig = fig_ess_evolution(idata)
                    path = out_dir / f"{s}_llc_ess_evolution.png"
                elif p == "ess_quantile":
                    fig = fig_ess_quantile(idata)
                    path = out_dir / f"{s}_llc_ess_quantile.png"
                elif p == "autocorr":
                    fig = fig_autocorr_llc(idata)
                    path = out_dir / f"{s}_llc_autocorr.png"
                elif p == "energy":
                    # only if this idata actually has sample_stats.energy
                    if "energy" in (getattr(idata, "sample_stats", {}) or {}):
                        fig = fig_energy(idata)
                        path = out_dir / f"{s}_energy.png"
                    else:
                        # silently skip for samplers like SGLD
                        continue
                elif p == "theta":
                    fig = fig_theta_trace(idata, dims=4)
                    path = out_dir / f"{s}_theta_trace.png"
                else:
                    continue

                if path.exists() and not overwrite:
                    logger.info(f"[analyze] exists: {path.name} (use --overwrite)")
                else:
                    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
                    logger.info(f"[analyze] saved: {path.name}")

                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception as e:
                logger.error(f"[analyze] failed {s}:{p}: {e}")
