#!/usr/bin/env python3
"""
scripts/make_readme_assets.py

Runs SGLD, HMC, MCLMC once each (quick preset), then promotes selected ArviZ plots
into assets/readme/ for inclusion in the front-page README.

Usage:
  uv run python scripts/make_readme_assets.py
"""
import os
import shutil
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt

from llc.config import Config, apply_preset, override_config
from llc.run import run_one
from llc.analysis import (
    fig_running_llc, fig_rank_llc, fig_autocorr_llc, fig_energy, fig_theta_trace
)

ASSETS_DIR = Path("assets/readme")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def run_and_promote(sampler: str, base_cfg: Config) -> None:
    # Ensure single-sampler run with plots enabled
    cfg = override_config(base_cfg, {"samplers": (sampler,), "save_plots": True})
    out = run_one(cfg, save_artifacts=True, skip_if_exists=False)
    run_dir = Path(out["run_dir"])
    nc_path = run_dir / f"{sampler}.nc"

    # Copy the running-LLC plot if run_one already saved it
    src_running = run_dir / f"{sampler}_running_llc.png"
    dst_running = ASSETS_DIR / f"{sampler}_llc_running.png"
    if src_running.exists():
        shutil.copyfile(src_running, dst_running)

    # Load idata and generate a small gallery of ArviZ figures
    idata = az.from_netcdf(nc_path)
    beta = float(idata.attrs.get("beta", 1.0))
    n = int(idata.attrs.get("n_data", 1))
    L0 = float(idata.attrs.get("L0", 0.0))

    # Running LLC (recreate to ensure consistent styling)
    fig = fig_running_llc(idata, n, beta, L0, f"{sampler.upper()} Running LLC")
    _save(fig, ASSETS_DIR / f"{sampler}_llc_running.png")

    # Rank
    try:
        fig = fig_rank_llc(idata)
        _save(fig, ASSETS_DIR / f"{sampler}_rank.png")
    except Exception:
        pass

    # Autocorr
    try:
        fig = fig_autocorr_llc(idata)
        _save(fig, ASSETS_DIR / f"{sampler}_autocorr.png")
    except Exception:
        pass

    # Energy (HMC/MCLMC populate sample_stats.energy)
    fig = fig_energy(idata)
    if fig is not None:
        _save(fig, ASSETS_DIR / f"{sampler}_energy.png")

    # Theta trace (stores first few dims if available)
    fig = fig_theta_trace(idata)
    if fig is not None:
        _save(fig, ASSETS_DIR / f"{sampler}_theta.png")

def main():
    # Prefer CPU by default for repeatability; comment out to auto-pick GPU if present
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    base = apply_preset(Config(), "quick")
    # Use a tiny quadratic or DLN target for speed; uncomment one:
    # base = override_config(base, {"target": "quadratic", "quad_dim": 16})
    base = override_config(base, {"target": "mlp"})  # default small MLP in quick preset

    for sampler in ("sgld", "hmc", "mclmc"):
        print(f"Running {sampler.upper()}...")
        run_and_promote(sampler, base)

    print(f"Assets written to {ASSETS_DIR.resolve()}")

if __name__ == "__main__":
    main()