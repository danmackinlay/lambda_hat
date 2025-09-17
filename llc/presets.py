"""Unified preset configurations"""

from dataclasses import replace
from .config import Config


def apply_preset(cfg: Config, preset: str) -> Config:
    """Apply preset configurations for quick testing or thorough analysis"""
    if preset == "quick":
        return replace(
            cfg,
            sgld_steps=1000,
            sgld_warmup=200,
            sgld_eval_every=20,  # More frequent for smooth plots
            sgld_thin=5,
            hmc_draws=200,
            hmc_warmup=100,
            hmc_eval_every=20,
            hmc_thin=2,
            mclmc_draws=400,
            mclmc_eval_every=40,
            mclmc_thin=2,
            chains=4,  # 4 chains for robust R-hat and rank plots
            n_data=1000,
            save_plots=True,  # Key: always save plots in quick preset
        )
    elif preset == "full":
        return replace(
            cfg,
            sgld_steps=10000,
            sgld_warmup=2000,
            sgld_eval_every=100,
            sgld_thin=10,
            hmc_draws=2000,
            hmc_warmup=1000,
            hmc_eval_every=20,
            hmc_thin=5,
            mclmc_draws=4000,
            mclmc_eval_every=40,
            mclmc_thin=5,
            chains=4,
            n_data=5000,
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")
