"""Configuration override utilities for CLI."""

from __future__ import annotations
from dataclasses import replace
from typing import Optional

from llc.config import Config
from llc.presets import apply_preset


def override_config(cfg: Config, args: dict) -> Config:
    """Apply command-line overrides to configuration (mirror argparse override_config)."""
    overrides = {}

    # Simple direct mappings
    direct_mappings = [
        "n_data",
        "seed",
        "loss",
        "depth",
        "chains",
        "sgld_steps",
        "sgld_warmup",
        "sgld_step_size",
        "sgld_batch_size",
        "sgld_eval_every",
        "sgld_thin",
        "hmc_draws",
        "hmc_warmup",
        "hmc_eval_every",
        "hmc_thin",
        "mclmc_draws",
        "mclmc_eval_every",
        "mclmc_thin",
    ]
    for attr in direct_mappings:
        if args.get(attr) is not None:
            overrides[attr] = args[attr]

    # SGLD preconditioning
    if args.get("sgld_precond") is not None:
        overrides["sgld_precond"] = args["sgld_precond"]
    if args.get("sgld_beta1") is not None:
        overrides["sgld_beta1"] = args["sgld_beta1"]
    if args.get("sgld_beta2") is not None:
        overrides["sgld_beta2"] = args["sgld_beta2"]
    if args.get("sgld_eps") is not None:
        overrides["sgld_eps"] = args["sgld_eps"]
    if args.get("sgld_bias_correction") is not None:
        overrides["sgld_bias_correction"] = args["sgld_bias_correction"]

    # Batched chains
    if args.get("use_batched_chains") is not None:
        overrides["use_batched_chains"] = args["use_batched_chains"]

    # width -> widths
    if args.get("width") is not None:
        overrides["widths"] = [args["width"]] * cfg.depth

    # target_params -> infer widths
    if args.get("target_params") is not None:
        from llc.models import infer_widths

        overrides["widths"] = infer_widths(
            cfg.in_dim, cfg.out_dim, cfg.depth, args["target_params"]
        )

    # Target selection
    if args.get("target") is not None:
        overrides["target"] = args["target"]
    if args.get("quad_dim") is not None:
        overrides["quad_dim"] = args["quad_dim"]

    # Plotting
    if args.get("save_plots") is not None:
        overrides["save_plots"] = args["save_plots"]

    return replace(cfg, **overrides) if overrides else cfg


def apply_preset_then_overrides(
    cfg: Config, preset: Optional[str], kwargs: dict
) -> Config:
    """Apply preset then command-line overrides to configuration."""
    out = cfg
    if preset:
        out = apply_preset(out, preset)
    return override_config(out, kwargs)