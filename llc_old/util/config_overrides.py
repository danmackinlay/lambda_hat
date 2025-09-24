"""Configuration override utilities for CLI."""

from __future__ import annotations
from dataclasses import fields, replace
from typing import Optional

from llc.config import Config
from llc.presets import apply_preset

# Type mapping for Config fields that might come as strings from YAML
_NUMERIC_FIELDS = {
    "n_data": int, "seed": int, "chains": int,
    "sgld_steps": int, "sgld_warmup": int, "sgld_batch_size": int,
    "sgld_eval_every": int, "sgld_thin": int, "sgld_step_size": float,
    "sgld_beta1": float, "sgld_beta2": float, "sgld_eps": float,
    "sghmc_steps": int, "sghmc_warmup": int, "sghmc_batch_size": int,
    "sghmc_step_size": float, "sghmc_temperature": float,
    "hmc_draws": int, "hmc_warmup": int, "hmc_num_integration_steps": int,
    "hmc_eval_every": int, "hmc_thin": int,
    "mclmc_draws": int, "mclmc_num_steps": int, "mclmc_eval_every": int, "mclmc_thin": int,
    "mclmc_frac_tune1": float, "mclmc_frac_tune2": float, "mclmc_frac_tune3": float,
    "mclmc_desired_energy_var": float, "mclmc_trust_in_estimate": float, "mclmc_num_effective_samples": float,
    "beta0": float, "gamma": float, "prior_radius": float,
    "noise_scale": float, "hetero_scale": float, "student_df": float,
    "outlier_frac": float, "outlier_scale": float,
    "depth": int, "in_dim": int, "out_dim": int, "target_params": int,
}


def _coerce_types(cfg: Config) -> Config:
    """Coerce string values from YAML to proper numeric types."""
    d = cfg.__dict__.copy()
    for k, caster in _NUMERIC_FIELDS.items():
        if k in d and d[k] is not None and not isinstance(d[k], (int, float)):
            try:
                d[k] = caster(d[k])
            except Exception:
                raise SystemExit(f"Bad type for '{k}': expected {caster.__name__}, got {type(d[k]).__name__} ({d[k]!r})")
    return replace(cfg, **d)


def override_config(cfg: Config, args: dict) -> Config:
    """Apply command-line overrides to configuration using dataclass introspection."""
    # Build valid field names from Config dataclass
    valid = {f.name for f in fields(Config)}

    # Extract all valid overrides (excluding None values)
    overrides = {k: v for k, v in args.items() if v is not None and k in valid}

    # Special case 1: width -> widths (single width applied to all layers)
    if args.get("width") is not None:
        overrides["widths"] = [args["width"]] * cfg.depth

    # Special case 2: target_params -> infer widths (compute architecture from param count)
    if args.get("target_params") is not None:
        from llc.models import infer_widths

        overrides["widths"] = infer_widths(
            cfg.in_dim, cfg.out_dim, cfg.depth, args["target_params"]
        )

    return replace(cfg, **overrides) if overrides else cfg


def apply_preset_then_overrides(
    cfg: Config, preset: Optional[str], kwargs: dict
) -> Config:
    """Apply preset then command-line overrides to configuration."""
    out = cfg
    if preset:
        out = apply_preset(out, preset)
    out = override_config(out, kwargs)
    return _coerce_types(out)  # 🔧 enforce numeric types once
