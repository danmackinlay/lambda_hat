"""Configuration override utilities for CLI."""

from __future__ import annotations
from dataclasses import fields, replace
from typing import Optional

from llc.config import Config
from llc.presets import apply_preset


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
    return override_config(out, kwargs)
