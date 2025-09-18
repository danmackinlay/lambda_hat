# llc/tasks.py
"""
Experiment task runner. All execution paths now use batched samplers exclusively.
Safe for pickling and cluster/cloud execution.
"""

from __future__ import annotations
from typing import Dict, Any


# IMPORTANT: keep imports inside the function if you want to minimize process import overhead
def run_experiment_task(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one experiment config and return a small, JSON-serializable result.
    Safe to pickle (top-level def) and safe to import on cluster/cloud.

    Always delegates to pipeline.run_one with save_artifacts controlling I/O.
    Returns uniform shape with cfg, run_dir, and llc_{sampler} values.
    """
    from dataclasses import fields
    from llc.config import Config, config_schema_hash
    from llc.pipeline import run_one

    cfg_dict_clean = dict(cfg_dict)
    # control flags (not part of Config)
    save_artifacts = bool(cfg_dict_clean.pop("save_artifacts", False))
    skip_if_exists = bool(cfg_dict_clean.pop("skip_if_exists", True))
    provided_schema = cfg_dict_clean.pop("config_schema", None)

    # Drop unknown keys to tolerate remote/client skew (but be loud).
    allowed = {f.name for f in fields(Config)}
    dropped = sorted(set(cfg_dict_clean) - allowed)
    cfg_kwargs = {k: v for k, v in cfg_dict_clean.items() if k in allowed}

    # Schema handshake: if provided, must match.
    local_schema = config_schema_hash()
    if provided_schema and provided_schema != local_schema:
        raise RuntimeError(
            "Config schema mismatch between client and worker.\n"
            f"  client schema: {provided_schema}\n"
            f"  worker schema: {local_schema}\n"
            "Redeploy the Modal app or use object-based remote function to auto-deploy."
        )
    if dropped:
        print(f"[llc] Warning: dropping unknown config keys: {dropped}")

    cfg = Config(**cfg_kwargs)

    # Single path: pipeline is source of truth. save_artifacts governs I/O.
    out = run_one(cfg, save_artifacts=save_artifacts, skip_if_exists=skip_if_exists)

    # Uniform, JSON-serializable result shape.
    result: Dict[str, Any] = {
        "cfg": cfg_dict,
        "run_dir": out.run_dir or "",
    }
    for s in ("sgld", "hmc", "mclmc"):
        k = f"{s}_llc_mean"
        if k in out.metrics:
            result[f"llc_{s}"] = float(out.metrics[k])
    return result
