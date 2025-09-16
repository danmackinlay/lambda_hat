# llc/tasks.py
from __future__ import annotations
from typing import Dict, Any


# IMPORTANT: keep imports inside the function if you want to minimize process import overhead
def run_experiment_task(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one experiment config and return a small, JSON-serializable result.
    Safe to pickle (top-level def) and safe to import on cluster/cloud.

    If cfg_dict contains 'save_artifacts': True, will generate full artifact pipeline
    and return run_dir path for artifact retrieval.
    """
    from dataclasses import fields
    from llc.config import Config, config_schema_hash
    from llc.experiments import run_experiment

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

    if save_artifacts:
        # Use the unified pipeline for full artifact generation
        from llc.pipeline import run_one

        out = run_one(cfg, save_artifacts=True, skip_if_exists=skip_if_exists)

        # Keep return shape backwards-compatible with existing callers
        result = {
            "cfg": cfg_dict,
            "run_dir": out.run_dir,
        }

        # Add individual sampler results for backwards compatibility
        for s in ("sgld", "hmc", "mclmc"):
            if f"{s}_llc_mean" in out.metrics:
                result[f"llc_{s}"] = float(out.metrics[f"{s}_llc_mean"])

        # Legacy keys for backwards compatibility
        if "sgld_llc_mean" in out.metrics and "hmc_llc_mean" in out.metrics:
            result["llc_sgld"] = float(out.metrics["sgld_llc_mean"])
            result["llc_hmc"] = float(out.metrics["hmc_llc_mean"])

        return result

    else:
        # Original lightweight mode - just run experiment function
        try:
            llc_sgld, llc_hmc = run_experiment(cfg, verbose=False)
            return {
                "cfg": cfg_dict,
                "llc_sgld": float(llc_sgld),
                "llc_hmc": float(llc_hmc),
            }
        except ValueError as e:
            # Handle cases where sampling doesn't produce enough samples
            if "min() arg is an empty sequence" in str(
                e
            ) or "too many values to unpack" in str(e):
                return {
                    "cfg": cfg_dict,
                    "llc_sgld": float("nan"),
                    "llc_hmc": float("nan"),
                    "error": str(e),
                }
            else:
                raise
