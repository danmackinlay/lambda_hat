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
    from llc.config import Config
    from llc.experiments import run_experiment

    # Extract control flags before creating Config
    cfg_dict_clean = cfg_dict.copy()
    save_artifacts = cfg_dict_clean.pop("save_artifacts", False)
    # Keep artifacts_dir so Modal can pass volume paths

    cfg = Config(**cfg_dict_clean)

    if save_artifacts:
        # Use the unified pipeline for full artifact generation
        from llc.pipeline import run_one

        out = run_one(cfg, save_artifacts=True, skip_if_exists=True)

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
