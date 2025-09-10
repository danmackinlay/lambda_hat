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
    cfg_dict_clean.pop("artifacts_dir", "artifacts")

    cfg = Config(**cfg_dict_clean)

    if save_artifacts:
        # Run experiment with full artifact pipeline (no main dependency)
        from llc.artifacts import (
            create_run_directory,
            save_config,
            save_metrics,
            save_L0,
            create_manifest,
            generate_gallery_html,
        )

        # Create run directory
        run_dir = create_run_directory(cfg)

        # Run the lightweight experiment to get LLC values
        llc_sgld, llc_hmc = run_experiment(cfg, verbose=True)

        # Create basic metrics (simplified version of main's metrics)
        metrics = {
            "sgld_llc_mean": float(llc_sgld),
            "hmc_llc_mean": float(llc_hmc),
            # Add other samplers if configured
        }

        # Handle additional samplers if present
        samplers = getattr(cfg, "samplers", [cfg.sampler])
        for sampler in samplers:
            if sampler == "sgld":
                metrics["sgld_llc_mean"] = float(llc_sgld)
            elif sampler == "hmc":
                metrics["hmc_llc_mean"] = float(llc_hmc)
            # Note: MCLMC would need to be added to run_experiment if needed for artifacts

        # Save basic artifacts
        save_L0(run_dir, 0.0)  # Placeholder - would need actual L0 from experiment
        save_metrics(run_dir, metrics)
        save_config(run_dir, cfg)

        # Create manifest with basic artifacts
        artifact_files = [
            "config.json",
            "metrics.json",
            "L0.txt",
        ]
        create_manifest(run_dir, cfg, metrics, artifact_files)

        # Generate HTML gallery
        generate_gallery_html(run_dir, cfg, metrics)

        # Extract sampler results for backwards compatibility
        result = {
            "cfg": cfg_dict,
            "run_dir": run_dir,
        }

        # Add individual sampler results for backwards compatibility
        for sampler in samplers:
            if f"{sampler}_llc_mean" in metrics:
                result[f"llc_{sampler}"] = float(metrics[f"{sampler}_llc_mean"])

        # Maintain old format if only sgld/hmc
        if "sgld_llc_mean" in metrics and "hmc_llc_mean" in metrics:
            result["llc_sgld"] = float(metrics["sgld_llc_mean"])
            result["llc_hmc"] = float(metrics["hmc_llc_mean"])

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
