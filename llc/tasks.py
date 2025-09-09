# llc/tasks.py
from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any

# IMPORTANT: keep imports inside the function if you want to minimize process import overhead
def run_experiment_task(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one experiment config and return a small, JSON-serializable result.
    Safe to pickle (top-level def) and safe to import on cluster/cloud.
    
    If cfg_dict contains 'save_artifacts': True, will generate full artifact pipeline
    and return run_dir path for artifact retrieval.
    """
    from main import Config, run_experiment, main as run_main  # repo-local imports
    
    # Extract control flags before creating Config
    cfg_dict_clean = cfg_dict.copy()
    save_artifacts = cfg_dict_clean.pop('save_artifacts', False)
    artifacts_dir = cfg_dict_clean.pop('artifacts_dir', 'artifacts')
    
    cfg = Config(**cfg_dict_clean)
    
    if save_artifacts:
        # Run full main() pipeline with artifacts
        run_dir = run_main(cfg)
        
        # Extract just the metrics we need for backwards compatibility
        import json
        import os
        metrics_path = f"{run_dir}/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Extract sampler results
            samplers = getattr(cfg, 'samplers', [cfg.sampler])
            result = {
                "cfg": cfg_dict,
                "run_dir": run_dir,  # NEW: return run_dir for artifact access
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
            # Fallback if metrics file doesn't exist
            return {"cfg": cfg_dict, "run_dir": run_dir, "error": "metrics file not found"}
    
    else:
        # Original lightweight mode - just run experiment function
        llc_sgld, llc_hmc = run_experiment(cfg, verbose=False)
        return {
            "cfg": cfg_dict,
            "llc_sgld": float(llc_sgld),
            "llc_hmc": float(llc_hmc),
        }