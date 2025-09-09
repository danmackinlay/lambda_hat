# llc/tasks.py
from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any

# IMPORTANT: keep imports inside the function if you want to minimize process import overhead
def run_experiment_task(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one experiment config and return a small, JSON-serializable result.
    Safe to pickle (top-level def) and safe to import on cluster/cloud.
    """
    from main import Config, run_experiment  # repo-local imports
    cfg = Config(**cfg_dict)
    llc_sgld, llc_hmc = run_experiment(cfg, verbose=False)
    return {
        "cfg": cfg_dict,
        "llc_sgld": float(llc_sgld),
        "llc_hmc": float(llc_hmc),
    }