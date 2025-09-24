# llc/util/backend_bootstrap.py
"""
LEGACY: Backend bootstrap utilities. These are now consolidated in backend_dispatch.py.

Remaining functions are used by the unified dispatcher:
- select_jax_platform: Sets JAX_PLATFORMS environment variable
- validate_modal_gpu_types: Validates GPU types for Modal decorators
- pick_modal_remote_fn: Selects Modal remote function based on GPU mode
- schema_stamp: Adds metadata to config dicts for remote execution

For new code, use llc.util.backend_dispatch.BackendOptions and run_jobs() instead.
"""

import logging
import os
from typing import Any, Dict

from llc.config import Config, config_schema_hash

logger = logging.getLogger(__name__)


def select_jax_platform(gpu_mode: str) -> None:
    """Set JAX_PLATFORMS environment variable based on GPU mode."""
    os.environ["JAX_PLATFORMS"] = "cuda" if gpu_mode != "off" else "cpu"


def validate_modal_gpu_types(gpu_types_str: str) -> None:
    """Validate and set LLC_MODAL_GPU_LIST for Modal app decorators."""
    if not gpu_types_str:
        return

    requested = [g.strip() for g in gpu_types_str.split(",") if g.strip()]
    allowed = {"T4", "L4", "A10G", "A100", "H100", "L40S"}
    bad = [g for g in requested if g not in allowed]

    if bad:
        logger.warning(f"unknown GPU types {bad}; falling back to L40S")
        os.environ["LLC_MODAL_GPU_LIST"] = "L40S"
    else:
        os.environ["LLC_MODAL_GPU_LIST"] = ",".join(requested)


def pick_modal_remote_fn(gpu_mode: str):
    """Return appropriate Modal remote function based on GPU mode."""
    from llc.modal_app import run_experiment_remote, run_experiment_remote_gpu

    return run_experiment_remote_gpu if gpu_mode != "off" else run_experiment_remote


def schema_stamp(
    cfg: Config,
    save_artifacts: bool = False,
    skip_if_exists: bool = True,
    gpu_mode: str = "off",
) -> Dict[str, Any]:
    """Add schema, artifacts, and GPU metadata to config dict for remote execution."""
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["save_artifacts"] = save_artifacts
    cfg_dict["skip_if_exists"] = skip_if_exists
    cfg_dict["config_schema"] = config_schema_hash()
    cfg_dict["gpu_mode"] = gpu_mode
    return cfg_dict
