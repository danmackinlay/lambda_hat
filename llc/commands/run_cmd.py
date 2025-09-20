"""Run command implementation."""

import os
from dataclasses import replace
from llc.config import CFG
from llc.util.config_overrides import apply_preset_then_overrides
from llc.util.modal_utils import extract_modal_runs_locally
from llc.util.backend_bootstrap import (
    select_jax_platform,
    validate_modal_gpu_types,
    pick_modal_remote_fn,
    schema_stamp,
)


def run_entry(kwargs: dict) -> None:
    """Entry point for run command."""
    save_artifacts = not kwargs.pop("no_artifacts", False)
    skip_if_exists = kwargs.pop("skip_if_exists", True)
    preset = kwargs.pop("preset", None)
    sampler_choice = kwargs.pop("sampler", None)
    backend = (kwargs.pop("backend") or "local").lower()
    gpu_mode = kwargs.pop("gpu_mode", "off")
    gpu_types = kwargs.pop("gpu_types", "")

    # Submitit-specific parameters
    slurm_partition = kwargs.pop("slurm_partition", None)
    timeout_min = kwargs.pop("timeout_min", 180)
    cpus = kwargs.pop("cpus", 4)
    mem_gb = kwargs.pop("mem_gb", 16)
    slurm_signal_delay_s = kwargs.pop("slurm_signal_delay_s", 120)

    # Set JAX platform for local backend only (remote decides from decorator)
    if backend == "local":
        select_jax_platform(gpu_mode)

    # Validate and set GPU type list for modal_app decorators (evaluated at import time)
    validate_modal_gpu_types(gpu_types)

    # Build config = preset + overrides
    cfg = apply_preset_then_overrides(CFG, preset, kwargs)

    # Enforce single-sampler intent for `run`
    if sampler_choice:
        cfg = replace(cfg, samplers=[sampler_choice])
    if not cfg.samplers or len(cfg.samplers) != 1:
        raise SystemExit(
            "llc run: exactly one sampler must be specified. "
            "Pass --sampler {sgld|sghmc|hmc|mclmc} or use `llc sweep` for multi-sampler jobs."
        )

    # Map GPU mode to batching configuration
    if gpu_mode == "vectorized":
        cfg = replace(cfg, use_batched_chains=True)
    elif gpu_mode == "sequential":
        cfg = replace(cfg, use_batched_chains=False)

    if backend == "local":
        # Import heavy JAX-touching modules only when needed
        from llc.pipeline import run_one

        result = run_one(
            cfg, save_artifacts=save_artifacts, skip_if_exists=skip_if_exists
        )
        _print_summary_like_argparse(result)
        return

    # Prepare payload for remote executors
    cfg_dict = schema_stamp(cfg, save_artifacts, skip_if_exists, gpu_mode)

    # Import execution modules only when needed for remote backends
    from llc.execution import get_executor
    from llc.tasks import run_experiment_task

    if backend == "modal":
        from llc.modal_app import app, ping

        remote_fn = pick_modal_remote_fn(gpu_mode)

        with app.run():
            try:
                # Fast preflight: detect "out of funds" / account disabled immediately
                ping.remote()
            except Exception as e:
                msg = str(e).lower()
                if any(k in msg for k in ["insufficient", "funds", "balance", "quota", "billing"]):
                    raise SystemExit(
                        "Modal preflight failed: likely out of funds or billing disabled.\n"
                        "Tip: top up your Modal balance or set auto-recharge, then retry."
                    )
                raise

            executor = get_executor(backend="modal", remote_fn=remote_fn)
            [result_dict] = executor.map(run_experiment_task, [cfg_dict])

        # Download artifacts locally (optional convenience)
        extract_modal_runs_locally(result_dict)

        # Adapt to summary printer shape
        result = type("RunOutputs", (), {})()
        result.run_dir = result_dict.get("run_dir", "")
        result.metrics = {}
        for s in ("sgld", "sghmc", "hmc", "mclmc"):
            k = f"llc_{s}"
            if k in result_dict:
                result.metrics[f"{s}_llc_mean"] = float(result_dict[k])
        result.histories = {}
        result.L0 = 0.0
        _print_summary_like_argparse(result)
        return

    elif backend == "submitit":
        # Honor GPU intent and pass Submitit parameters
        gpus_per_node = 1 if gpu_mode != "off" else 0
        submitit_kwargs = {
            "gpus_per_node": gpus_per_node,
            "timeout_min": timeout_min,
            "cpus_per_task": cpus,
            "mem_gb": mem_gb,
            "slurm_signal_delay_s": slurm_signal_delay_s,
        }
        if slurm_partition:
            submitit_kwargs["slurm_partition"] = slurm_partition

        executor = get_executor(backend="submitit", **submitit_kwargs)
        [result_dict] = executor.map(run_experiment_task, [cfg_dict])

        # No artifact auto-download for submitit (local FS)
        result = type("RunOutputs", (), {})()
        result.run_dir = result_dict.get("run_dir", "")
        result.metrics = {}
        for s in ("sgld", "sghmc", "hmc", "mclmc"):
            k = f"llc_{s}"
            if k in result_dict:
                result.metrics[f"{s}_llc_mean"] = float(result_dict[k])
        result.histories = {}
        result.L0 = 0.0
        _print_summary_like_argparse(result)
        return

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _print_summary_like_argparse(result):
    """Print summary in argparse-compatible format."""
    import logging
    logger = logging.getLogger(__name__)

    logger.info("=== Final Results ===")
    for key, value in (result.metrics or {}).items():
        if "llc_mean" in key:
            sampler = key.replace("_llc_mean", "").upper()
            se_key = key.replace("_mean", "_se")
            se_value = (result.metrics or {}).get(se_key, 0)
            logger.info(f"{sampler} LLC: {value:.4f} ± {float(se_value):.4f}")
    if getattr(result, "run_dir", ""):
        logger.info(f"Run saved to: {result.run_dir}")
