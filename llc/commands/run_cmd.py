"""Run command implementation."""

import os
from dataclasses import replace
from llc.config import CFG, config_schema_hash
from llc.util.config_overrides import apply_preset_then_overrides
from llc.util.modal_utils import extract_modal_runs_locally


def run_entry(kwargs: dict) -> None:
    """Entry point for run command."""
    save_artifacts = not kwargs.pop("no_artifacts", False)
    skip_if_exists = kwargs.pop("skip_if_exists", True)
    preset = kwargs.pop("preset", None)
    backend = (kwargs.pop("backend") or "local").lower()
    gpu_mode = kwargs.pop("gpu_mode", "off")
    cuda_devices = kwargs.pop("cuda_devices", None)

    # Set JAX platform before any JAX imports
    if gpu_mode == "off":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ["JAX_PLATFORMS"] = "cuda"
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    # Build config = preset + overrides
    cfg = apply_preset_then_overrides(CFG, preset, kwargs)

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
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["save_artifacts"] = save_artifacts
    cfg_dict["skip_if_exists"] = skip_if_exists
    cfg_dict["config_schema"] = config_schema_hash()
    cfg_dict["gpu_mode"] = gpu_mode  # Pass GPU mode to remote executor

    # Import execution modules only when needed for remote backends
    from llc.execution import get_executor
    from llc.tasks import run_experiment_task

    if backend == "modal":
        # Choose GPU or CPU Modal function based on gpu_mode
        if gpu_mode == "off":
            from llc.modal_app import app, run_experiment_remote

            remote_fn = run_experiment_remote
        else:
            from llc.modal_app import app, run_experiment_remote_gpu

            remote_fn = run_experiment_remote_gpu

        with app.run():
            executor = get_executor(backend="modal", remote_fn=remote_fn)
            [result_dict] = executor.map(run_experiment_task, [cfg_dict])

        # Download artifacts locally (optional convenience)
        extract_modal_runs_locally(result_dict)

        # Adapt to summary printer shape
        result = type("RunOutputs", (), {})()
        result.run_dir = result_dict.get("run_dir", "")
        result.metrics = {}
        for s in ("sgld", "hmc", "mclmc"):
            k = f"llc_{s}"
            if k in result_dict:
                result.metrics[f"{s}_llc_mean"] = float(result_dict[k])
        result.histories = {}
        result.L0 = 0.0
        _print_summary_like_argparse(result)
        return

    elif backend == "submitit":
        executor = get_executor(backend="submitit")
        [result_dict] = executor.map(run_experiment_task, [cfg_dict])

        # No artifact auto-download for submitit (local FS)
        result = type("RunOutputs", (), {})()
        result.run_dir = result_dict.get("run_dir", "")
        result.metrics = {}
        for s in ("sgld", "hmc", "mclmc"):
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
    print("\n=== Final Results ===")
    for key, value in (result.metrics or {}).items():
        if "llc_mean" in key:
            sampler = key.replace("_llc_mean", "").upper()
            se_key = key.replace("_mean", "_se")
            se_value = (result.metrics or {}).get(se_key, 0)
            print(f"{sampler} LLC: {value:.4f} ± {float(se_value):.4f}")
    if getattr(result, "run_dir", ""):
        print(f"\nArtifacts saved to: {result.run_dir}")
