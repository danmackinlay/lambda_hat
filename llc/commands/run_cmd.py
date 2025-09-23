"""Run command implementation."""

from dataclasses import replace
from llc.config import CFG
from llc.util.config_overrides import apply_preset_then_overrides
from llc.util.backend_dispatch import BackendOptions, prepare_payloads, run_jobs
from llc.tasks import run_experiment_task


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
    slurm_account = kwargs.pop("slurm_account", None)
    timeout_min = kwargs.pop("timeout_min", 180)
    cpus = kwargs.pop("cpus", 4)
    mem_gb = kwargs.pop("mem_gb", 16)
    slurm_signal_delay_s = kwargs.pop("slurm_signal_delay_s", 120)


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

    # Use unified backend dispatcher for remote execution
    opts = BackendOptions(
        backend=backend,
        gpu_mode=gpu_mode,
        gpu_types=gpu_types,
        slurm_partition=slurm_partition,
        slurm_account=slurm_account,
        timeout_min=timeout_min,
        cpus=cpus,
        mem_gb=mem_gb,
        slurm_signal_delay_s=slurm_signal_delay_s,
    )

    cfg_payloads = prepare_payloads([cfg], save_artifacts=save_artifacts, skip_if_exists=skip_if_exists, gpu_mode=gpu_mode)

    # Use unified backend dispatcher for remote execution

    [result_dict] = run_jobs(cfg_payloads=cfg_payloads, opts=opts, task_fn=run_experiment_task)

    # Adapt to summary printer shape for all backends
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
