"""Showcase command for generating README images."""

from llc.util.config_overrides import apply_preset_then_overrides
from llc.config import CFG
from llc.util.backend_dispatch import BackendOptions, prepare_payloads, run_jobs
from llc.tasks import run_experiment_task
from llc.commands.analyze_cmd import analyze_entry
from llc.commands.promote_cmd import promote_readme_images_entry
from dataclasses import replace
import logging

logger = logging.getLogger(__name__)


def showcase_readme_entry(**kwargs):
    """
    Run a 'full' preset job on the chosen backend/GPU, then analyze & promote README images.
    Inherits backend flags from run_shared_options (backend, gpu-mode, slurm, modal, etc).
    """
    # Extract backend-related kwargs
    backend = (kwargs.pop("backend", None) or "local").lower()
    gpu_mode = kwargs.pop("gpu_mode", "off")
    gpu_types = kwargs.pop("gpu_types", "")
    sampler_cli = kwargs.pop("sampler", None)  # CLI override for single sampler

    # Submitit-specific parameters
    slurm_partition = kwargs.pop("slurm_partition", None)
    slurm_account = kwargs.pop("slurm_account", None)
    timeout_min = kwargs.pop("timeout_min", 180)
    cpus = kwargs.pop("cpus", 4)
    mem_gb = kwargs.pop("mem_gb", 16)
    slurm_signal_delay_s = kwargs.pop("slurm_signal_delay_s", 120)

    # 1) Build config (full preset; ensure plots are saved)
    cfg = apply_preset_then_overrides(CFG, "full", {"save_plots": True, "show_plots": False})

    # If the user passed --sampler, honor it and make the config atomic up front
    if sampler_cli:
        cfg = replace(cfg, samplers=(sampler_cli,))

    # Map GPU mode to batching configuration
    if gpu_mode == "vectorized":
        cfg = replace(cfg, use_batched_chains=True)
    elif gpu_mode == "sequential":
        cfg = replace(cfg, use_batched_chains=False)

    # 2) Prepare payloads with automatic fan-out for multi-sampler configs
    save_artifacts = True
    skip_if_exists = False
    payloads = prepare_payloads(
        [cfg],
        save_artifacts=save_artifacts,
        skip_if_exists=skip_if_exists,
        gpu_mode=gpu_mode,
        explode_samplers=True,  # Fan-out happens centrally in prepare_payloads
    )

    # 3) Run via the unified dispatcher (local / submitit / modal)
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

    logger.info(f"Running showcase on {backend} backend with gpu_mode={gpu_mode}")

    # For local backend, set JAX platform
    if backend == "local":
        from llc.util.backend_bootstrap import select_jax_platform
        select_jax_platform(gpu_mode)

    results = run_jobs(cfg_payloads=payloads, opts=opts, task_fn=run_experiment_task)

    # 4) Check for errors in any result
    for i, res in enumerate(results):
        if res.get("status") == "error":
            raise SystemExit(f"showcase: job {i+1}/{len(results)} failed with error: {res.get('error', 'unknown')}")

    # 5) Analyze & promote images from all runs
    logger.info(f"All {len(results)} jobs completed successfully")

    for i, res in enumerate(results):
        run_dir = res.get("run_dir", "")
        if not run_dir:
            logger.warning(f"Job {i+1}: no run_dir returned; skipping analysis")
            continue

        sampler = res.get("meta", {}).get("sampler", "unknown")
        logger.info(f"Analyzing {sampler} run: {run_dir}")

        # 6) Analyze each run
        analyze_entry(run_dir, which="all",
                      plots="running_llc,rank,ess_evolution,autocorr,energy,theta",
                      out=None, overwrite=True)

    # 7) Promote images from the first successful run (maintain backward compatibility)
    first_run_dir = next((res.get("run_dir") for res in results if res.get("run_dir")), None)
    if first_run_dir:
        promote_readme_images_entry(first_run_dir)
        logger.info(f"README assets refreshed from {first_run_dir}")
    else:
        logger.warning("No successful runs found for README image promotion")