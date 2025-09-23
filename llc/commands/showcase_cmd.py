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

    # Submitit-specific parameters
    slurm_partition = kwargs.pop("slurm_partition", None)
    slurm_account = kwargs.pop("slurm_account", None)
    timeout_min = kwargs.pop("timeout_min", 180)
    cpus = kwargs.pop("cpus", 4)
    mem_gb = kwargs.pop("mem_gb", 16)
    slurm_signal_delay_s = kwargs.pop("slurm_signal_delay_s", 120)

    # 1) Build config (full preset; ensure plots are saved)
    cfg = apply_preset_then_overrides(CFG, "full", {"save_plots": True, "show_plots": False})

    # Map GPU mode to batching configuration
    if gpu_mode == "vectorized":
        cfg = replace(cfg, use_batched_chains=True)
    elif gpu_mode == "sequential":
        cfg = replace(cfg, use_batched_chains=False)

    # 2) Prepare one job payload using the shared schema stamper
    save_artifacts = True
    skip_if_exists = False
    [payload] = prepare_payloads([cfg], save_artifacts=save_artifacts,
                                 skip_if_exists=skip_if_exists, gpu_mode=gpu_mode)

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

    [res] = run_jobs(cfg_payloads=[payload], opts=opts, task_fn=run_experiment_task)

    # 4) Check for errors
    if res.get("status") == "error":
        raise SystemExit(f"showcase: job failed with error: {res.get('error', 'unknown')}")

    # 5) After the run, artifacts are local (Modal path auto-extracted if enabled)
    run_dir = res.get("run_dir", "")
    if not run_dir:
        raise SystemExit("showcase: no run_dir returned; check backend logs")

    logger.info(f"Run completed, analyzing {run_dir}")

    # 6) Analyze & promote
    analyze_entry(run_dir, which="all",
                  plots="running_llc,rank,ess_evolution,autocorr,energy,theta",
                  out=None, overwrite=True)
    promote_readme_images_entry(run_dir)

    logger.info(f"README assets refreshed from {run_dir}")