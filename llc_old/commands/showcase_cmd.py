"""Showcase command for generating README images."""

from llc.util.config_overrides import apply_preset_then_overrides
from llc.config import CFG
from llc.util.backend_dispatch import BackendOptions, prepare_payloads, run_jobs
from llc.tasks import run_experiment_task
from llc.commands.analyze_cmd import analyze_entry
from llc.commands.promote_cmd import promote_readme_images_entry
from dataclasses import replace
import logging
import os

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

    # 4) Validate results and require run_dir on success
    verbose = logger.isEnabledFor(logging.DEBUG) or logging.getLogger().isEnabledFor(logging.DEBUG)
    good_runs = []  # list of (sampler, run_dir)
    for i, res in enumerate(results):
        # Treat "ok/no run_dir" as an error, but do NOT override real worker errors.
        if res.get("status") == "ok" and not res.get("run_dir"):
            res = {**res}
            res["status"] = "error"
            res.setdefault("error_type", "TaskContractError")
            res.setdefault("error", "status=ok but returned no run_dir")

        sampler = res.get("sampler") or (res.get("meta") or {}).get("sampler", "unknown")
        if res.get("status") == "error":
            # Collect details
            etype = res.get("error_type", "Error")
            msg = res.get("error", "unknown")
            tb = res.get("traceback", "")
            job = res.get("_job", {}) or {}
            rid = res.get("rid")
            stderr_path = job.get("stderr")
            stdout_path = job.get("stdout")
            jid = job.get("id")
            details = []
            details.append(f"showcase: job {i+1}/{len(results)} FAILED [{etype}] {msg}")
            details.append(f"  sampler={sampler} rid={rid}")
            if jid:
                details.append(f"  submitit job_id: {jid}")
            if stdout_path:
                details.append(f"  stdout: {stdout_path}")
            if stderr_path:
                details.append(f"  stderr: {stderr_path}")
                # Tail last ~200 lines if verbose and file exists
                if verbose and isinstance(stderr_path, str) and os.path.exists(stderr_path):
                    try:
                        with open(stderr_path, "rb") as fh:
                            tail = fh.read()[-20000:]  # ~20KB
                        details.append("  --- stderr tail ---")
                        details.append(tail.decode(errors="replace"))
                        details.append("  --- end stderr tail ---")
                    except Exception:
                        pass
            if verbose and tb:
                details.append("  --- traceback ---")
                details.append(tb)
                details.append("  --- end traceback ---")
            raise SystemExit("\n".join(details))

        # status ok: must have run_dir or it is a protocol error
        run_dir = res.get("run_dir", "")
        if not run_dir:
            # escalate as error with best available context
            job = res.get("_job", {}) or {}
            jid = job.get("id")
            raise SystemExit(
                f"showcase: job {i+1}/{len(results)} reported status=ok but returned no run_dir "
                f"(sampler={sampler}, submitit job_id={jid}). This indicates a task contract bug."
            )
        good_runs.append((sampler, run_dir))

    # 5) Analyze & promote images from all runs
    logger.info(f"All {len(good_runs)} jobs completed successfully")

    for sampler, run_dir in good_runs:
        logger.info(f"Analyzing {sampler} run: {run_dir}")

        # 6) Analyze each run - only generate running_llc plots for README
        analyze_entry(run_dir, which="all",
                      plots="running_llc",
                      out=None, overwrite=True)

    # 7) Promote images from ALL successful runs (sgld/sghmc/hmc/mclmc)
    promote_readme_images_entry(good_runs)
    logger.info("README assets refreshed from %s", ", ".join(rd for _, rd in good_runs))