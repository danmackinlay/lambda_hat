"""Sweep command implementation."""

import json
import logging
import os
from dataclasses import replace

from llc.config import CFG
from llc.util.config_overrides import apply_preset_then_overrides
from llc.util.backend_dispatch import BackendOptions, prepare_payloads, run_jobs

# Module-level logger
logger = logging.getLogger(__name__)


def sweep_entry(kwargs: dict) -> None:
    """Entry point for sweep command."""
    backend = (kwargs.pop("backend") or "local").lower()
    workers = kwargs.pop("workers", 0)
    n_seeds = kwargs.pop("n_seeds", 2)
    save_artifacts = not kwargs.pop("no_artifacts", False)
    skip_if_exists = kwargs.pop("skip_if_exists", True)
    preset = kwargs.pop("preset", None)
    gpu_mode = kwargs.pop("gpu_mode", "off")
    gpu_types = kwargs.pop("gpu_types", "")
    study_path = kwargs.pop("study", None)
    sampler_grid_json = kwargs.pop("sampler_grid", None)
    problem_grid_json = kwargs.pop("problem_grid", None)

    # Submitit-specific parameters
    slurm_partition = kwargs.pop("slurm_partition", None)
    slurm_account = kwargs.pop("slurm_account", None)
    timeout_min = kwargs.pop("timeout_min", 119)
    cpus = kwargs.pop("cpus", 4)
    mem_gb = kwargs.pop("mem_gb", 16)
    slurm_signal_delay_s = kwargs.pop("slurm_signal_delay_s", 120)

    # Build base config for sweep
    base_cfg = apply_preset_then_overrides(CFG, preset, kwargs)

    # Map GPU mode to batching configuration for base config
    if gpu_mode == "vectorized":
        base_cfg = replace(base_cfg, use_batched_chains=True)
    elif gpu_mode == "sequential":
        base_cfg = replace(base_cfg, use_batched_chains=False)

    # --- Build worklist: prefer study YAML / JSON grids; fallback to legacy sweep_space ---
    items = []
    if study_path or sampler_grid_json or problem_grid_json:
        from llc.experiments_matrix import ProblemVariant, SamplerVariant, expand_matrix
        import json

        problems = []
        samplers = []
        seeds = list(range(n_seeds))

        if study_path:
            try:
                import yaml  # requires PyYAML
            except Exception as e:
                raise SystemExit(
                    "`--study` requires PyYAML. Install: `uv add pyyaml` or `pip install pyyaml`."
                ) from e
            with open(study_path) as f:
                study = yaml.safe_load(f)
            # base overrides (preset already applied)
            base_cfg = apply_preset_then_overrides(
                base_cfg, study.get("base", {}).get("preset"), study.get("base", {})
            )
            problems = [ProblemVariant(**p) for p in (study.get("problems") or [])]
            samplers = [SamplerVariant(**s) for s in (study.get("samplers") or [])]
            if "seeds" in study:
                seeds = list(study["seeds"])
        else:
            # JSON grids via CLI
            if problem_grid_json:
                problems = [
                    ProblemVariant(**obj) for obj in json.loads(problem_grid_json)
                ]
            else:
                problems = [ProblemVariant("default", {})]
            if sampler_grid_json:
                samplers = [
                    SamplerVariant(**obj) for obj in json.loads(sampler_grid_json)
                ]
            else:
                # default: all four samplers
                samplers = [
                    SamplerVariant(name, {})
                    for name in ("sgld", "sghmc", "hmc", "mclmc")
                ]

        items = list(expand_matrix(base_cfg, problems, samplers, seeds))
    else:
        # Legacy fallback (kept for convenience)
        from llc.experiments import build_sweep_worklist, sweep_space

        sw = sweep_space()
        sw["base"] = base_cfg
        # Emits tuples: (name, param, val, seed, cfg) — convert to matrix shape
        legacy = build_sweep_worklist(sw, n_seeds=n_seeds)
        # Legacy fallback (kept for convenience) — always split per sampler
        for name, param, val, seed, cfg in legacy:
            for s in cfg.samplers:
                items.append((name, s, seed, replace(cfg, samplers=[s])))

    logger.info(f"Running sweep with {len(items)} jobs on {backend} backend")
    logger.info("Each job = (problem, sampler, seed).")
    if backend == "local" and workers > 1:
        logger.info(f"Using {workers} parallel workers")

    # Build cfg dicts with schema hash (one per job)
    cfg_dicts = []
    for _problem, _sampler, _seed, cfg in items:
        cfg_dicts.append(cfg)

    # Use unified backend dispatcher
    opts = BackendOptions(
        backend=backend,
        gpu_mode=gpu_mode,
        gpu_types=gpu_types,
        local_workers=workers,
        slurm_partition=slurm_partition,
        slurm_account=slurm_account,
        timeout_min=timeout_min,
        cpus=cpus,
        mem_gb=mem_gb,
        slurm_signal_delay_s=slurm_signal_delay_s,
        modal_chunk_size=16,  # Process in batches to avoid long heartbeats
        modal_autoscaler_cap=min(
            8, len(items)
        ),  # Cap at 8 containers for reasonable concurrency
    )

    cfg_payloads = prepare_payloads(
        cfg_dicts,
        save_artifacts=save_artifacts,
        skip_if_exists=skip_if_exists,
        gpu_mode=gpu_mode,
    )

    # Include problem names for CSV
    for i, (_problem, _sampler, _seed, cfg) in enumerate(items):
        cfg_payloads[i]["problem_name"] = _problem

    # Import task function only when needed
    from llc.tasks import run_experiment_task

    results = run_jobs(
        cfg_payloads=cfg_payloads, opts=opts, task_fn=run_experiment_task
    )

    # Save long-form CSV with WNV fields (same as argparse version)
    _save_sweep_results(results)


def _save_sweep_results(results):
    """Save sweep results to CSV, including error ledger."""
    rows = []
    error_rows = []

    for r in results:
        # Handle errors separately
        if r.get("status") == "error":
            error_row = {
                "run_id": r.get("run_id", "unknown"),
                "status": "error",
                "stage": r.get("stage", "unknown"),
                "error_type": r.get("error_type", "Unknown"),
                "duration_s": r.get("duration_s", 0),
                "error": r.get("error", "")[:200],  # First 200 chars of error
            }
            # Include Submitit log paths if available
            if "submitit_stdout_path" in r:
                error_row["submitit_stdout_path"] = r["submitit_stdout_path"]
            if "submitit_stderr_path" in r:
                error_row["submitit_stderr_path"] = r["submitit_stderr_path"]
            if "submitit_stderr_tail" in r:
                error_row["submitit_stderr_tail"] = r["submitit_stderr_tail"][:1000]
            error_rows.append(error_row)
            continue

        run_dir = r.get("run_dir")
        if not run_dir:
            continue

        metrics_path = os.path.join(run_dir, "metrics.json")
        config_path = os.path.join(run_dir, "config.json")
        try:
            with open(metrics_path) as f:
                M = json.load(f)
            with open(config_path) as f:
                C = json.load(f)
        except Exception:
            continue

        # Fallback if metrics didn't carry family_id (older runs):
        if "family_id" in M:
            family_id = M["family_id"]
        else:
            from llc.config import Config
            from llc.cache import run_family_id

            family_id = run_family_id(Config(**C))

        for s in ("sgld", "sghmc", "hmc", "mclmc"):
            if f"{s}_llc_mean" not in M:
                continue
            rows.append(
                {
                    "sweep": "matrix",
                    "problem": C.get("problem_name", ""),
                    "family_id": family_id,
                    "target_params": C.get("target_params"),
                    "depth": C.get("depth"),
                    "activation": C.get("activation"),
                    "sampler": s,
                    "seed": C.get("seed"),
                    "llc_mean": M.get(f"{s}_llc_mean"),
                    "llc_se": M.get(f"{s}_llc_se"),
                    "ess": M.get(f"{s}_ess"),
                    "t_sampling": M.get("timing_sampling")
                    or M.get(f"{s}_timing_sampling"),
                    "work_grad": (
                        M.get(f"{s}_n_leapfrog_grads") or M.get(f"{s}_n_steps") or 0
                    ),
                    "wnv_time": M.get(f"{s}_wnv_time"),
                    "wnv_fde": M.get(f"{s}_wnv_fde"),
                    "run_dir": run_dir,
                }
            )

    # Save main results CSV
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv("llc_sweep_results.csv", index=False)
        logger.info(
            "Sweep complete! Results saved to llc_sweep_results.csv with WNV fields."
        )
        logger.info(f"Successful runs: {len(rows)} rows from {len(results)} jobs")
    else:
        logger.info("No successful runs to save.")

    # Save error ledger if any errors occurred
    if error_rows:
        import pandas as pd

        error_df = pd.DataFrame(error_rows)
        error_df.to_csv("llc_sweep_errors.csv", index=False)
        logger.info(
            f"Errors logged to llc_sweep_errors.csv: {len(error_rows)} failed jobs"
        )

        # Surface Submitit logs for easier debugging
        for er in error_rows:
            if "submitit_stderr_path" in er:
                logger.info(f"\n[Submitit Error] Job {er.get('run_id', 'unknown')}")
                logger.info(f"  stderr log: {er['submitit_stderr_path']}")
                if "submitit_stdout_path" in er:
                    logger.info(f"  stdout log: {er['submitit_stdout_path']}")
                if "submitit_stderr_tail" in er and er["submitit_stderr_tail"].strip():
                    logger.info("  Last lines of stderr:")
                    for line in er["submitit_stderr_tail"].strip().split('\n')[-10:]:
                        logger.info(f"    {line}")

        # Hint about log locations
        if any("submitit_stderr_path" in er for er in error_rows):
            logger.info(
                "\nHint: Submitit logs are stored in the 'slurm_logs/' folder by default. "
                "Check the paths above or llc_sweep_errors.csv for full details."
            )
