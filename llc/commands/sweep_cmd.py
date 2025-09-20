"""Sweep command implementation."""

import json
import logging
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


def sweep_entry(kwargs: dict) -> None:
    """Entry point for sweep command."""
    logger = logging.getLogger(__name__)
    backend = (kwargs.pop("backend") or "local").lower()
    workers = kwargs.pop("workers", 0)
    n_seeds = kwargs.pop("n_seeds", 2)
    save_artifacts = not kwargs.pop("no_artifacts", False)
    skip_if_exists = kwargs.pop("skip_if_exists", True)
    preset = kwargs.pop("preset", None)
    gpu_mode = kwargs.pop("gpu_mode", "off")
    gpu_types = kwargs.pop("gpu_types", "")
    split_samplers = kwargs.pop("split_samplers", True)
    study_path = kwargs.pop("study", None)
    sampler_grid_json = kwargs.pop("sampler_grid", None)
    problem_grid_json = kwargs.pop("problem_grid", None)

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
                raise SystemExit("`--study` requires PyYAML. Install: `uv add pyyaml` or `pip install pyyaml`.") from e
            with open(study_path) as f:
                study = yaml.safe_load(f)
            # base overrides (preset already applied)
            base_cfg = apply_preset_then_overrides(base_cfg, study.get("base", {}).get("preset"), study.get("base", {}))
            problems = [ProblemVariant(**p) for p in (study.get("problems") or [])]
            samplers = [SamplerVariant(**s) for s in (study.get("samplers") or [])]
            if "seeds" in study:
                seeds = list(study["seeds"])
        else:
            # JSON grids via CLI
            if problem_grid_json:
                problems = [ProblemVariant(**obj) for obj in json.loads(problem_grid_json)]
            else:
                problems = [ProblemVariant("default", {})]
            if sampler_grid_json:
                samplers = [SamplerVariant(**obj) for obj in json.loads(sampler_grid_json)]
            else:
                # default: all four samplers
                samplers = [SamplerVariant(name, {}) for name in ("sgld", "sghmc", "hmc", "mclmc")]

        items = list(expand_matrix(base_cfg, problems, samplers, seeds))
    else:
        # Legacy fallback (kept for convenience)
        from llc.experiments import build_sweep_worklist, sweep_space
        sw = sweep_space()
        sw["base"] = base_cfg
        # Emits tuples: (name, param, val, seed, cfg) — convert to matrix shape
        legacy = build_sweep_worklist(sw, n_seeds=n_seeds)
        for name, param, val, seed, cfg in legacy:
            if split_samplers:
                for s in cfg.samplers:
                    items.append((name, s, seed, replace(cfg, samplers=[s])))
            else:
                # Keep single job that still lists multiple samplers (legacy)
                # but prefer matrix style: submit one per sampler anyway
                for s in cfg.samplers:
                    items.append((name, s, seed, replace(cfg, samplers=[s])))

    logger.info(f"Running sweep with {len(items)} jobs on {backend} backend")
    logger.info("Each job = (problem, sampler, seed).")
    if backend == "local" and workers > 1:
        logger.info(f"Using {workers} parallel workers")

    # Modal handle (if needed)
    remote_fn = None
    if backend == "modal":
        from llc.modal_app import app, ping

        remote_fn = pick_modal_remote_fn(gpu_mode)

        # Scale concurrency automatically
        maxc = min(8, len(items))  # Cap at 8 containers for reasonable concurrency
        modal_opts = {
            "max_containers": maxc,
            "min_containers": 0,
            "buffer_containers": max(1, maxc // 2),
        }
    else:
        modal_opts = None

    # Import execution modules only when needed
    from llc.execution import get_executor
    from llc.tasks import run_experiment_task

    def _run_map():
        if backend == "submitit":
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
            ex = get_executor(backend="submitit", **submitit_kwargs)
        else:
            ex = get_executor(
                backend=backend,
                workers=workers if backend == "local" else None,
                remote_fn=remote_fn,
                options=modal_opts,
            )
        return ex.map(run_experiment_task, cfg_dicts)

    # Build cfg dicts with schema hash (one per job)
    cfg_dicts = []
    for _problem, _sampler, _seed, cfg in items:
        d = schema_stamp(cfg, save_artifacts, skip_if_exists, gpu_mode)
        d["problem_name"] = _problem  # Include problem name for CSV
        cfg_dicts.append(d)

    if backend == "modal":
        # Hydrate functions by running inside the app context
        # Chunk the work to avoid multi-hour heartbeats
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

            results = []
            chunk_size = 16  # Process in batches to avoid long heartbeats

            for i in range(0, len(cfg_dicts), chunk_size):
                batch = cfg_dicts[i:i+chunk_size]
                ex = get_executor(
                    backend=backend,
                    remote_fn=remote_fn,
                    options=modal_opts,
                )
                try:
                    batch_results = ex.map(run_experiment_task, batch)
                    results.extend(batch_results)
                except Exception as e:
                    # Emit a synthetic error row so llc_sweep_errors.csv captures it
                    results.append({
                        "status": "error",
                        "run_id": "N/A",
                        "stage": "scheduling",
                        "error_type": e.__class__.__name__,
                        "error": str(e)[:2000],
                        "duration_s": 0,
                    })
                    logger.error(f"[sweep] Scheduling failed for batch: {e}")
                logger.info(f"[sweep] Completed batch {i//chunk_size + 1}/{(len(cfg_dicts) + chunk_size - 1)//chunk_size}")
    else:
        results = _run_map()

    # For Modal, optionally pull runs locally (convenience)
    if backend == "modal":
        os.makedirs("runs", exist_ok=True)
        for r in results:
            # Check if run had an error
            if r.get("status") == "error":
                logger.warning(f"[sweep] Run {r.get('run_id', 'unknown')} failed: {r.get('error_type', 'Unknown')} at stage {r.get('stage', 'unknown')}")
                continue
            try:
                extract_modal_runs_locally(r)
            except Exception as e:
                logger.warning(f"failed to extract runs for a job: {e}")

    # Save long-form CSV with WNV fields (same as argparse version)
    _save_sweep_results(results)


def _save_sweep_results(results):
    """Save sweep results to CSV, including error ledger."""
    rows = []
    error_rows = []

    for r in results:
        # Handle errors separately
        if r.get("status") == "error":
            error_rows.append({
                "run_id": r.get("run_id", "unknown"),
                "status": "error",
                "stage": r.get("stage", "unknown"),
                "error_type": r.get("error_type", "Unknown"),
                "duration_s": r.get("duration_s", 0),
                "error": r.get("error", "")[:200],  # First 200 chars of error
            })
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
        logger.info(
            f"Successful runs: {len(rows)} rows from {len(results)} jobs"
        )
    else:
        logger.info("No successful runs to save.")

    # Save error ledger if any errors occurred
    if error_rows:
        import pandas as pd

        error_df = pd.DataFrame(error_rows)
        error_df.to_csv("llc_sweep_errors.csv", index=False)
        logger.info(f"Errors logged to llc_sweep_errors.csv: {len(error_rows)} failed jobs")
