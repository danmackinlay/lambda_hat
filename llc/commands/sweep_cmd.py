"""Sweep command implementation."""

import json
import os
from dataclasses import replace

from llc.config import CFG, config_schema_hash
from llc.util.config_overrides import apply_preset_then_overrides
from llc.util.modal_utils import extract_modal_runs_locally


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
    split_samplers = kwargs.pop("split_samplers", False)

    # Submitit-specific parameters
    slurm_partition = kwargs.pop("slurm_partition", None)
    timeout_min = kwargs.pop("timeout_min", 180)
    cpus = kwargs.pop("cpus", 4)
    mem_gb = kwargs.pop("mem_gb", 16)
    slurm_signal_delay_s = kwargs.pop("slurm_signal_delay_s", 120)

    # Set JAX platform for local backend only (remote decides from decorator)
    if backend == "local":
        os.environ["JAX_PLATFORMS"] = "cuda" if gpu_mode != "off" else "cpu"
    # Validate and set GPU type list for modal_app decorators (evaluated at import time)
    if gpu_types:
        valid_gpus = {"A100", "H100", "L40S", "T4", "A10G"}
        requested = [t.strip() for t in gpu_types.split(",") if t.strip()]
        bad = [t for t in requested if t not in valid_gpus]
        if bad:
            print(f"[warn] unknown GPU types {bad}; falling back to L40S")
            os.environ["LLC_MODAL_GPU_LIST"] = "L40S"
        else:
            os.environ["LLC_MODAL_GPU_LIST"] = ",".join(requested)

    # Build base config for sweep
    base_cfg = apply_preset_then_overrides(CFG, preset, kwargs)

    # Map GPU mode to batching configuration for base config
    if gpu_mode == "vectorized":
        base_cfg = replace(base_cfg, use_batched_chains=True)
    elif gpu_mode == "sequential":
        base_cfg = replace(base_cfg, use_batched_chains=False)

    # Build worklist - import only when needed
    from llc.experiments import build_sweep_worklist, sweep_space

    sw = sweep_space()
    sw["base"] = base_cfg
    items = build_sweep_worklist(sw, n_seeds=n_seeds)

    # Expand into one job per sampler if requested
    if split_samplers:
        expanded = []
        for name, param, val, seed, cfg in items:
            for s in cfg.samplers:
                expanded.append((name, param, val, seed, replace(cfg, samplers=[s])))
        items = expanded

    print(f"Running sweep with {len(items)} configurations on {backend} backend")
    if split_samplers:
        print(f"Split samplers enabled - each job runs one sampler")
    if backend == "local" and workers > 1:
        print(f"Using {workers} parallel workers")

    # Modal handle (if needed)
    remote_fn = None
    if backend == "modal":
        # Choose CPU or GPU function based on --gpu-mode (like llc run)
        if gpu_mode == "off":
            from llc.modal_app import app, run_experiment_remote
            remote_fn = run_experiment_remote
        else:
            from llc.modal_app import app, run_experiment_remote_gpu
            remote_fn = run_experiment_remote_gpu

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

    # Build cfg dicts with schema hash
    cfg_dicts = []
    schema = config_schema_hash()
    for _name, _param, _val, _seed, cfg in items:
        d = cfg.__dict__.copy()
        d["save_artifacts"] = save_artifacts
        d["skip_if_exists"] = skip_if_exists
        d["config_schema"] = schema
        d["gpu_mode"] = gpu_mode  # Pass GPU mode to remote workers
        cfg_dicts.append(d)

    if backend == "modal":
        # Hydrate functions by running inside the app context
        # Chunk the work to avoid multi-hour heartbeats
        with app.run():
            results = []
            chunk_size = 16  # Process in batches to avoid long heartbeats

            for i in range(0, len(cfg_dicts), chunk_size):
                batch = cfg_dicts[i:i+chunk_size]
                ex = get_executor(
                    backend=backend,
                    remote_fn=remote_fn,
                    options=modal_opts,
                )
                batch_results = ex.map(run_experiment_task, batch)
                results.extend(batch_results)
                print(f"[sweep] Completed batch {i//chunk_size + 1}/{(len(cfg_dicts) + chunk_size - 1)//chunk_size}")
    else:
        results = _run_map()

    # For Modal, optionally pull runs locally (convenience)
    if backend == "modal":
        os.makedirs("runs", exist_ok=True)
        for r in results:
            # Check if run had an error
            if r.get("status") == "error":
                print(f"[sweep] Run {r.get('run_id', 'unknown')} failed: {r.get('error_type', 'Unknown')} at stage {r.get('stage', 'unknown')}")
                continue
            try:
                extract_modal_runs_locally(r)
            except Exception as e:
                print(f"Warning: failed to extract runs for a job: {e}")

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

        for s in ("sgld", "hmc", "mclmc"):
            if f"{s}_llc_mean" not in M:
                continue
            rows.append(
                {
                    "sweep": "dim",
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
        print(
            "\nSweep complete! Results saved to llc_sweep_results.csv with WNV fields."
        )
        print(
            f"Successful runs: {len(rows)} rows from {len(results)} jobs"
        )
    else:
        print("\nNo successful runs to save.")

    # Save error ledger if any errors occurred
    if error_rows:
        import pandas as pd

        error_df = pd.DataFrame(error_rows)
        error_df.to_csv("llc_sweep_errors.csv", index=False)
        print(f"\nErrors logged to llc_sweep_errors.csv: {len(error_rows)} failed jobs")
