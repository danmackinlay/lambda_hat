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
    cuda_devices = kwargs.pop("cuda_devices", None)

    # Set JAX platform before any JAX imports
    if gpu_mode == "off":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ["JAX_PLATFORMS"] = "cuda"
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

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
    print(f"Running sweep with {len(items)} configurations on {backend} backend")
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

        for s in ("sgld", "hmc", "mclmc"):
            if f"{s}_llc_mean" not in M:
                continue
            rows.append(
                {
                    "sweep": "dim",
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
