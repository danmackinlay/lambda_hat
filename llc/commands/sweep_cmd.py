"""Sweep command implementation."""

import json
import os

from llc.config import CFG, config_schema_hash
from llc.util.config_overrides import apply_preset_then_overrides
from llc.util.modal_utils import extract_modal_artifacts_locally


def sweep_entry(kwargs: dict) -> None:
    """Entry point for sweep command."""
    backend = (kwargs.pop("backend") or "local").lower()
    workers = kwargs.pop("workers", 0)
    n_seeds = kwargs.pop("n_seeds", 2)
    save_artifacts = not kwargs.pop("no_artifacts", False)
    skip_if_exists = kwargs.pop("skip_if_exists", True)
    preset = kwargs.pop("preset", None)

    # Build base config for sweep
    base_cfg = apply_preset_then_overrides(CFG, preset, kwargs)

    # Build worklist
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
        from modal_app import app, run_experiment_remote

        remote_fn = run_experiment_remote
        modal_opts = {"max_containers": 1, "min_containers": 0, "buffer_containers": 0}
    else:
        modal_opts = None

    # Executor
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
        with app.run():
            results = _run_map()
    else:
        results = _run_map()

    # For Modal, optionally pull artifacts locally (convenience)
    if backend == "modal":
        os.makedirs("artifacts", exist_ok=True)
        for r in results:
            try:
                extract_modal_artifacts_locally(r)
            except Exception as e:
                print(f"Warning: failed to extract artifacts for a job: {e}")

    # Save long-form CSV with WNV fields (same as argparse version)
    _save_sweep_results(results)


def _save_sweep_results(results):
    """Save sweep results to CSV."""
    rows = []
    for r in results:
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

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv("llc_sweep_results.csv", index=False)
        print(
            "\nSweep complete! Results saved to llc_sweep_results.csv with WNV fields."
        )
        print(
            f"Successful runs: {len(rows)}/{len(results)} (rows include per-sampler results)"
        )
    else:
        print("\nNo successful runs to save.")