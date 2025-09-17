# llc/cli_click.py
"""
Click-based CLI for LLC:
- Subcommands: run, sweep
- Modal uses object-based imports (auto-deploys current code).
- Threads config_schema hash and skip_if_exists through to remote worker.
- Clean flag-only configuration (no environment variable overrides).
"""

from __future__ import annotations
import os
import logging
import io
import tarfile
import json
from dataclasses import replace
from typing import Optional

import click

from llc.config import CFG, Config, config_schema_hash
from llc.presets import apply_preset
from llc.pipeline import run_one


# ---------- Helpers ----------


def _override_config(cfg: Config, args: dict) -> Config:
    """Apply command-line overrides to configuration (mirror argparse override_config)."""
    overrides = {}

    # Simple direct mappings
    direct_mappings = [
        "n_data",
        "seed",
        "loss",
        "depth",
        "chains",
        "sgld_steps",
        "sgld_warmup",
        "sgld_step_size",
        "sgld_batch_size",
        "sgld_eval_every",
        "sgld_thin",
        "hmc_draws",
        "hmc_warmup",
        "hmc_eval_every",
        "hmc_thin",
        "mclmc_draws",
        "mclmc_eval_every",
        "mclmc_thin",
    ]
    for attr in direct_mappings:
        if args.get(attr) is not None:
            overrides[attr] = args[attr]

    # SGLD preconditioning
    if args.get("sgld_precond") is not None:
        overrides["sgld_precond"] = args["sgld_precond"]
    if args.get("sgld_beta1") is not None:
        overrides["sgld_beta1"] = args["sgld_beta1"]
    if args.get("sgld_beta2") is not None:
        overrides["sgld_beta2"] = args["sgld_beta2"]
    if args.get("sgld_eps") is not None:
        overrides["sgld_eps"] = args["sgld_eps"]
    if args.get("sgld_bias_correction") is not None:
        overrides["sgld_bias_correction"] = args["sgld_bias_correction"]

    # Batched chains
    if args.get("use_batched_chains") is not None:
        overrides["use_batched_chains"] = args["use_batched_chains"]

    # width -> widths
    if args.get("width") is not None:
        overrides["widths"] = [args["width"]] * cfg.depth

    # target_params -> infer widths
    if args.get("target_params") is not None:
        from llc.models import infer_widths

        overrides["widths"] = infer_widths(
            cfg.in_dim, cfg.out_dim, cfg.depth, args["target_params"]
        )

    # Target selection
    if args.get("target") is not None:
        overrides["target"] = args["target"]
    if args.get("quad_dim") is not None:
        overrides["quad_dim"] = args["quad_dim"]

    # Plotting
    if args.get("save_plots") is not None:
        overrides["save_plots"] = args["save_plots"]

    return replace(cfg, **overrides) if overrides else cfg


def _extract_modal_artifacts_locally(result_dict: dict) -> None:
    """Download and extract artifact tarball to ./artifacts/<run_id> if present."""
    if result_dict.get("artifact_tar") and result_dict.get("run_id"):
        rid = result_dict["run_id"]
        dest_root = "artifacts"
        os.makedirs(dest_root, exist_ok=True)
        dest = os.path.join(dest_root, rid)

        # Clean existing
        if os.path.exists(dest):
            import shutil

            shutil.rmtree(dest)

        with tarfile.open(
            fileobj=io.BytesIO(result_dict["artifact_tar"]), mode="r:gz"
        ) as tf:
            tf.extractall(dest_root)
        print(f"Artifacts downloaded and extracted to: {dest}")
        result_dict["run_dir"] = dest


def _apply_preset_then_overrides(
    cfg: Config, preset: Optional[str], kwargs: dict
) -> Config:
    out = cfg
    if preset:
        out = apply_preset(out, preset)
    return _override_config(out, kwargs)


# ---------- Click CLI ----------


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool):
    """LLC command-line interface (Click)."""
    # env for plotting/headless at import time (match argparse CLI behavior)
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("MPLBACKEND", "Agg")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


# ----- Shared options (kept explicit for clarity) -----


def _run_shared_options(f):
    # Backend / basics
    f = click.option(
        "--backend",
        type=click.Choice(["local", "submitit", "modal"]),
        default="local",
        help="Execution backend.",
    )(f)
    f = click.option(
        "--preset",
        type=click.Choice(["quick", "full"]),
        default=None,
        help="Apply preset configuration.",
    )(f)
    # Control flags
    f = click.option(
        "--skip/--no-skip",
        "skip_if_exists",
        default=True,
        help="Skip run if results already exist.",
    )(f)
    f = click.option(
        "--no-artifacts",
        is_flag=True,
        default=False,
        help="Don't save artifacts (plots, data files, gallery).",
    )(f)
    # Plotting
    f = click.option(
        "--save-plots/--no-save-plots",
        "save_plots",
        default=None,
        help="Enable/disable saving diagnostic plots.",
    )(f)

    # Data parameters
    f = click.option("--n-data", type=int, help="Number of data points")(f)
    f = click.option("--seed", type=int, help="Random seed")(f)
    f = click.option(
        "--loss", type=click.Choice(["mse", "t_regression"]), help="Loss function"
    )(f)

    # Model architecture
    f = click.option("--depth", type=int, help="Number of hidden layers")(f)
    f = click.option("--width", type=int, help="Hidden layer width")(f)
    f = click.option(
        "--target-params", "target_params", type=int, help="Target parameter count"
    )(f)

    # Target selection
    f = click.option(
        "--target", type=click.Choice(["mlp", "quadratic"]), help="Target model"
    )(f)
    f = click.option(
        "--quad-dim", type=int, help="Parameter dimension for quadratic target"
    )(f)

    # SGLD parameters
    f = click.option("--sgld-steps", type=int, help="SGLD total steps")(f)
    f = click.option("--sgld-warmup", type=int, help="SGLD warmup steps")(f)
    f = click.option("--sgld-step-size", type=float, help="SGLD step size")(f)
    f = click.option("--sgld-batch-size", type=int, help="SGLD minibatch size")(f)
    f = click.option("--sgld-eval-every", type=int, help="SGLD evaluation frequency")(f)
    f = click.option("--sgld-thin", type=int, help="SGLD thinning factor")(f)

    # SGLD preconditioning
    f = click.option(
        "--sgld-precond",
        type=click.Choice(["none", "rmsprop", "adam"]),
        default=None,
        help="Diagonal preconditioning for SGLD",
    )(f)
    f = click.option("--sgld-beta1", type=float, help="Adam beta1 (first moment)")(f)
    f = click.option(
        "--sgld-beta2", type=float, help="RMSProp/Adam beta2 (second moment)"
    )(f)
    f = click.option("--sgld-eps", type=float, help="Preconditioner epsilon")(f)
    f = click.option(
        "--sgld-bias-correction/--no-sgld-bias-correction",
        "sgld_bias_correction",
        default=None,
        help="Enable/disable Adam bias correction",
    )(f)

    # Plotting defaults already done above

    # HMC parameters
    f = click.option("--hmc-draws", type=int, help="HMC total draws")(f)
    f = click.option("--hmc-warmup", type=int, help="HMC warmup steps")(f)
    f = click.option("--hmc-eval-every", type=int, help="HMC evaluation frequency")(f)
    f = click.option("--hmc-thin", type=int, help="HMC thinning factor")(f)

    # MCLMC parameters
    f = click.option("--mclmc-draws", type=int, help="MCLMC total draws")(f)
    f = click.option("--mclmc-eval-every", type=int, help="MCLMC evaluation frequency")(
        f
    )
    f = click.option("--mclmc-thin", type=int, help="MCLMC thinning factor")(f)

    # Chain parameters
    f = click.option("--chains", type=int, help="Number of parallel chains")(f)
    f = click.option(
        "--use-batched-chains/--no-batched-chains",
        default=None,
        help="Use batched chain execution (vmap+scan) for speed",
    )(f)

    return f


def _sweep_shared_options(f):
    f = click.option(
        "--workers", type=int, default=0, help="Number of local workers (0/1=serial)"
    )(f)
    f = click.option(
        "--n-seeds",
        type=int,
        default=2,
        help="Number of random seeds per configuration",
    )(f)
    # Reuse run-shared too (includes backend)
    return _run_shared_options(f)


# ----- Commands -----


@cli.command()
@_run_shared_options
def run(**kwargs):
    """Run a single experiment (local / submitit / modal)."""
    save_artifacts = not kwargs.pop("no_artifacts", False)
    skip_if_exists = kwargs.pop("skip_if_exists", True)
    preset = kwargs.pop("preset", None)
    backend = (kwargs.pop("backend") or "local").lower()

    # Build config = preset + overrides
    cfg = _apply_preset_then_overrides(CFG, preset, kwargs)

    if backend == "local":
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

    from llc.execution import get_executor
    from llc.tasks import run_experiment_task

    if backend == "modal":
        # Ensure the Modal App is running so the function hydrates
        from modal_app import app, run_experiment_remote

        with app.run():
            executor = get_executor(backend="modal", remote_fn=run_experiment_remote)
            [result_dict] = executor.map(run_experiment_task, [cfg_dict])

        # Download artifacts locally (optional convenience)
        _extract_modal_artifacts_locally(result_dict)

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
        raise click.BadParameter(f"Unknown backend: {backend}")


def _print_summary_like_argparse(result):
    print("\n=== Final Results ===")
    for key, value in (result.metrics or {}).items():
        if "llc_mean" in key:
            sampler = key.replace("_llc_mean", "").upper()
            se_key = key.replace("_mean", "_se")
            se_value = (result.metrics or {}).get(se_key, 0)
            print(f"{sampler} LLC: {value:.4f} ± {float(se_value):.4f}")
    if getattr(result, "run_dir", ""):
        print(f"\nArtifacts saved to: {result.run_dir}")


@cli.command()
@_sweep_shared_options
def sweep(**kwargs):
    """Run a parameter sweep (local / submitit / modal)."""
    backend = (kwargs.pop("backend") or "local").lower()
    workers = kwargs.pop("workers", 0)
    n_seeds = kwargs.pop("n_seeds", 2)
    save_artifacts = not kwargs.pop("no_artifacts", False)
    skip_if_exists = kwargs.pop("skip_if_exists", True)
    preset = kwargs.pop("preset", None)

    # Build base config for sweep
    base_cfg = _apply_preset_then_overrides(CFG, preset, kwargs)

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
                _extract_modal_artifacts_locally(r)
            except Exception as e:
                print(f"Warning: failed to extract artifacts for a job: {e}")

    # Save long-form CSV with WNV fields (same as argparse version)
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
                    "wnv_grad": M.get(f"{s}_wnv_grad"),
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


if __name__ == "__main__":
    cli()
