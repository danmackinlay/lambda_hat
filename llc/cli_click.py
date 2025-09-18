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


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--which", type=click.Choice(["all", "sgld", "hmc", "mclmc"]), default="all", help="Which samplers to analyze")
@click.option("--plots", default="running_llc,rank,ess_evolution,ess_quantile,autocorr,energy,theta", help="Comma-separated figure types to generate")
@click.option("--out", type=click.Path(), default=None, help="Output directory (default: <run_dir>)")
@click.option("--overwrite", is_flag=True, help="Overwrite existing PNGs")
def analyze(run_dir, which, plots, out, overwrite):
    """Post-hoc analysis on saved .nc files"""
    import os
    from pathlib import Path
    from arviz import from_netcdf
    from llc.analysis import (
        llc_point_se, efficiency_metrics,
        fig_running_llc, fig_rank_llc, fig_ess_evolution, fig_ess_quantile, fig_autocorr_llc, fig_energy, fig_theta_trace
    )

    run_dir = Path(run_dir)
    out_dir = Path(out or run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    which = ["sgld", "hmc", "mclmc"] if which == "all" else [which]
    plots = [p.strip() for p in plots.split(",") if p.strip()]

    for s in which:
        nc = run_dir / f"{s}.nc"
        if not nc.exists():
            print(f"[analyze] skip {s}: {nc.name} not found")
            continue

        try:
            idata = from_netcdf(nc)
        except Exception as e:
            print(f"[analyze] failed to load {s}: {e}")
            continue

        # metrics
        m = llc_point_se(idata)
        print(f"[{s}] mean={m.get('llc_mean', float('nan')):.4g} se={m.get('llc_se', float('nan')):.3g} "
              f"ESS={m.get('ess_bulk', float('nan')):.1f} Rhat={m.get('rhat', float('nan')):.3f}")

        # figures
        for p in plots:
            try:
                if p == "running_llc":
                    # need L0, n, beta; derive from attrs if you stored them, else skip:
                    n = int(idata.attrs.get("n_data", 0) or idata.posterior["L"].shape[1])
                    beta = float(idata.attrs.get("beta", 1.0))
                    L0 = float(idata.attrs.get("L0", 0.0))
                    fig = fig_running_llc(idata, n, beta, L0, f"{s} Running LLC")
                    path = out_dir / f"{s}_running_llc.png"
                elif p == "rank":
                    fig = fig_rank_llc(idata); path = out_dir / f"{s}_llc_rank.png"
                elif p == "ess_evolution":
                    fig = fig_ess_evolution(idata); path = out_dir / f"{s}_llc_ess_evolution.png"
                elif p == "ess_quantile":
                    fig = fig_ess_quantile(idata); path = out_dir / f"{s}_llc_ess_quantile.png"
                elif p == "autocorr":
                    fig = fig_autocorr_llc(idata); path = out_dir / f"{s}_llc_autocorr.png"
                elif p == "energy":
                    fig = fig_energy(idata); path = out_dir / f"{s}_energy.png"
                elif p == "theta":
                    fig = fig_theta_trace(idata, dims=4); path = out_dir / f"{s}_theta_trace.png"
                else:
                    continue

                if path.exists() and not overwrite:
                    print(f"[analyze] exists: {path.name} (use --overwrite)")
                else:
                    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
                    print(f"[analyze] saved: {path.name}")

                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                print(f"[analyze] failed {s}:{p}: {e}")


@cli.command("promote-readme-images")
@click.argument("run_dir", required=False, type=click.Path(exists=True, file_okay=False))
@click.option("--root", type=click.Path(exists=True, file_okay=False), default=".", help="Repository root (default: .)")
def promote_readme_images_cmd(run_dir, root):
    """
    Copy a curated set of diagnostic PNGs from a run dir into assets/readme/.
    If RUN_DIR is omitted, pick the newest completed run under runs/.
    """
    from pathlib import Path
    import shutil
    import re
    from datetime import datetime
    import sys

    root = Path(root).resolve()
    assets = root / "assets" / "readme"
    assets.mkdir(parents=True, exist_ok=True)

    # --- selection identical to scripts/promote_readme_images.py ---
    SELECT = [
        ("sgld_running_llc.png", "sgld_llc_running.png"),
        ("hmc_running_llc.png", "hmc_llc_running.png"),
        ("mclmc_running_llc.png", "mclmc_llc_running.png"),
        ("hmc_acceptance.png", "hmc_acceptance.png"),
        ("hmc_energy.png", "hmc_energy.png"),
        ("hmc_llc_rank.png", "llc_rank.png"),
        ("hmc_llc_ess_evolution.png", "llc_ess_evolution.png"),
        ("hmc_Ln_centered.png", "Ln_centered.png"),
        ("mclmc_energy_hist.png", "mclmc_energy_hist.png"),
    ]

    # helpers (ported from scripts module)
    sys.path.insert(0, str(root))  # allow 'llc' imports from CLI context
    from llc.manifest import is_run_completed, get_run_start_time

    def _has_needed_artifacts(p: Path) -> bool:
        if (p / "metrics.json").exists():
            return True
        pngs = [q.name for q in p.glob("*.png")]
        return any(key in name for key, _ in SELECT for name in pngs)

    def _latest_from_artifacts(artifacts_dir: Path) -> Path:
        candidates = []
        for p in artifacts_dir.iterdir():
            if not p.is_dir():
                continue
            try:
                q = p.resolve()
            except Exception:
                q = p
            if not _has_needed_artifacts(q):
                continue
            ts = None
            if re.fullmatch(r"\d{8}-\d{6}", p.name):
                try:
                    ts = datetime.strptime(p.name, "%Y%m%d-%H%M%S").timestamp()
                except ValueError:
                    ts = None
            if ts is None:
                mtimes = [f.stat().st_mtime for f in q.glob("*.png")]
                if (q / "metrics.json").exists():
                    mtimes.append((q / "metrics.json").stat().st_mtime)
                ts = max(mtimes) if mtimes else q.stat().st_mtime
            candidates.append((ts, p))
        if not candidates:
            raise SystemExit("No runs with artifacts found under artifacts/")
        candidates.sort(key=lambda t: t[0])
        return candidates[-1][1]

    def latest_run_dir(repo_root: Path) -> Path:
        runs_dir = repo_root / "runs"
        if not runs_dir.exists():
            artifacts_dir = repo_root / "artifacts"
            if not artifacts_dir.exists():
                raise SystemExit("Neither runs/ nor artifacts/ directory found")
            return _latest_from_artifacts(artifacts_dir)
        candidates = []
        for rd in runs_dir.iterdir():
            if not rd.is_dir():
                continue
            if not is_run_completed(rd):
                continue
            start_time = get_run_start_time(rd) or rd.stat().st_mtime
            candidates.append((start_time, rd))
        if not candidates:
            raise SystemExit("No completed runs found in runs/")
        candidates.sort(key=lambda t: t[0])
        return candidates[-1][1]

    def find_first_match(run_dir: Path, key: str) -> Path | None:
        base = run_dir.resolve() if run_dir.exists() else run_dir
        exact = base / key
        if exact.exists():
            return exact
        for p in sorted(base.glob("*.png")):
            if key in p.name:
                return p
        return None

    run_dir = Path(run_dir).resolve() if run_dir else latest_run_dir(root)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    click.echo(f"Promoting images from: {run_dir}")
    copied = 0
    for key, outname in SELECT:
        src = find_first_match(run_dir, key)
        if not src:
            click.echo(f"  [skip] no match for '{key}'")
            continue
        dst = assets / outname
        shutil.copy2(src, dst)
        click.echo(f"  copied {src.name} -> {dst.relative_to(root)}")
        copied += 1

    if copied == 0:
        click.echo("No images copied. Did this run save plots? (save_plots=True) "
                   "Or are you running an old diagnostics set?")
    else:
        click.echo(f"Done. Copied {copied} images.")
        click.echo("Commit updated assets/readme/*.png and refresh README references if needed.")


@cli.command("pull-artifacts")
@click.argument("run_id", required=False)
@click.option("--target", default="artifacts", show_default=True,
              type=click.Path(file_okay=False), help="Local target root")
def pull_artifacts_cmd(run_id, target):
    """
    Pull artifacts from Modal using the deployed SDK functions.
    If RUN_ID is omitted, discover the latest on the server.
    """
    import io, tarfile
    from pathlib import Path
    import modal

    APP = "llc-experiments"
    FN_LIST = "list_artifacts"
    FN_EXPORT = "export_artifacts"

    list_fn = modal.Function.from_name(APP, FN_LIST)
    export_fn = modal.Function.from_name(APP, FN_EXPORT)

    if run_id:
        click.echo(f"[pull-sdk] Pulling specific run: {run_id}")
    else:
        click.echo("[pull-sdk] Discovering latest run on server...")
        paths = list_fn.remote("/artifacts")
        if not paths:
            raise SystemExit("No remote artifacts found.")
        run_id = Path(sorted(paths)[-1]).name
        click.echo(f"[pull-sdk] Latest on server: {run_id}")

    click.echo(f"[pull-sdk] Downloading and extracting {run_id}...")
    data = export_fn.remote(run_id)
    dest_root = Path(target)
    dest_root.mkdir(parents=True, exist_ok=True)

    target_dir = dest_root / run_id
    if target_dir.exists():
        import shutil
        shutil.rmtree(target_dir)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(dest_root)
    click.echo(f"[pull-sdk] Extracted into {dest_root / run_id}")


if __name__ == "__main__":
    cli()
