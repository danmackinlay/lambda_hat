# llc/cli.py
"""
Unified CLI for Local Learning Coefficient Analysis.
Consolidates the duplicate CLI parsers from main.py into one clean interface.
"""

import os
import sys
import argparse
from dataclasses import replace

# Set environment variables early
os.environ.setdefault("JAX_ENABLE_X64", "true")  # HMC benefits from float64
os.environ.setdefault("MPLBACKEND", "Agg")  # Headless rendering - no GUI windows

import logging

from llc.config import Config, CFG
from llc.pipeline import run_one


def create_parser() -> argparse.ArgumentParser:
    """Create the unified argument parser"""
    parser = argparse.ArgumentParser(description="Local Learning Coefficient Analysis")
    subparsers = parser.add_subparsers(dest="cmd", required=False)

    # Single run subcommand (default behavior)
    run_parser = subparsers.add_parser(
        "run", help="Run single experiment (default if no subcommand)"
    )
    # Add backend support for single runs
    run_parser.add_argument(
        "--backend",
        choices=["local", "submitit", "modal"],
        default="local",
        help="Execution backend for this run (default: local)",
    )
    add_run_arguments(run_parser)

    # Sweep subcommand
    sweep_parser = subparsers.add_parser(
        "sweep", help="Run parameter sweep with parallel backends"
    )
    add_sweep_arguments(sweep_parser)
    add_run_arguments(sweep_parser)  # Sweep inherits run arguments

    # For backwards compatibility, add run arguments to main parser too
    add_run_arguments(parser)

    return parser


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for running experiments"""

    # Control flags
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        default=True,
        help="Skip run if results already exist (default: True)",
    )
    parser.add_argument(
        "--no-skip",
        dest="skip_if_exists",
        action="store_false",
        help="Always run, even if results exist",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Don't save artifacts (plots, data files, gallery)",
    )

    # Preset configurations
    parser.add_argument(
        "--preset",
        choices=["quick", "full"],
        help="Apply preset configuration (quick: fast smoke test, full: thorough)",
    )

    # Data parameters
    parser.add_argument("--n-data", type=int, help="Number of data points")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--loss", choices=["mse", "t_regression"], help="Loss function")

    # Model architecture
    parser.add_argument("--depth", type=int, help="Number of hidden layers")
    parser.add_argument("--width", type=int, help="Hidden layer width")
    parser.add_argument("--target-params", type=int, help="Target parameter count")

    # SGLD parameters
    parser.add_argument("--sgld-steps", type=int, help="SGLD total steps")
    parser.add_argument("--sgld-warmup", type=int, help="SGLD warmup steps")
    parser.add_argument("--sgld-step-size", type=float, help="SGLD step size")
    parser.add_argument("--sgld-batch-size", type=int, help="SGLD minibatch size")
    parser.add_argument("--sgld-eval-every", type=int, help="SGLD evaluation frequency")
    parser.add_argument("--sgld-thin", type=int, help="SGLD thinning factor")
    # SGLD preconditioning
    parser.add_argument(
        "--sgld-precond",
        choices=["none", "rmsprop", "adam"],
        help="Diagonal preconditioning for SGLD (default: none)",
        default="adam",
    )
    parser.add_argument("--sgld-beta1", type=float, help="Adam beta1 (first moment)")
    parser.add_argument(
        "--sgld-beta2", type=float, help="RMSProp/Adam beta2 (second moment)"
    )
    parser.add_argument("--sgld-eps", type=float, help="Preconditioner epsilon")
    parser.add_argument(
        "--sgld-bias-correction",
        dest="sgld_bias_correction",
        action="store_true",
        help="Enable Adam bias correction",
    )
    parser.add_argument(
        "--no-sgld-bias-correction",
        dest="sgld_bias_correction",
        action="store_false",
        help="Disable Adam bias correction",
    )
    parser.set_defaults(sgld_bias_correction=None)

    # HMC parameters
    parser.add_argument("--hmc-draws", type=int, help="HMC total draws")
    parser.add_argument("--hmc-warmup", type=int, help="HMC warmup steps")
    parser.add_argument("--hmc-eval-every", type=int, help="HMC evaluation frequency")
    parser.add_argument("--hmc-thin", type=int, help="HMC thinning factor")

    # MCLMC parameters
    parser.add_argument("--mclmc-draws", type=int, help="MCLMC total draws")
    parser.add_argument(
        "--mclmc-eval-every", type=int, help="MCLMC evaluation frequency"
    )
    parser.add_argument("--mclmc-thin", type=int, help="MCLMC thinning factor")

    # Chain parameters
    parser.add_argument("--chains", type=int, help="Number of parallel chains")
    parser.add_argument(
        "--use-batched-chains",
        action="store_true",
        help="Use batched chain execution (vmap+scan) for speed",
    )
    parser.add_argument(
        "--no-batched-chains",
        dest="use_batched_chains",
        action="store_false",
        help="Use sequential chain execution (default)",
    )
    parser.set_defaults(use_batched_chains=None)


def add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to sweep mode"""
    parser.add_argument(
        "--backend",
        choices=["local", "submitit", "modal"],
        default="local",
        help="Execution backend for parallel sweep",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of local workers (0/1=serial)"
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=2,
        help="Number of random seeds per configuration",
    )


def apply_preset(cfg: Config, preset: str) -> Config:
    """Apply preset configurations for quick testing or thorough analysis"""
    if preset == "quick":
        return replace(
            cfg,
            sgld_steps=1000,
            sgld_warmup=200,
            sgld_eval_every=100,
            sgld_thin=5,
            hmc_draws=200,
            hmc_warmup=100,
            hmc_eval_every=20,
            hmc_thin=2,
            mclmc_draws=400,
            mclmc_eval_every=40,
            mclmc_thin=2,
            chains=2,
            n_data=1000,
        )
    elif preset == "full":
        return replace(
            cfg,
            sgld_steps=10000,
            sgld_warmup=2000,
            sgld_eval_every=100,
            sgld_thin=10,
            hmc_draws=2000,
            hmc_warmup=1000,
            hmc_eval_every=20,
            hmc_thin=5,
            mclmc_draws=4000,
            mclmc_eval_every=40,
            mclmc_thin=5,
            chains=4,
            n_data=5000,
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")


def override_config(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply command line overrides to configuration"""
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
        value = getattr(args, attr.replace("-", "_"), None)
        if value is not None:
            overrides[attr] = value

    # Preconditioning mappings (separate so None keeps defaults)
    if getattr(args, "sgld_precond", None) is not None:
        overrides["sgld_precond"] = args.sgld_precond
    if getattr(args, "sgld_beta1", None) is not None:
        overrides["sgld_beta1"] = args.sgld_beta1
    if getattr(args, "sgld_beta2", None) is not None:
        overrides["sgld_beta2"] = args.sgld_beta2
    if getattr(args, "sgld_eps", None) is not None:
        overrides["sgld_eps"] = args.sgld_eps
    if getattr(args, "sgld_bias_correction", None) is not None:
        overrides["sgld_bias_correction"] = args.sgld_bias_correction

    # Batched chains option
    if getattr(args, "use_batched_chains", None) is not None:
        overrides["use_batched_chains"] = args.use_batched_chains

    # Special handling for width -> use as uniform width
    if hasattr(args, "width") and args.width is not None:
        overrides["widths"] = [args.width] * cfg.depth

    # Special handling for target_params -> infer widths
    if hasattr(args, "target_params") and args.target_params is not None:
        from llc.models import infer_widths

        overrides["widths"] = infer_widths(
            cfg.in_dim, cfg.out_dim, cfg.depth, args.target_params
        )

    return replace(cfg, **overrides) if overrides else cfg


def handle_run_command(args: argparse.Namespace) -> None:
    """Handle the run subcommand (local, submitit, or modal)"""
    cfg = CFG

    # Apply preset if specified
    if hasattr(args, "preset") and args.preset:
        cfg = apply_preset(cfg, args.preset)

    # Apply command line overrides
    cfg = override_config(cfg, args)

    save_artifacts = not getattr(args, "no_artifacts", False)
    skip_if_exists = getattr(args, "skip_if_exists", True)

    backend = (getattr(args, "backend", "local") or "local").lower()

    if backend == "local":
        # Original local path
        result = run_one(cfg, save_artifacts=save_artifacts, skip_if_exists=skip_if_exists)

    else:
        # Route through the same executor/task used by sweep
        from llc.execution import get_executor
        from llc.tasks import run_experiment_task

        cfg_dict = cfg.__dict__.copy()
        cfg_dict["save_artifacts"] = save_artifacts

        if backend == "modal":
            # Use deployed function by name (Modal 1.0 API)
            import modal
            remote_fn = modal.Function.from_name("llc-experiments", "run_experiment_remote")
            executor = get_executor(backend="modal", remote_fn=remote_fn)
            [result_dict] = executor.map(run_experiment_task, [cfg_dict])
        elif backend == "submitit":
            # Fire one job on the cluster and wait
            executor = get_executor(backend="submitit")
            [result_dict] = executor.map(run_experiment_task, [cfg_dict])
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Print a small summary and adapt shape to local 'result'
        result = type("RunOutputs", (), {})()
        result.run_dir = result_dict.get("run_dir", "")
        result.metrics = {}
        for s in ("sgld", "hmc", "mclmc"):
            if f"llc_{s}" in result_dict:
                result.metrics[f"{s}_llc_mean"] = float(result_dict[f"llc_{s}"])
        result.histories = {}
        result.L0 = 0.0

    # Print summary
    print("\n=== Final Results ===")
    for key, value in result.metrics.items():
        if "llc_mean" in key:
            sampler = key.replace("_llc_mean", "").upper()
            se_key = key.replace("_mean", "_se")
            se_value = result.metrics.get(se_key, 0)
            print(f"{sampler} LLC: {value:.4f} ± {se_value:.4f}")

    if result.run_dir:
        print(f"\nArtifacts saved to: {result.run_dir}")


def handle_sweep_command(args: argparse.Namespace) -> None:
    """Handle the sweep subcommand"""
    # Import sweep & execution utilities
    from llc.experiments import build_sweep_worklist, sweep_space
    from llc.execution import get_executor
    from llc.tasks import run_experiment_task  # used for local/submitit

    # For modal backend we need the remote function handle:
    remote_fn = None
    if (args.backend or "local").lower() == "modal":
        import modal

        # Look up the DEPLOYED function (do NOT import modal_app.run_experiment_remote)
        remote_fn = modal.Function.from_name("llc-experiments", "run_experiment_remote")

    # Build sweep configuration
    base_cfg = CFG
    if hasattr(args, "preset") and args.preset:
        base_cfg = apply_preset(base_cfg, args.preset)

    base_cfg = override_config(base_cfg, args)

    # build_sweep_worklist expects a dict returned by sweep_space():
    sweep_cfg = sweep_space()
    sweep_cfg["base"] = base_cfg  # use our preset/overridden base config
    items = build_sweep_worklist(sweep_cfg, n_seeds=args.n_seeds)

    print(f"Running sweep with {len(items)} configurations on {args.backend} backend")
    if args.backend == "local" and args.workers > 1:
        print(f"Using {args.workers} parallel workers")

    # Get executor and run
    modal_opts = None
    if args.backend == "modal":
        # Cap Modal parallelism for cost control during development
        modal_opts = {"max_containers": 1, "min_containers": 0, "buffer_containers": 0}

    executor = get_executor(
        backend=args.backend,
        workers=args.workers if args.backend == "local" else None,
        remote_fn=remote_fn,  # required for modal backend
        options=modal_opts,
    )

    # Normalize items -> list of cfg dicts the task/remote_fn expects
    cfg_dicts = []
    for _name, _param, _val, _seed, cfg in items:
        # dataclass -> dict; you can also use dataclasses.asdict(cfg)
        cfg_d = cfg.__dict__.copy()
        cfg_d["save_artifacts"] = not getattr(args, "no_artifacts", False)
        cfg_dicts.append(cfg_d)

    # Local/submitit: executor.map calls run_experiment_task(cfg_dict)
    # Modal: ModalExecutor.map ignores `fn` and calls remote_fn.map(cfg_dict)
    results = executor.map(run_experiment_task, cfg_dicts)

    # Save results summary
    import pandas as pd

    summary_data = []
    for result in results:
        if result and "error" not in result:
            row = result["cfg"].copy()
            for key in ["llc_sgld", "llc_hmc", "llc_mclmc"]:
                if key in result:
                    row[key] = result[key]
            summary_data.append(row)

    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv("llc_sweep_results.csv", index=False)
        print("\nSweep complete! Results saved to llc_sweep_results.csv")
        print(f"Successful runs: {len(summary_data)}/{len(results)}")
    else:
        print("\nNo successful runs to save.")


def main() -> None:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging based on verbosity
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Handle subcommands
    if args.cmd == "sweep":
        handle_sweep_command(args)
    elif args.cmd == "run" or args.cmd is None:
        # Default to run if no subcommand or explicit run
        handle_run_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
