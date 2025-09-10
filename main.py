# main.py
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")  # HMC benefits from float64
os.environ.setdefault("MPLBACKEND", "Agg")  # Headless rendering - no GUI windows

import argparse
from dataclasses import replace

import arviz as az
import jax
import jax.numpy as jnp
from jax import random, jit
from jax.flatten_util import ravel_pytree
import scipy.stats as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import new sampler adapters and utility modules
from llc.samplers.base import prepare_diag_targets
from llc.diagnostics import (
    llc_mean_and_se_from_histories,
    llc_ci_from_histories,
    plot_diagnostics,
)
from llc.artifacts import (
    create_run_directory,
    save_config,
    save_idata_L,
    save_idata_theta,
    save_metrics,
    create_manifest,
    generate_gallery_html,
    save_L0,
)
from llc.models import infer_widths, init_mlp_params
from llc.data import make_dataset
from llc.losses import as_dtype, make_loss_fns
from llc.posterior import (
    compute_beta_gamma,
    make_logpost_and_score,
    make_logdensity_for_mclmc,
)
from llc.runners import (
    RunStats,
    tic,
    toc,
    run_sgld_online,
    run_hmc_online_with_adaptation,
    run_mclmc_online,
)
from llc.config import Config, CFG
from llc.experiments import train_erm, sweep_space, build_sweep_worklist


plt.switch_backend("Agg")  # Ensure headless backend even if pyplot was already imported


def get_accept(info):
    """Robust accessor for HMC acceptance rate across BlackJAX versions

    BlackJAX >=1.2: HMCInfo.acceptance_rate (float)
    Some versions expose nested 'acceptance.rate'
    This function handles both patterns.
    """
    if hasattr(info, "acceptance_rate"):
        return float(info.acceptance_rate)
    acc = getattr(info, "acceptance", None)
    return float(getattr(acc, "rate", np.nan)) if acc is not None else np.nan


def work_normalized_variance(se, time_seconds: float, grad_work: int):
    """Compute WNV in both time and gradient units"""
    return dict(
        WNV_seconds=float(se**2 * max(1e-12, time_seconds)),
        WNV_grads=float(se**2 * max(1.0, grad_work)),
    )


def scalar_chain_diagnostics(series_per_chain, name="L"):
    """Compute ESS and R-hat for a scalar quantity across chains"""
    # Truncate to common length
    valid = [np.asarray(s) for s in series_per_chain if len(s) > 1]
    if not valid:
        return dict(ess=np.nan, rhat=np.nan)
    m = min(len(s) for s in valid)
    H = np.stack([s[:m] for s in valid], axis=0)  # (chains, m)
    idata = az.from_dict(posterior={name: H})
    ess_result = az.ess(idata, var_names=[name])
    rhat_result = az.rhat(idata, var_names=[name])
    ess = float(np.nanmedian(ess_result[name].values))
    rhat = float(np.nanmax(rhat_result[name].values))
    return dict(ess=ess, rhat=rhat)


# ----------------------------
# CLI Argument Parsing
# ----------------------------
def parse_args():
    """Parse command line arguments to override Config defaults"""
    parser = argparse.ArgumentParser(description="Local Learning Coefficient Analysis")

    # Sampler selection
    parser.add_argument(
        "--samplers",
        type=str,
        default=None,
        help="Comma-separated list of samplers (sgld,hmc,mclmc)",
    )

    # Data parameters
    parser.add_argument(
        "--n-data", type=int, default=None, help="Number of data points"
    )

    # Sampling parameters
    parser.add_argument(
        "--chains", type=int, default=None, help="Number of chains to run"
    )

    # SGLD parameters
    parser.add_argument(
        "--sgld-steps", type=int, default=None, help="Number of SGLD steps"
    )
    parser.add_argument(
        "--sgld-warmup", type=int, default=None, help="SGLD warmup steps"
    )
    parser.add_argument(
        "--sgld-step-size", type=float, default=None, help="SGLD step size"
    )

    # HMC parameters
    parser.add_argument(
        "--hmc-draws", type=int, default=None, help="Number of HMC draws"
    )
    parser.add_argument("--hmc-warmup", type=int, default=None, help="HMC warmup steps")
    parser.add_argument(
        "--hmc-steps", type=int, default=None, help="HMC integration steps"
    )

    # MCLMC parameters
    parser.add_argument(
        "--mclmc-draws", type=int, default=None, help="Number of MCLMC draws"
    )

    # Output control
    parser.add_argument(
        "--save-plots", action="store_true", default=None, help="Save diagnostic plots"
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        default=None,
        help="Don't save diagnostic plots",
    )

    # Presets
    parser.add_argument(
        "--preset",
        choices=["quick", "full"],
        default=None,
        help="Use quick or full preset",
    )

    # Model parameters
    parser.add_argument(
        "--target-params",
        type=int,
        default=None,
        help="Target parameter count for model",
    )

    return parser.parse_args()


def apply_preset(cfg: Config, preset: str) -> Config:
    """Apply quick or full preset configurations"""
    if preset == "quick":
        # Quick preset: fewer steps, larger eval_every, more thinning
        return replace(
            cfg,
            sgld_steps=1000,
            sgld_warmup=200,
            sgld_eval_every=20,
            sgld_thin=10,
            hmc_draws=200,
            hmc_warmup=200,
            hmc_eval_every=5,
            hmc_thin=5,
            mclmc_draws=500,
            mclmc_eval_every=10,
            mclmc_thin=5,
            progress_update_every=100,
        )
    elif preset == "full":
        # Full preset: current defaults (no changes)
        return cfg
    else:
        return cfg


def override_config(cfg: Config, args) -> Config:
    """Override config with command line arguments"""
    overrides = {}

    # Handle samplers list
    if args.samplers:
        samplers = [s.strip() for s in args.samplers.split(",")]
        # For now, just set the primary sampler to the first one
        if samplers:
            overrides["sampler"] = samplers[0]

    # Simple parameter overrides
    if args.n_data is not None:
        overrides["n_data"] = args.n_data
    if args.chains is not None:
        overrides["chains"] = args.chains
    if args.sgld_steps is not None:
        overrides["sgld_steps"] = args.sgld_steps
    if args.sgld_warmup is not None:
        overrides["sgld_warmup"] = args.sgld_warmup
    if args.sgld_step_size is not None:
        overrides["sgld_step_size"] = args.sgld_step_size
    if args.hmc_draws is not None:
        overrides["hmc_draws"] = args.hmc_draws
    if args.hmc_warmup is not None:
        overrides["hmc_warmup"] = args.hmc_warmup
    if args.hmc_steps is not None:
        overrides["hmc_num_integration_steps"] = args.hmc_steps
    if args.mclmc_draws is not None:
        overrides["mclmc_draws"] = args.mclmc_draws
    if args.target_params is not None:
        overrides["target_params"] = args.target_params

    # Handle save plots
    if args.save_plots:
        overrides["save_plots"] = True
    elif args.no_save_plots:
        overrides["save_plots"] = False

    return replace(cfg, **overrides)


# ----------------------------
# Main
# ----------------------------
def main(cfg: Config = CFG):
    """Main experiment runner - now delegates to the unified pipeline"""
    from llc.pipeline import run_one
    
    # Run the unified pipeline  
    outputs = run_one(cfg, save_artifacts=True, skip_if_exists=False)
    
    # Print summary metrics for backwards compatibility
    print("\n=== Final Results ===")
    for sampler in ("sgld", "hmc", "mclmc"):
        if f"{sampler}_llc_mean" in outputs.metrics:
            llc = outputs.metrics[f"{sampler}_llc_mean"]
            se = outputs.metrics[f"{sampler}_llc_se"]
            ess = outputs.metrics[f"{sampler}_ess"]
            print(f"{sampler.upper()}: λ̂={llc:.4f} ± {se:.4f} (ESS: {ess:.1f})")
    
    return outputs.run_dir


# ----------------------------
# Experiment runner for parameter sweeps
# ----------------------------


if __name__ == "__main__":
    import sys
    import argparse
    import pandas as pd

    # Create main parser
    parser = argparse.ArgumentParser(description="Local Learning Coefficient Analysis")
    sub = parser.add_subparsers(dest="cmd")

    # Single run (default) - inherit from existing CLI
    single_parser = sub.add_parser(
        "run", help="Run single experiment (default)", add_help=False
    )

    # Sweep mode with parallel backends
    sweep_parser = sub.add_parser(
        "sweep", help="Run parameter sweep (optionally parallel)"
    )
    sweep_parser.add_argument(
        "--backend", choices=["local", "submitit", "modal"], default="local"
    )
    sweep_parser.add_argument(
        "--workers", type=int, default=0, help="Local workers (0/1=serial)"
    )
    sweep_parser.add_argument("--n-seeds", type=int, default=2)

    # submitit params
    sweep_parser.add_argument("--partition", type=str, default=None)
    sweep_parser.add_argument("--timeout-min", type=int, default=60)
    sweep_parser.add_argument("--gpus", type=int, default=0)
    sweep_parser.add_argument("--cpus", type=int, default=4)
    sweep_parser.add_argument("--mem-gb", type=int, default=16)
    sweep_parser.add_argument("--account", type=str, default=None)
    sweep_parser.add_argument("--qos", type=str, default=None)
    sweep_parser.add_argument("--constraint", type=str, default=None)

    # timeout and artifact control
    sweep_parser.add_argument(
        "--timeout-s", type=int, default=None, help="Local executor timeout in seconds"
    )
    # Note: Modal timeouts are set in modal_app.py decorator, not at runtime
    sweep_parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save full artifacts (plots, data, HTML)",
    )
    sweep_parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Base artifacts directory",
    )

    # If no subcommand, default to single run behavior
    if len(sys.argv) == 1 or (
        len(sys.argv) > 1 and sys.argv[1] not in ["sweep", "run"]
    ):
        # Parse command line arguments and run main (existing behavior)
        args = parse_args()
        cfg = CFG  # Start with default config

        # Apply preset if specified
        if args.preset:
            cfg = apply_preset(cfg, args.preset)

        # Apply command line overrides
        cfg = override_config(cfg, args)

        # Run main with configured settings
        main(cfg)
    else:
        args, unknown = parser.parse_known_args()

        if args.cmd == "sweep":
            from llc.tasks import run_experiment_task
            from llc.execution import get_executor

            sw = sweep_space()
            work = build_sweep_worklist(sw, n_seeds=args.n_seeds)

            # Transform to the minimal payload the executors expect
            items = []
            for name, param, val, seed, cfg in work:
                cfg_dict = cfg.__dict__.copy()
                # Add artifact configuration
                if args.save_artifacts:
                    cfg_dict["save_artifacts"] = True
                    cfg_dict["artifacts_dir"] = args.artifacts_dir

                items.append(
                    {
                        "cfg": cfg_dict,
                        "tag": {
                            "sweep": name,
                            "param": param,
                            "value": val,
                            "seed": seed,
                        },
                    }
                )

            # Pick executor and run
            if args.backend == "local":
                ex = get_executor(
                    "local", workers=args.workers, timeout_s=args.timeout_s
                )
                results = ex.map(lambda it: run_experiment_task(it["cfg"]), items)
            elif args.backend == "submitit":
                slurm_additional = {}
                if args.account:
                    slurm_additional["account"] = args.account
                if args.qos:
                    slurm_additional["qos"] = args.qos
                if args.constraint:
                    slurm_additional["constraint"] = args.constraint

                ex = get_executor(
                    "submitit",
                    folder="slurm_logs",
                    timeout_min=args.timeout_min,
                    slurm_partition=args.partition,
                    gpus_per_node=args.gpus,
                    cpus_per_task=args.cpus,
                    mem_gb=args.mem_gb,
                    name="llc",
                    slurm_additional_parameters=slurm_additional
                    if slurm_additional
                    else None,
                )
                results = ex.map(lambda it: run_experiment_task(it["cfg"]), items)
            elif args.backend == "modal":
                try:
                    import modal

                    # Look up the deployed function by app & function name
                    f = modal.Function.from_name(
                        "llc-experiments", "run_experiment_remote"
                    )
                    ex = get_executor(
                        "modal", remote_fn=f
                    )  # no runtime options; decorator controls resources
                except ImportError:
                    raise RuntimeError(
                        "Modal is not installed. Install with: uv sync --extra modal"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to look up deployed Modal function. "
                        f"Make sure you've deployed with 'modal deploy modal_app.py'. Error: {e}"
                    )
                # pass only the cfg dict (modal function signature matches)
                results = ex.map(None, [it["cfg"] for it in items])
            else:
                raise SystemExit(f"Unknown backend: {args.backend}")

            # Fold results into a DataFrame
            rows = []
            for it, r in zip(items, results):
                rows.append(
                    {
                        "sweep": it["tag"]["sweep"],
                        "param": it["tag"]["param"],
                        "value": it["tag"]["value"],
                        "seed": it["tag"]["seed"],
                        "llc_sgld": r.get("llc_sgld"),
                        "llc_hmc": r.get("llc_hmc"),
                    }
                )

            df = pd.DataFrame(rows)
            df.to_csv("llc_sweep_results.csv", index=False)
            print("\n=== Sweep Results ===")
            print(
                df.groupby(["sweep", "param", "value"]).agg(
                    {
                        "llc_sgld": ["mean", "std"],
                        "llc_hmc": ["mean", "std"],
                        "seed": "count",
                    }
                )
            )
            print("\nResults saved to llc_sweep_results.csv")
        else:
            # Single run mode - use existing behavior
            args = parse_args()
            cfg = CFG

            if args.preset:
                cfg = apply_preset(cfg, args.preset)

            cfg = override_config(cfg, args)
            main(cfg)
