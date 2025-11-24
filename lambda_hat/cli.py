#!/usr/bin/env python3
"""Lambda-Hat unified CLI - all commands under one interface."""

import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

import click


@click.group()
def cli():
    """Lambda-Hat: Neural network Bayesian inference toolkit.

    Unified CLI for building targets, running samplers, managing artifacts,
    promoting results, and orchestrating Parsl workflows.
    """


# =============================================================================
# Stage A: Build Targets
# =============================================================================


@cli.command()
@click.option(
    "--config-yaml",
    required=True,
    type=click.Path(exists=True),
    help="Path to composed YAML config",
)
@click.option("--target-id", required=True, help="Target ID string (e.g., tgt_abc123)")
@click.option(
    "--experiment",
    default=None,
    help="Experiment name (defaults from config then env)",
)
def build(config_yaml, target_id, experiment):
    """Build target artifact (Stage A: train neural network)."""

    from lambda_hat.commands.build_cmd import build_entry

    # Enable target diagnostics for manual builds (dev visibility)
    os.environ.setdefault("LAMBDA_HAT_SKIP_DIAGNOSTICS", "0")

    result = build_entry(config_yaml, target_id, experiment)
    click.echo(f"✓ Built {result['target_id']}: {result['urn']}")


# =============================================================================
# Stage B: Sample from Targets
# =============================================================================


@cli.command()
@click.option(
    "--config-yaml",
    required=True,
    type=click.Path(exists=True),
    help="Path to composed YAML config",
)
@click.option("--target-id", required=True, help="Target ID to sample from")
@click.option(
    "--experiment",
    default=None,
    help="Experiment name (defaults from config then env)",
)
def sample(config_yaml, target_id, experiment):
    """Run sampler on target (Stage B: MCMC/VI inference)."""
    from lambda_hat.commands.sample_cmd import sample_entry

    result = sample_entry(config_yaml, target_id, experiment)
    click.echo(f"✓ Completed run {result['run_id']}")


# =============================================================================
# Stage C: Generate Diagnostics
# =============================================================================


@cli.command()
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to run directory containing trace.nc or traces_raw.json",
)
@click.option(
    "--mode",
    type=click.Choice(["light", "full"]),
    default="light",
    show_default=True,
    help="Diagnostic depth: 'light' (basic plots) or 'full' (+ expensive plots)",
)
def diagnose(run_dir, mode):
    """Generate offline diagnostics for a completed sampling run (Stage C)."""
    from lambda_hat.commands.diagnose_cmd import diagnose_entry

    result = diagnose_entry(run_dir, mode)
    click.echo(f"✓ Generated {len(result['plots_generated'])} plots in {result['diagnostics_dir']}")


@cli.command()
@click.option(
    "--experiment",
    required=True,
    help="Experiment name (e.g., 'smoke', 'dev')",
)
@click.option(
    "--mode",
    type=click.Choice(["light", "full"]),
    default="light",
    show_default=True,
    help="Diagnostic depth: 'light' (basic plots) or 'full' (+ expensive plots)",
)
@click.option(
    "--samplers",
    default=None,
    help="Comma-separated sampler names to process (default: all)",
)
def diagnose_experiment(experiment, mode, samplers):
    """Generate diagnostics for all runs in an experiment (Stage C)."""
    from lambda_hat.commands.diagnose_cmd import diagnose_experiment_entry

    sampler_list = [s.strip() for s in samplers.split(",")] if samplers else None
    result = diagnose_experiment_entry(experiment, mode, sampler_list)
    click.echo(
        f"✓ Processed {result['num_runs']} runs: "
        f"{result['num_success']} success, {result['num_failed']} failed"
    )
    if result["failed_runs"]:
        click.echo("Failed runs:", err=True)
        for fr in result["failed_runs"]:
            click.echo(f"  • {fr}", err=True)


@cli.command()
@click.option(
    "--target-id",
    required=True,
    help="Target ID (e.g., 'tgt_abc123')",
)
@click.option(
    "--experiment",
    required=True,
    help="Experiment name (e.g., 'dev', 'smoke')",
)
def diagnose_target(target_id, experiment):
    """Generate target diagnostics (teacher comparison plots) for a single target."""
    from lambda_hat.commands.diagnose_target_cmd import diagnose_target_entry

    result = diagnose_target_entry(target_id, experiment)
    click.echo(f"✓ Generated {len(result['plots_generated'])} plots in {result['diagnostics_dir']}")


@cli.command()
@click.option(
    "--experiment",
    required=True,
    help="Experiment name (e.g., 'dev', 'smoke')",
)
def diagnose_targets(experiment):
    """Generate target diagnostics for all targets in an experiment."""
    from lambda_hat.commands.diagnose_target_cmd import diagnose_targets_entry

    result = diagnose_targets_entry(experiment)
    click.echo(
        f"✓ Processed {result['num_targets']} targets: "
        f"{result['num_success']} success, {result['num_failed']} failed"
    )
    if result["failed_targets"]:
        click.echo("Failed targets:", err=True)
        for ft in result["failed_targets"]:
            click.echo(f"  • {ft}", err=True)


# =============================================================================
# Artifact Management
# =============================================================================


@cli.group()
def artifacts():
    """Artifact management (GC, list, TensorBoard)."""


@artifacts.command("gc")
@click.option(
    "--ttl-days",
    type=int,
    default=None,
    help="Time-to-live in days (default: from LAMBDA_HAT_TTL_DAYS env or 30)",
)
def artifacts_gc(ttl_days):
    """Garbage collect unreachable artifacts."""
    from lambda_hat.commands.artifacts_cmd import gc_entry

    result = gc_entry(ttl_days)
    click.echo(
        f"✓ Removed {result['removed']} unreachable objects (older than {result['ttl_days']}d)"
    )


@artifacts.command("ls")
def artifacts_ls():
    """List experiments and runs."""
    from lambda_hat.commands.artifacts_cmd import ls_entry

    ls_entry()


@artifacts.command("tb")
@click.argument("experiment")
def artifacts_tb(experiment):
    """Show TensorBoard logdir for an experiment.

    Example:
        lambda-hat artifacts tb my_experiment | xargs tensorboard --logdir
    """
    from lambda_hat.commands.artifacts_cmd import tb_entry

    tb_entry(experiment)


# =============================================================================
# Stage D: Promote Results
# =============================================================================


@cli.group()
def promote():
    """Promote plots to galleries (Stage D)."""


@promote.command("single")
@click.option(
    "--runs-root",
    required=True,
    type=click.Path(exists=True),
    help="Root directory containing run subdirectories",
)
@click.option(
    "--samplers",
    required=True,
    help="Comma-separated sampler names (e.g., sgld,hmc,mclmc)",
)
@click.option(
    "--outdir", required=True, type=click.Path(), help="Output directory for promoted plots"
)
@click.option(
    "--plot-name",
    default="trace.png",
    show_default=True,
    help="Name of plot file to copy",
)
def promote_single(runs_root, samplers, outdir, plot_name):
    """Promote plots for a single target."""
    from lambda_hat.commands.promote_cmd import promote_single_entry

    sampler_list = [s.strip() for s in samplers.split(",")]
    promote_single_entry(runs_root, sampler_list, outdir, plot_name)


@promote.command("gallery")
@click.option(
    "--runs-root",
    required=True,
    type=click.Path(exists=True),
    help="Root directory containing run subdirectories",
)
@click.option(
    "--samplers",
    required=True,
    help="Comma-separated sampler names (e.g., sgld,hmc,mclmc)",
)
@click.option(
    "--outdir",
    default="runs/promotion",
    show_default=True,
    type=click.Path(),
    help="Output directory for gallery",
)
@click.option(
    "--plot-name",
    default="trace.png",
    show_default=True,
    help="Name of plot file to copy",
)
@click.option(
    "--snippet-out",
    default=None,
    type=click.Path(),
    help="Optional path to write HTML snippet",
)
def promote_gallery(runs_root, samplers, outdir, plot_name, snippet_out):
    """Generate gallery HTML of all targets."""
    from lambda_hat.commands.promote_cmd import promote_gallery_entry

    sampler_list = [s.strip() for s in samplers.split(",")]
    promote_gallery_entry(runs_root, sampler_list, outdir, plot_name, snippet_out)


# =============================================================================
# Workflow Orchestration
# =============================================================================


@cli.group()
def workflow():
    """Parsl workflow orchestration."""


@workflow.command("sample")
@click.option(
    "--config",
    default="config/experiments.yaml",
    show_default=True,
    type=click.Path(exists=True),
    help="Path to experiments config",
)
@click.option(
    "--experiment",
    default=None,
    help="Experiment name (default: from config or env)",
)
@click.option(
    "--backend",
    type=click.Choice(["local", "slurm-cpu", "slurm-gpu"]),
    default=lambda: os.environ.get("LAMBDA_HAT_BACKEND", "local"),
    show_default="local (or LAMBDA_HAT_BACKEND env var)",
    help="Execution backend: local HTEX, SLURM CPU nodes, or SLURM GPU nodes",
)
@click.option(
    "--set",
    "parsl_sets",
    multiple=True,
    help="Override Parsl card values (e.g., --set walltime=04:00:00)",
)
@click.option(
    "--promote",
    is_flag=True,
    help="Run promotion stage (gallery generation) after sampling (opt-in)",
)
@click.option(
    "--promote-plots",
    default="trace.png,llc_convergence_combined.png",
    show_default=True,
    help="Comma-separated plots to promote when --promote is used",
)
def workflow_sample(config, experiment, backend, parsl_sets, promote, promote_plots):
    """Run N×M targets×samplers workflow.

    This orchestrates the full pipeline:
      A. Build targets (neural networks + datasets)
      B. Run samplers (MCMC/VI) for each target
      C. Generate diagnostics (offline plots from traces) - OPTIONAL (runs when --promote)
      D. Promote results (gallery + aggregation) - OPTIONAL (opt-in via --promote)

    Examples:

      # Local testing (no promotion, fast)
      lambda-hat workflow sample --config config/experiments.yaml --backend local

      # SLURM GPU cluster with promotion (includes diagnostics)
      lambda-hat workflow sample --config config/experiments.yaml \\
          --backend slurm-gpu --promote

      # SLURM CPU cluster
      lambda-hat workflow sample --config config/experiments.yaml --backend slurm-cpu
    """
    # Import main workflow logic from parsl_llc entrypoint
    # We'll delegate to the existing run_workflow function
    import parsl

    from lambda_hat.artifacts import Paths, RunContext
    from lambda_hat.parsl_cards import load_parsl_config_from_card
    from lambda_hat.workflows.sample import run_workflow

    # Initialize artifact system early to get RunContext for Parsl run_dir
    paths_early = Paths.from_env()
    paths_early.ensure()
    exp_config = OmegaConf.load(config)
    experiment_name = experiment or exp_config.get("experiment") or "dev"
    ctx_early = RunContext.create(experiment=experiment_name, algo="parsl_llc", paths=paths_early)

    # Map backend to Parsl card path
    backend_cards = {
        "local": "config/parsl/local.yaml",
        "slurm-cpu": "config/parsl/slurm/cpu.yaml",
        "slurm-gpu": "config/parsl/slurm/gpu-a100.yaml",
    }

    card_path = Path(backend_cards[backend])
    if not card_path.exists():
        click.echo(f"Error: Parsl card not found: {card_path}", err=True)
        sys.exit(1)

    click.echo(f"Using backend: {backend}")
    click.echo(f"  Parsl card: {card_path}")

    # Add run_dir override to parsl_sets
    parsl_sets_list = list(parsl_sets) + [f"run_dir={ctx_early.parsl_dir}"]
    if parsl_sets:
        click.echo(f"  Overrides: {list(parsl_sets)}")

    parsl_cfg = load_parsl_config_from_card(card_path, parsl_sets_list)

    # Parse promote plots
    promote_plots_list = [p.strip() for p in promote_plots.split(",") if p.strip()]

    # Run workflow
    click.echo(f"Using experiments config: {config}")
    if experiment:
        click.echo(f"Experiment: {experiment}")
    if promote:
        click.echo(f"Promotion enabled: {promote_plots_list}")
    else:
        click.echo("Promotion disabled (use --promote to enable)\n")

    # Load Parsl config
    parsl.load(parsl_cfg)

    try:
        output_path = run_workflow(
            config,
            experiment=experiment,
            enable_promotion=promote,
            promote_plots=promote_plots_list,
            is_local=(backend == "local"),  # Enable target diagnostics for local dev
        )
        click.echo(f"\n✓ Workflow complete! Results: {output_path}")
    except Exception as e:
        click.echo(f"\n✗ Workflow failed: {e}", err=True)
        raise
    finally:
        parsl.clear()


@workflow.command("tune")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to Optuna config YAML (defines problems, methods, search spaces, budgets, etc.)",
)
@click.option(
    "--backend",
    type=click.Choice(["local", "slurm-cpu", "slurm-gpu"]),
    default=lambda: os.environ.get("LAMBDA_HAT_BACKEND", "local"),
    show_default="local (or LAMBDA_HAT_BACKEND env var)",
    help="Execution backend: local HTEX, SLURM CPU nodes, or SLURM GPU nodes",
)
@click.option(
    "--set",
    "config_sets",
    multiple=True,
    help="Override config values (OmegaConf dotlist), e.g.: --set optuna.max_trials_per_method=50",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print resolved config and executor routing without running workflow",
)
def workflow_optuna(config, backend, config_sets, dry_run):
    """Run Bayesian hyperparameter optimization (YAML-first).

    This uses Optuna + Parsl to optimize sampler hyperparameters. All settings
    (problems, methods, search spaces, budgets, concurrency) are defined in the
    YAML config file. Use --set for quick overrides.

    Examples:
      # Local testing
      lambda-hat workflow tune --config config/optuna/default.yaml --backend local

      # Override trials and batch size
      lambda-hat workflow tune --config config/optuna/default.yaml --backend local \\
          --set optuna.max_trials_per_method=24 \\
          --set optuna.concurrency.batch_size=6

      # SLURM GPU cluster
      lambda-hat workflow tune --config config/optuna/default.yaml \\
          --backend slurm-gpu

      # Dry-run to preview config
      lambda-hat workflow tune --config config/optuna/default.yaml \\
          --backend local --dry-run
    """

    import parsl

    from lambda_hat.artifacts import Paths, RunContext
    from lambda_hat.config_optuna import load_cfg
    from lambda_hat.logging_config import configure_logging
    from lambda_hat.parsl_cards import load_parsl_config_from_card
    from lambda_hat.workflows.tune import run_optuna_workflow

    # Configure logging
    configure_logging()

    # Initialize artifact system early to get RunContext for Parsl run_dir override
    paths = Paths.from_env()
    paths.ensure()

    # Load config to get namespace for RunContext
    cfg_early = OmegaConf.load(config)
    experiment = cfg_early.store.get("namespace", "optuna")
    ctx = RunContext.create(experiment=experiment, algo="optuna_workflow", paths=paths)

    # Map backend to Parsl card path
    backend_cards = {
        "local": "config/parsl/local.yaml",
        "slurm-cpu": "config/parsl/slurm/cpu.yaml",
        "slurm-gpu": "config/parsl/slurm/gpu-a100.yaml",
    }

    card_path = Path(backend_cards[backend])
    if not card_path.exists():
        click.echo(f"Error: Parsl card not found: {card_path}", err=True)
        sys.exit(1)

    click.echo(f"Using backend: {backend}")
    click.echo(f"  Parsl card: {card_path}")

    parsl_cfg = load_parsl_config_from_card(card_path, [f"run_dir={ctx.parsl_dir}"])

    # Load Parsl
    parsl.load(parsl_cfg)

    # Load and validate Optuna config
    cfg = load_cfg(config, dotlist_overrides=list(config_sets))

    # Dry-run mode: print resolved config and exit
    if dry_run:
        click.echo("\n=== Resolved Optuna Configuration ===")
        click.echo(OmegaConf.to_yaml(cfg))
        click.echo("\n✓ Dry-run complete (no workflow executed)")
        parsl.clear()
        return

    # Run workflow
    click.echo(f"Using Optuna config: {config}")
    if config_sets:
        click.echo(f"Config overrides: {list(config_sets)}")

    try:
        output_path = run_optuna_workflow(cfg)
        click.echo(f"\n✓ Results: {output_path}")
    except Exception as e:
        click.echo(f"\n✗ Workflow failed: {e}", err=True)
        raise
    finally:
        parsl.clear()


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    cli()
