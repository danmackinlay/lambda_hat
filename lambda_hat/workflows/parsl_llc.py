#!/usr/bin/env python3
"""Parsl workflow for Lambda-Hat: N targets × M samplers with optional promotion.

Stages:
  A. Build targets (neural networks + datasets)
  B. Run samplers (MCMC/VI) for each target
  C. Generate diagnostics (offline plots from traces) - OPTIONAL, runs when --promote enabled
  D. Promote results (gallery + aggregation) - OPTIONAL, opt-in via --promote

Usage:
  # Local testing (no promotion)
  lambda-hat workflow llc --config config/experiments.yaml --local

  # SLURM cluster with promotion (includes diagnostics)
  lambda-hat workflow llc --config config/experiments.yaml \\
      --parsl-card config/parsl/slurm/gpu-a100.yaml --promote
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import parsl
from omegaconf import OmegaConf
from parsl import python_app

from lambda_hat.artifacts import Paths, RunContext
from lambda_hat.logging_config import configure_logging
from lambda_hat.parsl_cards import load_parsl_config_from_card
from lambda_hat.workflow_utils import (
    compose_build_cfg,
    compose_sample_cfg,
    run_id_for,
    target_id_for,
)

log = logging.getLogger(__name__)

# ============================================================================
# Parsl Error Extraction
# ============================================================================


def unwrap_parsl_future(future, name: str):
    """Extract and surface exceptions from Parsl futures with full diagnostics.

    Args:
        future: Parsl AppFuture to unwrap
        name: Descriptive name for logging (e.g., "build_tgt_abc123")

    Returns:
        Future result if successful

    Raises:
        Original exception with enhanced logging of remote stdout/stderr
    """
    try:
        return future.result()
    except Exception as e:
        import traceback

        log.error(f"[{name}] FAILED in worker:")
        log.error("".join(traceback.format_exception(type(e), e, e.__traceback__)))

        # Dump remote debug info if available
        if hasattr(e, "stdout") and e.stdout:
            log.error("---- WORKER STDOUT ----\n%s", e.stdout)
        if hasattr(e, "stderr") and e.stderr:
            log.error("---- WORKER STDERR ----\n%s", e.stderr)

        # Log exception attributes for debugging Parsl wrappers
        log.error("Exception type: %s", type(e).__name__)
        log.error("Exception attributes: %s", dir(e))

        raise


# ============================================================================
# Helper Functions
# ============================================================================


def get_executor_for_dtype(dtype_str: str) -> str:
    """Determine which executor to use based on dtype.

    Args:
        dtype_str: Either "float64" or "float32"

    Returns:
        Executor label: "htex64" for float64, "htex32" for float32
    """
    return "htex64" if dtype_str == "float64" else "htex32"


def get_sampler_dtype(sample_cfg: OmegaConf, sampler_name: str) -> str:
    """Extract sampler's dtype from sample config.

    Args:
        sample_cfg: Composed sample configuration
        sampler_name: Sampler name (hmc, mclmc, sgld, vi)

    Returns:
        dtype string: "float64" or "float32"
    """
    # Read dtype from sampler config, matching sampling_runner.py logic
    if sampler_name in ("hmc", "mclmc"):
        return str(sample_cfg.sampler.get(sampler_name, {}).get("dtype", "float64"))
    elif sampler_name == "sgld":
        return str(sample_cfg.sampler.sgld.get("dtype", "float32"))
    elif sampler_name == "vi":
        return str(sample_cfg.sampler.vi.dtype)
    else:
        # Default to float32 for unknown samplers
        return "float32"


# ============================================================================
# Parsl Apps (task definitions)
# ============================================================================


@python_app
def build_target_app(cfg_yaml, target_id, experiment):
    """Build a target (train neural network) via direct command call.

    Environment is set by executor's worker_init (JAX_ENABLE_X64, MPLBACKEND).

    Args:
        cfg_yaml: Path to composed build config YAML
        target_id: Target identifier (e.g., 'tgt_abc123')
        experiment: Experiment name for artifact system

    Returns:
        dict: Build result from build_entry with keys:
            - urn: Artifact URN
            - target_id: Target ID
            - run_id: Run ID
            - L0: Initial loss
            - experiment: Experiment name
    """
    from lambda_hat.commands.build_cmd import build_entry

    return build_entry(cfg_yaml, target_id, experiment)


@python_app
def run_sampler_app(cfg_yaml, target_id, experiment, inputs=None):
    """Run a sampler (MCMC/VI) for a target via direct command call.

    Environment is set by executor's worker_init (JAX_ENABLE_X64, MPLBACKEND).

    Args:
        cfg_yaml: Path to composed sample config YAML
        target_id: Target identifier
        experiment: Experiment name for artifact system
        inputs: List of futures this task depends on (target build)

    Returns:
        dict: Sample result from sample_entry with keys:
            - run_id: Run ID
            - run_dir: Path to run directory
            - metrics: Analysis metrics
            - experiment: Experiment name
    """
    from lambda_hat.commands.sample_cmd import sample_entry

    return sample_entry(cfg_yaml, target_id, experiment)


@python_app
def diagnose_app(run_dir, mode, inputs=None):
    """Generate offline diagnostics for a completed sampling run via direct command call.

    Args:
        run_dir: Path to run directory containing trace.nc or traces_raw.json
        mode: Diagnostic depth - "light" or "full"
        inputs: List of futures this task depends on (sampling run)

    Returns:
        dict: Diagnostic result from diagnose_entry with keys:
            - run_dir: Path to run directory
            - diagnostics_dir: Path to diagnostics output
            - plots_generated: List of plot filenames created
            - mode: Diagnostic mode used
    """
    from lambda_hat.commands.diagnose_cmd import diagnose_entry

    return diagnose_entry(run_dir, mode)


@python_app
def promote_app(store_root, samplers, outdir, plot_name, inputs=None):
    """Promote results: create gallery with newest run per sampler via direct command call.

    Args:
        store_root: Root directory for runs
        samplers: List of sampler names
        outdir: Output directory for promotion assets
        plot_name: Name of plot to promote (e.g., 'trace.png')
        inputs: List of futures this task depends on (all diagnostics)

    Returns:
        str: Path to generated markdown snippet
    """
    from pathlib import Path

    from lambda_hat.commands.promote_cmd import promote_gallery_entry

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    snippet_out = outdir / f"gallery_{plot_name.replace('.png', '')}.md"

    return promote_gallery_entry(
        runs_root=store_root,
        samplers=samplers,
        outdir=str(outdir),
        plot_name=plot_name,
        snippet_out=str(snippet_out),
    )


# ============================================================================
# Main Workflow
# ============================================================================


def run_workflow(
    experiments_yaml,
    experiment=None,
    parsl_config_path=None,  # Deprecated, kept for compatibility
    enable_promotion=False,
    promote_plots=None,
    is_local=False,  # NEW: Whether running in local mode (vs SLURM/cluster)
):
    """Execute the full Lambda-Hat workflow: build → sample → (optional) promote.

    Args:
        experiments_yaml: Path to experiments config (e.g., config/experiments.yaml)
        experiment: Experiment name (default: from config or env LAMBDA_HAT_DEFAULT_EXPERIMENT)
        parsl_config_path: [DEPRECATED] Ignored (Parsl config loaded via main())
        enable_promotion: Whether to run promotion stage (default: False, opt-in)
        promote_plots: List of plot names to promote
            (default: ['trace.png', 'llc_convergence_combined.png'])
        is_local: Whether running in local mode (enables target diagnostics for dev visibility)

    Returns:
        Path to aggregated results parquet file
    """
    if promote_plots is None:
        promote_plots = ["trace.png", "llc_convergence_combined.png"]

    # Conditional defaults for diagnostics
    import os

    # Target diagnostics: ON for local dev, OFF for Parsl workflows (keep workers lightweight)
    if is_local:
        os.environ.setdefault("LAMBDA_HAT_SKIP_DIAGNOSTICS", "0")  # Local: teacher plots ON
    else:
        os.environ.setdefault("LAMBDA_HAT_SKIP_DIAGNOSTICS", "1")  # Parsl: teacher plots OFF

    os.environ.setdefault("LAMBDA_HAT_ANALYSIS_MODE", "none")  # Offline sampling diagnostics

    # Initialize artifact system
    paths = Paths.from_env()
    paths.ensure()

    # Load experiment configuration (Parsl config already loaded in main())
    exp = OmegaConf.load(experiments_yaml)
    experiment = experiment or exp.get("experiment") or "dev"
    jax_x64 = bool(exp.get("jax_enable_x64", True))
    # Legacy store_root for promote functionality (backward compatibility)
    store_root = exp.get("store_root", "runs")

    # Determine default executor for build tasks based on global jax_x64 setting
    build_executor = get_executor_for_dtype("float64" if jax_x64 else "float32")

    targets_conf = list(exp["targets"])
    samplers_conf = list(exp["samplers"])

    # Create RunContext for this workflow orchestration
    ctx = RunContext.create(experiment=experiment, algo="parsl_llc", paths=paths)

    log.info("Loaded %d targets and %d samplers", len(targets_conf), len(samplers_conf))
    log.info("Experiment: %s, JAX x64: %s", experiment, jax_x64)
    log.info("Run dir: %s", ctx.run_dir)
    log.info("Artifacts: %s", ctx.artifacts_dir)
    log.info("Logs: %s", ctx.logs_dir)
    log.info("Scratch: %s", ctx.scratch_dir)

    # Create config directory in scratch
    temp_cfg_dir = ctx.scratch_dir / "configs"
    temp_cfg_dir.mkdir(exist_ok=True, parents=True)

    # ========================================================================
    # Stage A: Build Targets
    # ========================================================================

    log.info("=== Stage A: Building Targets ===")
    target_futures = {}
    target_ids = []

    # Create log subdirectory for build tasks
    build_log_dir = ctx.logs_dir / "build_target"
    build_log_dir.mkdir(parents=True, exist_ok=True)

    for t in targets_conf:
        # Compose build config and compute target ID
        build_cfg = compose_build_cfg(t, jax_enable_x64=jax_x64)
        tid = target_id_for(build_cfg)
        target_ids.append(tid)

        # Write temp config YAML
        cfg_yaml_path = temp_cfg_dir / f"build_{tid}.yaml"
        cfg_yaml_path.write_text(OmegaConf.to_yaml(build_cfg))

        # Submit build job (uses artifact system via command modules)
        # Note: executor routing disabled for now - both executors support all dtypes
        log.info(
            "  Submitting build for %s (model=%s, data=%s)",
            tid,
            t["model"],
            t["data"],
        )
        future = build_target_app(
            cfg_yaml=str(cfg_yaml_path),
            target_id=tid,
            experiment=experiment,
        )
        target_futures[tid] = future

    # ========================================================================
    # Stage B: Run Samplers
    # ========================================================================

    log.info("=== Stage B: Running Samplers ===")
    run_futures = []
    run_records = []

    # Create log subdirectory for sampler tasks
    sample_log_dir = ctx.logs_dir / "run_sampler"
    sample_log_dir.mkdir(parents=True, exist_ok=True)

    for tid in target_ids:
        for s in samplers_conf:
            # Compose sample config and compute run ID
            sample_cfg = compose_sample_cfg(tid, s, jax_enable_x64=jax_x64)
            rid = run_id_for(sample_cfg)

            # Determine executor based on sampler's dtype
            sampler_name = s["name"]
            dtype = get_sampler_dtype(sample_cfg, sampler_name)
            executor = get_executor_for_dtype(dtype)

            # Write temp config YAML
            cfg_yaml_path = temp_cfg_dir / f"sample_{tid}_{s['name']}_{rid}.yaml"
            cfg_yaml_path.write_text(OmegaConf.to_yaml(sample_cfg))

            # Submit sampling job (uses artifact system via command modules)
            # Note: executor routing disabled for now - both executors support all dtypes
            log.info(
                "  Submitting %s for %s (run_id=%s, dtype=%s)",
                sampler_name,
                tid,
                rid,
                dtype,
            )
            future = run_sampler_app(
                cfg_yaml=str(cfg_yaml_path),
                target_id=tid,
                experiment=experiment,
                inputs=[target_futures[tid]],  # Dependency: wait for target build
            )
            run_futures.append(future)

            # Record metadata for aggregation (run_dir will be under artifact system)
            run_records.append(
                {
                    "target_id": tid,
                    "sampler": s["name"],
                    "run_id": rid,
                    "cfg_yaml": str(cfg_yaml_path),
                }
            )

    # ========================================================================
    # Wait for all runs to complete
    # ========================================================================

    log.info("=== Waiting for %d sampling runs to complete ===", len(run_futures))
    failed_runs = []

    for i, (future, record) in enumerate(zip(run_futures, run_records), 1):
        try:
            name = f"{record['target_id']}_{record['sampler']}_{record['run_id']}"
            unwrap_parsl_future(future, name)
            target_id = record["target_id"]
            sampler = record["sampler"]
            run_id = record["run_id"]
            log.info("  [%d/%d] ✓ %s/%s/%s", i, len(run_futures), target_id, sampler, run_id)
        except Exception as e:
            # Construct log paths from record metadata
            stderr_path = (
                sample_log_dir / f"{record['target_id']}_{record['sampler']}_{record['run_id']}.err"
            )
            stdout_path = (
                sample_log_dir / f"{record['target_id']}_{record['sampler']}_{record['run_id']}.log"
            )

            # Read last 15 lines of stderr if available
            stderr_tail = ""
            if stderr_path.exists():
                try:
                    with open(stderr_path) as f:
                        lines = f.readlines()
                        stderr_tail = "".join(lines[-15:])
                except Exception:
                    pass

            target_id = record["target_id"]
            sampler = record["sampler"]
            run_id = record["run_id"]
            log.error(
                "  [%d/%d] ✗ FAILED: %s/%s/%s", i, len(run_futures), target_id, sampler, run_id
            )
            log.error("    Error: %s", str(e))
            log.error("    Stderr:  %s", stderr_path)
            log.error("    Stdout:  %s", stdout_path)

            if stderr_tail:
                log.error("    --- Last 15 lines of stderr ---")
                for line in stderr_tail.splitlines():
                    log.error("    %s", line)
                log.error("    --- End stderr ---")

            failed_runs.append({**record, "error": str(e), "stderr_path": str(stderr_path)})

    # Summary of failures
    if failed_runs:
        log.error("⚠ FAILURE SUMMARY: %d of %d runs failed", len(failed_runs), len(run_futures))
        for fr in failed_runs:
            log.error("  • %s/%s/%s", fr["target_id"], fr["sampler"], fr["run_id"])
            log.error("    Check logs: %s", fr["stderr_path"])

    # ========================================================================
    # Stage C: Generate Diagnostics (optional, runs when promotion enabled)
    # ========================================================================

    if enable_promotion:
        log.info("=== Stage C: Generating Diagnostics ===")

        # Extract run directories from completed sampling runs
        diagnose_futures = []
        diagnose_records = []

        for i, (future, record) in enumerate(zip(run_futures, run_records)):
            # Get result to extract run_dir (sampling already completed in wait loop above)
            try:
                result = future.result()  # Already completed, won't block
                run_dir = result["run_dir"]

                # Submit diagnose job (light mode for faster processing)
                log.info(
                    "  Submitting diagnostics for %s/%s", record["target_id"], record["sampler"]
                )
                diagnose_future = diagnose_app(
                    run_dir=run_dir,
                    mode="light",
                    inputs=[future],  # Dependency: wait for sampling run
                )
                diagnose_futures.append(diagnose_future)
                diagnose_records.append({**record, "run_dir": run_dir})
            except Exception:
                # Skip failed runs (they're already logged in the wait loop)
                log.warning(
                    "  Skipping diagnostics for failed run: %s/%s",
                    record["target_id"],
                    record["sampler"],
                )
                continue

        # Wait for all diagnostics to complete
        log.info("=== Waiting for %d diagnostic tasks to complete ===", len(diagnose_futures))
        for i, (future, record) in enumerate(zip(diagnose_futures, diagnose_records), 1):
            try:
                name = f"diagnose_{record['target_id']}_{record['sampler']}"
                unwrap_parsl_future(future, name)
                log.info(
                    "  [%d/%d] ✓ Diagnostics: %s/%s",
                    i,
                    len(diagnose_futures),
                    record["target_id"],
                    record["sampler"],
                )
            except Exception as e:
                log.error(
                    "  [%d/%d] ✗ Diagnostics FAILED: %s/%s - %s",
                    i,
                    len(diagnose_futures),
                    record["target_id"],
                    record["sampler"],
                    e,
                )
    else:
        log.info("=== Stage C: Diagnostics skipped (use --promote to enable) ===")
        diagnose_futures = []  # Empty list for promotion dependency

    # ========================================================================
    # Stage D: Promotion (optional, opt-in)
    # ========================================================================

    if enable_promotion:
        log.info("=== Stage D: Promotion ===")
        unique_samplers = sorted({s["name"] for s in samplers_conf})

        # Promotion outputs go to artifacts directory
        outdir = ctx.artifacts_dir / "promotion"

        # Promote target diagnostics (teacher comparison plots)
        # NOTE: This runs inline (not as Parsl app) since it's just file copying
        # and target diagnostics only exist for local builds (SKIP_DIAGNOSTICS=0)
        from lambda_hat.promote.core import promote_target_diagnostics

        log.info("  Promoting target diagnostics...")
        try:
            runs_root = Path(store_root)
            promoted_targets = promote_target_diagnostics(runs_root, outdir)
            if promoted_targets:
                log.info("    → Promoted diagnostics for %d targets", len(promoted_targets))
                for target_id, plots in promoted_targets:
                    log.info("      • %s: %d plots", target_id, len(plots))
            else:
                log.info("    → No target diagnostics found (expected for Parsl runs)")
        except Exception as e:
            log.warning("    → Target promotion failed (non-fatal): %s", e)

        # Promote each plot type
        for plot_name in promote_plots:
            log.info("  Promoting %s...", plot_name)
            promote_future = promote_app(
                store_root=store_root,
                samplers=unique_samplers,
                outdir=str(outdir),
                plot_name=plot_name,
                inputs=diagnose_futures,  # Wait for all diagnostics
            )
            try:
                md_path = unwrap_parsl_future(promote_future, f"promote_{plot_name}")
                log.info("    → Gallery written to %s", md_path)
            except Exception as e:
                log.error("    → Promotion FAILED: %s", e)
    else:
        log.info("=== Stage D: Promotion skipped (use --promote to enable) ===")

    # ========================================================================
    # Aggregate results into single parquet file
    # ========================================================================

    log.info("=== Aggregating Results ===")
    rows = []

    # Find all run directories under the experiment (sample entrypoint creates them)
    # They're named like: 20251116T...-vi-tgt_abc123...-123abc/
    experiment_runs_dir = paths.experiments / experiment / "runs"
    if experiment_runs_dir.exists():
        for run_dir in experiment_runs_dir.glob("*/"):
            analysis_path = run_dir / "analysis.json"
            manifest_path = run_dir / "manifest.json"

            if analysis_path.exists() and manifest_path.exists():
                try:
                    metrics = json.loads(analysis_path.read_text())
                    manifest = json.loads(manifest_path.read_text())

                    # Combine manifest metadata with analysis metrics
                    row = {
                        "run_id": manifest.get("run_id"),
                        "target_id": manifest.get("target_id"),
                        "sampler": manifest.get("sampler"),
                        "experiment": manifest.get("experiment"),
                        **metrics,
                    }
                    rows.append(row)
                except Exception as e:
                    log.warning(
                        "  Warning: Failed to read %s or %s: %s", analysis_path, manifest_path, e
                    )
            elif not analysis_path.exists():
                log.warning("  Warning: Missing analysis.json at %s", analysis_path)

    df = pd.DataFrame(rows)
    output_path = ctx.artifacts_dir / "llc_runs.parquet"

    df.to_parquet(output_path, index=False)
    log.info("Wrote %d rows to %s", len(df), output_path)

    return output_path


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI entry point for Parsl workflow."""
    parser = argparse.ArgumentParser(
        description="Run Lambda-Hat workflow with Parsl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config/experiments.yaml",
        help="Path to experiments config (default: config/experiments.yaml)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment name (default: from config or env LAMBDA_HAT_DEFAULT_EXPERIMENT)",
    )
    parser.add_argument(
        "--parsl-card",
        default=None,
        help="Path to Parsl YAML card (e.g., config/parsl/slurm/cpu.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="parsl_sets",
        action="append",
        default=[],
        help=(
            "Override Parsl card values (OmegaConf dotlist), "
            "e.g.: --set walltime=04:00:00 --set gpus_per_node=1"
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local HTEX executors (equivalent to --parsl-card config/parsl/local.yaml)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Run promotion stage (gallery generation) after sampling (opt-in)",
    )
    parser.add_argument(
        "--promote-plots",
        default="trace.png,llc_convergence_combined.png",
        help="Comma-separated plots to promote when --promote is used (default: trace,convergence)",
    )

    args = parser.parse_args()

    # Configure logging at entrypoint
    configure_logging()

    # Initialize artifact system early to get RunContext for Parsl run_dir
    from lambda_hat.artifacts import Paths, RunContext

    paths_early = Paths.from_env()
    paths_early.ensure()
    exp_early = OmegaConf.load(args.config)
    experiment_early = args.experiment or exp_early.get("experiment") or "dev"
    ctx_early = RunContext.create(experiment=experiment_early, algo="parsl_llc", paths=paths_early)

    # Resolve Parsl config
    if args.local and not args.parsl_card:
        # Local mode: load local.yaml card with RunContext run_dir
        log.info("Using Parsl mode: local (dual HTEX)")
        local_card_path = Path("config/parsl/local.yaml")
        if not local_card_path.exists():
            log.error("Local card not found: %s", local_card_path)
            sys.exit(1)
        parsl_cfg = load_parsl_config_from_card(local_card_path, [f"run_dir={ctx_early.parsl_dir}"])
    elif args.parsl_card:
        # Card-based config with run_dir override
        card_path = Path(args.parsl_card)
        if not card_path.is_absolute():
            card_path = Path.cwd() / card_path
        if not card_path.exists():
            log.error("Parsl card not found: %s", card_path)
            sys.exit(1)
        log.info("Using Parsl card: %s", card_path)
        # Add run_dir override to parsl_sets
        parsl_sets_with_rundir = (args.parsl_sets or []) + [f"run_dir={ctx_early.parsl_dir}"]
        if args.parsl_sets:
            log.info("  Overrides: %s", args.parsl_sets)
        parsl_cfg = load_parsl_config_from_card(card_path, parsl_sets_with_rundir)
    else:
        log.error("Error: Must specify either --local or --parsl-card")
        sys.exit(1)

    # Parse promote plots
    promote_plots = [p.strip() for p in args.promote_plots.split(",") if p.strip()]

    # Run workflow
    log.info("Using experiments config: %s", args.config)
    if args.experiment:
        log.info("Experiment: %s", args.experiment)
    if args.promote:
        log.info("Promotion enabled: %s", promote_plots)
    else:
        log.info("Promotion disabled (use --promote to enable)")

    # Load Parsl config
    parsl.load(parsl_cfg)

    try:
        output_path = run_workflow(
            args.config,
            experiment=args.experiment,
            enable_promotion=args.promote,
            promote_plots=promote_plots,
        )
        log.info("✓ Workflow complete! Results: %s", output_path)
    except Exception as e:
        log.error("✗ Workflow failed: %s", e)
        raise
    finally:
        parsl.clear()


if __name__ == "__main__":
    main()
