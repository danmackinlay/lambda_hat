#!/usr/bin/env python3
"""Parsl workflow for Lambda-Hat: N targets × M samplers with optional promotion.

Stages:
  A. Build targets (neural networks + datasets)
  B. Run samplers (MCMC/VI) for each target
  C. Generate diagnostics (offline plots from traces) - OPTIONAL, runs when --promote enabled
  C-2. Copy target diagnostics to repository (docs/assets/) - OPTIONAL, runs when --promote enabled
  D. Promote sampler results (gallery + aggregation) to repository - OPTIONAL, opt-in via --promote

Promotion outputs (repository-visible):
  docs/assets/<experiment>/samplers/*.png    - Promoted sampler plots per algorithm
  docs/assets/<experiment>/samplers/*.md     - Gallery markdown snippets
  docs/assets/<experiment>/targets/          - Target diagnostic plots

Artifact outputs (immutable, content-addressed):
  artifacts/experiments/<exp>/runs/          - Run directories with diagnostics
  artifacts/experiments/<exp>/targets/       - Target directories with diagnostics
  artifacts/experiments/<exp>/artifacts/     - Workflow artifacts (e.g., llc_runs.parquet)

Usage:
  # Local testing (no promotion)
  lambda-hat workflow llc --config config/experiments.yaml --backend local

  # SLURM cluster with promotion (includes diagnostics + repo copy)
  lambda-hat workflow llc --config config/experiments.yaml \\
      --parsl-card config/parsl/slurm/gpu-a100.yaml --promote
"""

import argparse
import json
import logging
import shutil
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


def copy_promotion_to_repo(src_dir: Path, dest_dir: Path) -> None:
    """Copy promoted assets from artifact system to repository-visible location.

    Args:
        src_dir: Source directory in artifact system (e.g., artifacts/.../promotion/trace/)
        dest_dir: Destination in repository (e.g., docs/assets/<exp>/samplers/)

    Copies all .png and .md files, creating dest_dir if needed.
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)

    if not src_dir.exists():
        log.warning("Source directory does not exist: %s - skipping copy", src_dir)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy all .png and .md files
    copied_count = 0
    for pattern in ["*.png", "*.md"]:
        for src_file in src_dir.glob(pattern):
            dst_file = dest_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            log.info("  Copied to repo: %s -> %s", src_file.name, dst_file)
            copied_count += 1

    if copied_count == 0:
        log.warning("No .png or .md files found in %s", src_dir)


def copy_target_diagnostics_to_repo(targets_dir: Path, dest_dir: Path) -> None:
    """Copy target diagnostics from artifact system to repository-visible location.

    Args:
        targets_dir: Source targets directory (e.g., artifacts/experiments/<exp>/targets/)
        dest_dir: Destination in repository (e.g., docs/assets/<exp>/targets/)

    Recursively copies diagnostics/*.png preserving directory structure.
    """
    targets_dir = Path(targets_dir)
    dest_dir = Path(dest_dir)

    if not targets_dir.exists():
        log.warning("Targets directory does not exist: %s - skipping copy", targets_dir)
        return

    copied_count = 0
    for diagnostic_file in targets_dir.rglob("diagnostics/*.png"):
        # Compute relative path from targets_dir to preserve structure
        rel_path = diagnostic_file.relative_to(targets_dir)
        dst_file = dest_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(diagnostic_file, dst_file)
        log.info("  Copied target diagnostic: %s -> %s", rel_path, dst_file)
        copied_count += 1

    if copied_count > 0:
        log.info("  Copied %d target diagnostic files to repository", copied_count)
    else:
        log.info("  No target diagnostics found to copy")


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
        run_dir: Path to run directory containing trace.nc or traces_raw.npz
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
def promote_app(runs_root, samplers, outdir, plot_name, inputs=None):
    """Promote results: create gallery with newest run per sampler via direct command call.

    Args:
        runs_root: Path to experiments/{exp}/runs/ directory (artifact system)
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
        runs_root=runs_root,
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
    enable_promotion=False,
    promote_plots=None,
    is_local=False,  # NEW: Whether running in local mode (vs SLURM/cluster)
):
    """Execute the full Lambda-Hat workflow: build → sample → (optional) promote.

    Args:
        experiments_yaml: Path to experiments config (e.g., config/experiments.yaml)
        experiment: Experiment name (default: from config or env LAMBDA_HAT_DEFAULT_EXPERIMENT)
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

    # Note: LAMBDA_HAT_ANALYSIS_MODE removed - workers always write raw traces only
    # Diagnostics are deferred to Stage C (diagnose command) using the golden path

    # Initialize artifact system
    paths = Paths.from_env()
    paths.ensure()

    # Load experiment configuration (Parsl config already loaded in main())
    exp = OmegaConf.load(experiments_yaml)
    experiment = experiment or exp.get("experiment") or "dev"
    jax_x64 = bool(exp.get("jax_enable_x64", True))
    # Note: store_root removed - promotion now uses artifact system paths directly

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
            sampler_name = s["name"]

            # Write temp config YAML
            cfg_yaml_path = temp_cfg_dir / f"sample_{tid}_{s['name']}_{rid}.yaml"
            cfg_yaml_path.write_text(OmegaConf.to_yaml(sample_cfg))

            # Submit sampling job (uses artifact system via command modules)
            log.info(
                "  Submitting %s for %s (run_id=%s)",
                sampler_name,
                tid,
                rid,
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

        # ====================================================================
        # Stage C-2: Copy target diagnostics to repository (optional)
        # ====================================================================
        log.info("=== Stage C-2: Copying target diagnostics to repository ===")
        targets_src = paths.experiments / experiment / "targets"
        targets_dest = Path("docs/assets") / experiment / "targets"
        copy_target_diagnostics_to_repo(targets_src, targets_dest)
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

        # Note: Target diagnostics promotion removed (obsolete with artifact system)
        # Target diagnostics already exist at correct location:
        # artifacts/experiments/{exp}/targets/{id}/diagnostics/

        # Promote sampler diagnostic plots
        # Compute runs_root from artifact system paths
        runs_root = paths.experiments / experiment / "runs"

        for plot_name in promote_plots:
            log.info("  Promoting %s...", plot_name)
            promote_future = promote_app(
                runs_root=str(runs_root),
                samplers=unique_samplers,
                outdir=str(outdir),
                plot_name=plot_name,
                inputs=diagnose_futures,  # Wait for all diagnostics
            )
            try:
                md_path = unwrap_parsl_future(promote_future, f"promote_{plot_name}")
                log.info("    → Gallery written to %s", md_path)

                # Copy promoted assets to repository-visible location
                # promote_gallery() writes files directly to outdir (not in subdirectories)
                src_dir = outdir
                dest_dir = Path("docs/assets") / experiment / "samplers"
                log.info("    → Copying to repository: %s", dest_dir)
                copy_promotion_to_repo(src_dir, dest_dir)
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
            traces_raw_path = run_dir / "traces_raw.npz"

            traces_raw_exists = traces_raw_path.exists()
            manifest_exists = manifest_path.exists()

            # Skip non-sampler runs quietly (build/parsl orchestration)
            # Case 1: No manifest and no traces → not an artifact we care about
            if not manifest_exists and not traces_raw_exists:
                continue

            # Case 2: Manifest exists but no traces → build/parsl run, skip quietly
            if manifest_exists and not traces_raw_exists:
                continue

            # Case 3: Traces exist but no manifest → genuinely malformed, warn
            if traces_raw_exists and not manifest_exists:
                log.warning("  Warning: traces_raw.npz without manifest at %s, skipping", run_dir)
                continue

            # Case 4: Both exist → genuine sampling run, process it
            # Auto-diagnose if analysis.json is missing or stale
            if not analysis_path.exists() or (
                analysis_path.stat().st_mtime < traces_raw_path.stat().st_mtime
            ):
                log.info("  Auto-diagnosing %s (missing or stale analysis)", run_dir.name)
                try:
                    from lambda_hat.commands.diagnose_cmd import diagnose_entry

                    diagnose_entry(str(run_dir), mode="light")
                except Exception as e:
                    log.warning("  Warning: Failed to diagnose %s: %s", run_dir.name, e)
                    continue

            # Load metrics and manifest
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
        "--backend local ",
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
        log.error("Error: Must specify either --backend local  or --parsl-card")
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
