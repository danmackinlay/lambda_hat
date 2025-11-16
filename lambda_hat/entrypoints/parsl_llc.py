#!/usr/bin/env python3
"""Parsl workflow for Lambda-Hat: N targets × M samplers with optional promotion.

Stages:
  A. Build targets (neural networks + datasets)
  B. Run samplers (MCMC/VI) for each target
  C. Promote results (gallery + aggregation) - OPTIONAL, opt-in via --promote

Usage:
  # Local testing (no promotion)
  parsl-llc --config config/experiments.yaml --local

  # SLURM cluster with promotion
  parsl-llc --config config/experiments.yaml \\
      --parsl-card config/parsl/slurm/gpu-a100.yaml --promote
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import parsl
from omegaconf import OmegaConf
from parsl import bash_app, python_app

from lambda_hat.artifacts import ArtifactStore, Paths, RunContext
from lambda_hat.parsl_cards import build_parsl_config_from_card, load_parsl_config_from_card
from lambda_hat.promote.core import promote_gallery
from lambda_hat.workflow_utils import (
    compose_build_cfg,
    compose_sample_cfg,
    run_id_for,
    target_id_for,
)

# ============================================================================
# Parsl Apps (task definitions)
# ============================================================================


@bash_app
def build_target_app(cfg_yaml, target_id, experiment, jax_x64, stdout=None, stderr=None):
    """Build a target (train neural network) via CLI entrypoint.

    Args:
        cfg_yaml: Path to composed build config YAML
        target_id: Target identifier (e.g., 'tgt_abc123')
        experiment: Experiment name for artifact system
        jax_x64: JAX precision flag (0 or 1)
        stdout: Log file for stdout
        stderr: Log file for stderr

    Returns:
        Shell command string
    """
    return f"""
JAX_ENABLE_X64={jax_x64} uv run python -m lambda_hat.entrypoints.build_target \
  --config-yaml {cfg_yaml} \
  --target-id {target_id} \
  --experiment {experiment}
    """.strip()


@bash_app
def run_sampler_app(
    cfg_yaml, target_id, experiment, jax_x64, inputs=None, stdout=None, stderr=None
):
    """Run a sampler (MCMC/VI) for a target via CLI entrypoint.

    Args:
        cfg_yaml: Path to composed sample config YAML
        target_id: Target identifier
        experiment: Experiment name for artifact system
        jax_x64: JAX precision flag (0 or 1)
        inputs: List of futures this task depends on (target build)
        stdout: Log file for stdout
        stderr: Log file for stderr

    Returns:
        Shell command string
    """
    return f"""
JAX_ENABLE_X64={jax_x64} uv run python -m lambda_hat.entrypoints.sample \
  --config-yaml {cfg_yaml} \
  --target-id {target_id} \
  --experiment {experiment}
    """.strip()


@python_app
def promote_app(store_root, samplers, outdir, plot_name, inputs=None):
    """Promote results: create gallery with newest run per sampler.

    Args:
        store_root: Root directory for runs
        samplers: List of sampler names
        outdir: Output directory for promotion assets
        plot_name: Name of plot to promote (e.g., 'trace.png')
        inputs: List of futures this task depends on (all sampling runs)

    Returns:
        Path to generated markdown snippet
    """
    from pathlib import Path

    store_root = Path(store_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    md_snippet_out = outdir / f"gallery_{plot_name.replace('.png', '')}.md"

    promote_gallery(
        store_root,
        samplers,
        outdir,
        plot_name=plot_name,
        md_snippet_out=md_snippet_out,
    )

    return str(md_snippet_out)


# ============================================================================
# Main Workflow
# ============================================================================


def run_workflow(
    experiments_yaml,
    experiment=None,
    parsl_config_path=None,  # Deprecated, kept for compatibility
    enable_promotion=False,
    promote_plots=None,
):
    """Execute the full Lambda-Hat workflow: build → sample → (optional) promote.

    Args:
        experiments_yaml: Path to experiments config (e.g., config/experiments.yaml)
        experiment: Experiment name (default: from config or env LAMBDA_HAT_DEFAULT_EXPERIMENT)
        parsl_config_path: [DEPRECATED] Ignored (Parsl config loaded via main())
        enable_promotion: Whether to run promotion stage (default: False, opt-in)
        promote_plots: List of plot names to promote (default: ['trace.png', 'llc_convergence_combined.png'])

    Returns:
        Path to aggregated results parquet file
    """
    if promote_plots is None:
        promote_plots = ["trace.png", "llc_convergence_combined.png"]

    # Initialize artifact system
    paths = Paths.from_env()
    paths.ensure()
    store = ArtifactStore(paths.store)

    # Load experiment configuration (Parsl config already loaded in main())
    exp = OmegaConf.load(experiments_yaml)
    experiment = experiment or exp.get("experiment") or "dev"
    jax_x64 = bool(exp.get("jax_enable_x64", True))
    # Legacy store_root for promote functionality (backward compatibility)
    store_root = exp.get("store_root", "runs")
    jax_x64_flag = 1 if jax_x64 else 0

    targets_conf = list(exp["targets"])
    samplers_conf = list(exp["samplers"])

    # Create RunContext for this workflow orchestration
    ctx = RunContext.create(experiment=experiment, algo="parsl_llc", paths=paths)

    print(f"Loaded {len(targets_conf)} targets and {len(samplers_conf)} samplers")
    print(f"Experiment: {experiment}, JAX x64: {jax_x64}")
    print(f"Run dir: {ctx.run_dir}")
    print(f"Artifacts: {ctx.artifacts_dir}")
    print(f"Logs: {ctx.logs_dir}")
    print(f"Scratch: {ctx.scratch_dir}")

    # Create config directory in scratch
    temp_cfg_dir = ctx.scratch_dir / "configs"
    temp_cfg_dir.mkdir(exist_ok=True, parents=True)

    # ========================================================================
    # Stage A: Build Targets
    # ========================================================================

    print("\n=== Stage A: Building Targets ===")
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

        # Submit build job (no target_dir - entrypoint uses artifact system)
        print(f"  Submitting build for {tid} (model={t['model']}, data={t['data']})")
        future = build_target_app(
            cfg_yaml=str(cfg_yaml_path),
            target_id=tid,
            experiment=experiment,
            jax_x64=jax_x64_flag,
            stdout=str(build_log_dir / f"{tid}.log"),
            stderr=str(build_log_dir / f"{tid}.err"),
        )
        target_futures[tid] = future

    # ========================================================================
    # Stage B: Run Samplers
    # ========================================================================

    print("\n=== Stage B: Running Samplers ===")
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

            # Write temp config YAML
            cfg_yaml_path = temp_cfg_dir / f"sample_{tid}_{s['name']}_{rid}.yaml"
            cfg_yaml_path.write_text(OmegaConf.to_yaml(sample_cfg))

            # Submit sampling job (no run_dir - entrypoint uses artifact system)
            print(f"  Submitting {s['name']} for {tid} (run_id={rid})")
            future = run_sampler_app(
                cfg_yaml=str(cfg_yaml_path),
                target_id=tid,
                experiment=experiment,
                jax_x64=jax_x64_flag,
                inputs=[target_futures[tid]],  # Dependency: wait for target build
                stdout=str(sample_log_dir / f"{tid}_{s['name']}_{rid}.log"),
                stderr=str(sample_log_dir / f"{tid}_{s['name']}_{rid}.err"),
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

    print(f"\n=== Waiting for {len(run_futures)} sampling runs to complete ===")
    failed_runs = []

    for i, (future, record) in enumerate(zip(run_futures, run_records), 1):
        try:
            future.result()
            print(
                f"  [{i}/{len(run_futures)}] ✓ {record['target_id']}/{record['sampler']}/{record['run_id']}"
            )
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

            print(
                f"  [{i}/{len(run_futures)}] ✗ FAILED: {record['target_id']}/{record['sampler']}/{record['run_id']}"
            )
            print(f"    Stderr:  {stderr_path}")
            print(f"    Stdout:  {stdout_path}")

            if stderr_tail:
                print("    --- Last 15 lines of stderr ---")
                for line in stderr_tail.splitlines():
                    print(f"    {line}")
                print("    --- End stderr ---")

            failed_runs.append({**record, "error": str(e), "stderr_path": str(stderr_path)})

    # Summary of failures
    if failed_runs:
        print(f"\n⚠ FAILURE SUMMARY: {len(failed_runs)} of {len(run_futures)} runs failed")
        for fr in failed_runs:
            print(f"  • {fr['target_id']}/{fr['sampler']}/{fr['run_id']}")
            print(f"    Check logs: {fr['stderr_path']}")

    # ========================================================================
    # Stage C: Promotion (optional, opt-in)
    # ========================================================================

    if enable_promotion:
        print("\n=== Stage C: Promotion ===")
        unique_samplers = sorted({s["name"] for s in samplers_conf})

        # Promotion outputs go to artifacts directory
        outdir = ctx.artifacts_dir / "promotion"

        # Promote each plot type
        for plot_name in promote_plots:
            print(f"  Promoting {plot_name}...")
            promote_future = promote_app(
                store_root=store_root,
                samplers=unique_samplers,
                outdir=str(outdir),
                plot_name=plot_name,
                inputs=run_futures,  # Wait for all runs
            )
            try:
                md_path = promote_future.result()
                print(f"    → Gallery written to {md_path}")
            except Exception as e:
                print(f"    → Promotion FAILED: {e}")
    else:
        print("\n=== Stage C: Promotion skipped (use --promote to enable) ===")

    # ========================================================================
    # Aggregate results into single parquet file
    # ========================================================================

    print("\n=== Aggregating Results ===")
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
                    print(f"  Warning: Failed to read {analysis_path} or {manifest_path}: {e}")
            elif not analysis_path.exists():
                print(f"  Warning: Missing analysis.json at {analysis_path}")

    df = pd.DataFrame(rows)
    output_path = ctx.artifacts_dir / "llc_runs.parquet"

    df.to_parquet(output_path, index=False)
    print(f"\nWrote {len(df)} rows to {output_path}")

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
        help="Override Parsl card values (OmegaConf dotlist), e.g.: --set walltime=04:00:00 --set gpus_per_node=1",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local ThreadPool executor (equivalent to --parsl-card config/parsl/local.yaml)",
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

    # Initialize artifact system early to get RunContext for Parsl run_dir
    from lambda_hat.artifacts import Paths, RunContext

    paths_early = Paths.from_env()
    paths_early.ensure()
    exp_early = OmegaConf.load(args.config)
    experiment_early = args.experiment or exp_early.get("experiment") or "dev"
    ctx_early = RunContext.create(experiment=experiment_early, algo="parsl_llc", paths=paths_early)

    # Resolve Parsl config
    if args.local and not args.parsl_card:
        # Local mode: build config directly from card spec with RunContext run_dir
        print("Using Parsl mode: local (ThreadPool)")
        parsl_cfg = build_parsl_config_from_card(
            OmegaConf.create({"type": "local", "run_dir": str(ctx_early.parsl_dir)})
        )
    elif args.parsl_card:
        # Card-based config with run_dir override
        card_path = Path(args.parsl_card)
        if not card_path.is_absolute():
            card_path = Path.cwd() / card_path
        if not card_path.exists():
            print(f"Error: Parsl card not found: {card_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Using Parsl card: {card_path}")
        # Add run_dir override to parsl_sets
        parsl_sets_with_rundir = (args.parsl_sets or []) + [f"run_dir={ctx_early.parsl_dir}"]
        if args.parsl_sets:
            print(f"  Overrides: {args.parsl_sets}")
        parsl_cfg = load_parsl_config_from_card(card_path, parsl_sets_with_rundir)
    else:
        print("Error: Must specify either --local or --parsl-card", file=sys.stderr)
        sys.exit(1)

    # Parse promote plots
    promote_plots = [p.strip() for p in args.promote_plots.split(",") if p.strip()]

    # Run workflow
    print(f"Using experiments config: {args.config}")
    if args.experiment:
        print(f"Experiment: {args.experiment}")
    if args.promote:
        print(f"Promotion enabled: {promote_plots}")
    else:
        print("Promotion disabled (use --promote to enable)\n")

    # Load Parsl config
    parsl.load(parsl_cfg)

    try:
        output_path = run_workflow(
            args.config,
            experiment=args.experiment,
            enable_promotion=args.promote,
            promote_plots=promote_plots,
        )
        print(f"\n✓ Workflow complete! Results: {output_path}")
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}", file=sys.stderr)
        raise
    finally:
        parsl.clear()


if __name__ == "__main__":
    main()
