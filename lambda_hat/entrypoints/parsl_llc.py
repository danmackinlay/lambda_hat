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
      --parsl-config parsl_config_slurm.py --promote
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import parsl
from omegaconf import OmegaConf
from parsl import bash_app, python_app

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
def build_target_app(cfg_yaml, target_id, target_dir, jax_x64, stdout=None, stderr=None):
    """Build a target (train neural network) via CLI entrypoint.

    Args:
        cfg_yaml: Path to composed build config YAML
        target_id: Target identifier (e.g., 'tgt_abc123')
        target_dir: Output directory for target artifacts
        jax_x64: JAX precision flag (0 or 1)
        stdout: Log file for stdout
        stderr: Log file for stderr

    Returns:
        Shell command string
    """
    return f"""
mkdir -p {target_dir}
JAX_ENABLE_X64={jax_x64} uv run python -m lambda_hat.entrypoints.build_target \
  --config-yaml {cfg_yaml} \
  --target-id {target_id} \
  --target-dir {target_dir}
    """.strip()


@bash_app
def run_sampler_app(cfg_yaml, target_id, run_dir, jax_x64, inputs=None, stdout=None, stderr=None):
    """Run a sampler (MCMC/VI) for a target via CLI entrypoint.

    Args:
        cfg_yaml: Path to composed sample config YAML
        target_id: Target identifier
        run_dir: Output directory for run artifacts
        jax_x64: JAX precision flag (0 or 1)
        inputs: List of futures this task depends on (target build)
        stdout: Log file for stdout
        stderr: Log file for stderr

    Returns:
        Shell command string
    """
    return f"""
mkdir -p {run_dir}
JAX_ENABLE_X64={jax_x64} uv run python -m lambda_hat.entrypoints.sample \
  --config-yaml {cfg_yaml} \
  --target-id {target_id} \
  --run-dir {run_dir}
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


def load_parsl_config(config_path):
    """Dynamically load a Parsl config module.

    Args:
        config_path: Path to Python file containing 'config' variable

    Returns:
        Parsl Config object
    """
    spec = importlib.util.spec_from_file_location("parsl_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def run_workflow(
    experiments_yaml,
    parsl_config_path,
    enable_promotion=False,
    promote_plots=None,
    logs_dir="logs",
    temp_cfg_dir="temp_parsl_cfg",
    results_dir="results",
):
    """Execute the full Lambda-Hat workflow: build → sample → (optional) promote.

    Args:
        experiments_yaml: Path to experiments config (e.g., config/experiments.yaml)
        parsl_config_path: Path to Parsl executor config
        enable_promotion: Whether to run promotion stage (default: False, opt-in)
        promote_plots: List of plot names to promote (default: ['trace.png'])
        logs_dir: Directory for log files (default: "logs")
        temp_cfg_dir: Directory for temporary config files (default: "temp_parsl_cfg")
        results_dir: Directory for aggregated results (default: "results")

    Returns:
        Path to aggregated results parquet file
    """
    if promote_plots is None:
        promote_plots = ["trace.png", "llc_convergence_combined.png"]

    # Convert paths to Path objects
    logs_dir = Path(logs_dir)
    temp_cfg_dir = Path(temp_cfg_dir)
    results_dir = Path(results_dir)

    # Load Parsl config and initialize
    parsl_cfg = load_parsl_config(parsl_config_path)
    parsl.load(parsl_cfg)

    # Load experiment configuration
    exp = OmegaConf.load(experiments_yaml)
    store_root = exp.get("store_root", "runs")
    jax_x64 = bool(exp.get("jax_enable_x64", True))
    jax_x64_flag = 1 if jax_x64 else 0

    targets_conf = list(exp["targets"])
    samplers_conf = list(exp["samplers"])

    print(f"Loaded {len(targets_conf)} targets and {len(samplers_conf)} samplers")
    print(f"Store root: {store_root}, JAX x64: {jax_x64}")
    print(f"Logs: {logs_dir}, Temp configs: {temp_cfg_dir}, Results: {results_dir}")

    # Create temp config directory
    temp_cfg_dir.mkdir(exist_ok=True, parents=True)

    # ========================================================================
    # Stage A: Build Targets
    # ========================================================================

    print("\n=== Stage A: Building Targets ===")
    target_futures = {}
    target_ids = []

    for t in targets_conf:
        # Compose build config and compute target ID
        build_cfg = compose_build_cfg(t, store_root=store_root, jax_enable_x64=jax_x64)
        tid = target_id_for(build_cfg)
        target_ids.append(tid)

        # Write temp config YAML
        cfg_yaml_path = temp_cfg_dir / f"build_{tid}.yaml"
        cfg_yaml_path.write_text(OmegaConf.to_yaml(build_cfg))

        # Prepare target directory
        target_dir = Path(store_root) / "targets" / tid
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create log directory
        log_dir = logs_dir / "build_target"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Submit build job
        print(f"  Submitting build for {tid} (model={t['model']}, data={t['data']})")
        future = build_target_app(
            cfg_yaml=str(cfg_yaml_path),
            target_id=tid,
            target_dir=str(target_dir),
            jax_x64=jax_x64_flag,
            stdout=str(log_dir / f"{tid}.log"),
            stderr=str(log_dir / f"{tid}.err"),
        )
        target_futures[tid] = future

    # ========================================================================
    # Stage B: Run Samplers
    # ========================================================================

    print("\n=== Stage B: Running Samplers ===")
    run_futures = []
    run_records = []

    for tid in target_ids:
        for s in samplers_conf:
            # Compose sample config and compute run ID
            sample_cfg = compose_sample_cfg(tid, s, store_root=store_root, jax_enable_x64=jax_x64)
            rid = run_id_for(sample_cfg)

            # Write temp config YAML
            cfg_yaml_path = temp_cfg_dir / f"sample_{tid}_{s['name']}_{rid}.yaml"
            cfg_yaml_path.write_text(OmegaConf.to_yaml(sample_cfg))

            # Prepare run directory
            run_dir = Path(store_root) / "targets" / tid / f"run_{s['name']}_{rid}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create log directory
            log_dir = logs_dir / "run_sampler"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Submit sampling job with dependency on target build
            print(f"  Submitting {s['name']} for {tid} (run_id={rid})")
            future = run_sampler_app(
                cfg_yaml=str(cfg_yaml_path),
                target_id=tid,
                run_dir=str(run_dir),
                jax_x64=jax_x64_flag,
                inputs=[target_futures[tid]],  # Dependency: wait for target build
                stdout=str(log_dir / f"{tid}_{s['name']}_{rid}.log"),
                stderr=str(log_dir / f"{tid}_{s['name']}_{rid}.err"),
            )
            run_futures.append(future)

            # Record metadata for aggregation
            run_records.append(
                {
                    "target_id": tid,
                    "sampler": s["name"],
                    "run_id": rid,
                    "run_dir": str(run_dir),
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
                logs_dir
                / "run_sampler"
                / f"{record['target_id']}_{record['sampler']}_{record['run_id']}.err"
            )
            stdout_path = (
                logs_dir
                / "run_sampler"
                / f"{record['target_id']}_{record['sampler']}_{record['run_id']}.log"
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
            print(f"    Run dir: {record['run_dir']}")
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

        # Load promotion config for output directory
        from lambda_hat.conf import promote as promote_conf_module

        prom_cfg = OmegaConf.create(promote_conf_module.__dict__)
        outdir = Path(prom_cfg.get("outdir", "runs/promotion"))

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
    for rec in run_records:
        analysis_path = Path(rec["run_dir"]) / "analysis.json"
        if analysis_path.exists():
            try:
                metrics = json.loads(analysis_path.read_text())
                rows.append({**rec, **metrics})
            except Exception as e:
                print(f"  Warning: Failed to read {analysis_path}: {e}")
        else:
            print(f"  Warning: Missing analysis.json at {analysis_path}")

    df = pd.DataFrame(rows)
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "llc_runs.parquet"

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
        "--parsl-config",
        default="parsl_config_slurm.py",
        help="Path to Parsl executor config (default: parsl_config_slurm.py)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local ThreadPool executor (overrides --parsl-config)",
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
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory for log files (default: logs, relative to CWD)",
    )
    parser.add_argument(
        "--temp-cfg-dir",
        default="temp_parsl_cfg",
        help="Directory for temporary config files (default: temp_parsl_cfg, relative to CWD)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for aggregated results (default: results, relative to CWD)",
    )

    args = parser.parse_args()

    # Resolve config path
    cwd = Path.cwd()
    if args.local:
        parsl_config_path = cwd / "parsl_config_local.py"
    else:
        parsl_config_path = Path(args.parsl_config)
        if not parsl_config_path.is_absolute():
            parsl_config_path = cwd / parsl_config_path

    if not parsl_config_path.exists():
        print(f"Error: Parsl config not found: {parsl_config_path}", file=sys.stderr)
        sys.exit(1)

    # Parse promote plots
    promote_plots = [p.strip() for p in args.promote_plots.split(",") if p.strip()]

    # Run workflow
    print(f"Using Parsl config: {parsl_config_path}")
    print(f"Using experiments config: {args.config}")
    if args.promote:
        print(f"Promotion enabled: {promote_plots}")
    else:
        print("Promotion disabled (use --promote to enable)\n")

    try:
        output_path = run_workflow(
            args.config,
            parsl_config_path,
            enable_promotion=args.promote,
            promote_plots=promote_plots,
            logs_dir=args.logs_dir,
            temp_cfg_dir=args.temp_cfg_dir,
            results_dir=args.results_dir,
        )
        print(f"\n✓ Workflow complete! Results: {output_path}")
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}", file=sys.stderr)
        raise
    finally:
        parsl.clear()


if __name__ == "__main__":
    main()
