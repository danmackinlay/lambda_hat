#!/usr/bin/env python3
"""Optuna + Parsl orchestrator for hyperparameter optimization.

Bayesian optimization workflow that:
1. Computes HMC reference LLC estimates for N problems
2. Optimizes method hyperparameters (SGLD/VI/MCLMC) to minimize |LLC - LLC_ref|
3. Uses Optuna's ask-and-tell API with Parsl for parallel trial execution
4. Aggregates all trials into a single results parquet file

Usage:
  # Local testing
  parsl-optuna --config config/optuna_demo.yaml --local

  # SLURM cluster
  parsl-optuna --config config/optuna_demo.yaml \\
      --parsl-card config/parsl/slurm/cpu.yaml

See plans/optuna.md for design details.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import optuna
import pandas as pd
import parsl
from omegaconf import OmegaConf

from lambda_hat.id_utils import problem_id, trial_id
from lambda_hat.parsl_cards import build_parsl_config_from_card, load_parsl_config_from_card
from lambda_hat.runners.parsl_apps import compute_hmc_reference, run_method_trial


def huber_loss(x, delta=0.1):
    """Huber loss for robust objective.

    Args:
        x: Error value
        delta: Transition point between quadratic and linear (default: 0.1)

    Returns:
        float: Huber loss
    """
    ax = abs(x)
    return 0.5 * x * x / delta if ax <= delta else ax - 0.5 * delta


def objective_from_metrics(llc_hat, llc_ref, huber_delta=None):
    """Compute objective for Optuna from LLC estimates.

    Args:
        llc_hat: Estimated LLC from method
        llc_ref: Reference LLC from HMC
        huber_delta: Huber delta (None = absolute error)

    Returns:
        float: Objective value to minimize
    """
    diff = llc_hat - llc_ref
    return huber_loss(diff, huber_delta) if huber_delta else abs(diff)


def suggest_method_params(trial, method_name):
    """Suggest hyperparameters for a method using Optuna trial.

    Args:
        trial: Optuna Trial object
        method_name: Method name ("sgld", "vi", "mclmc")

    Returns:
        dict: Hyperparameters for the method
    """
    if method_name == "sgld":
        return {
            "eta0": trial.suggest_float("eta0", 1e-6, 1e-1, log=True),
            "gamma": trial.suggest_float("gamma", 0.3, 1.0),
            "batch": trial.suggest_categorical("batch", [32, 64, 128, 256]),
            "precond_type": trial.suggest_categorical("precond_type", ["rmsprop", "adam"]),
            "steps": trial.suggest_int("steps", 5000, 20000, step=1000),
        }
    elif method_name == "vi":
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-2, log=True),
            "M": trial.suggest_categorical("M", [4, 8, 16]),
            "r": trial.suggest_categorical("r", [1, 2, 4]),
            "whitening_mode": trial.suggest_categorical(
                "whitening_mode", ["none", "rmsprop", "adam"]
            ),
            "steps": trial.suggest_int("steps", 3000, 10000, step=500),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
    elif method_name == "mclmc":
        return {
            "step_size": trial.suggest_float("step_size", 1e-5, 1e-1, log=True),
            "target_accept": trial.suggest_float("target_accept", 0.5, 0.9),
            "L": trial.suggest_float("L", 0.5, 2.0),
            "steps": trial.suggest_int("steps", 5000, 20000, step=1000),
        }
    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_optuna_workflow(
    config_path,
    parsl_config_path=None,  # Deprecated, kept for compatibility
    max_trials_per_method=200,
    batch_size=32,
    hmc_budget_sec=36000,
    method_budget_sec=6000,
    artifacts_dir="artifacts",
    results_dir="results",
):
    """Execute Optuna hyperparameter optimization workflow.

    Args:
        config_path: Path to Optuna experiment config
        parsl_config_path: [DEPRECATED] Ignored (Parsl config loaded via main())

    Args:
        config_path: Path to Optuna config YAML (problem specs)
        parsl_config_path: Path to Parsl executor config
        max_trials_per_method: Maximum trials per (problem, method) (default: 200)
        batch_size: Concurrent trials per (problem, method) (default: 32)
        hmc_budget_sec: HMC reference time budget (default: 36000 = 10h)
        method_budget_sec: Method trial time budget (default: 6000 = 100min)
        artifacts_dir: Directory for artifacts (default: "artifacts", relative to CWD)
        results_dir: Directory for results (default: "results", relative to CWD)

    Returns:
        Path to results parquet file
    """
    # Convert paths to Path objects
    artifacts_dir = Path(artifacts_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load experiment configuration (Parsl config already loaded in main())
    print(f"Loading experiment config from {config_path}...")
    exp = OmegaConf.load(config_path)

    # Extract problems and methods
    problems = list(exp.get("problems", []))
    methods = list(exp.get("methods", ["sgld", "vi", "mclmc"]))

    print("\n=== Optuna Workflow Configuration ===")
    print(f"Problems: {len(problems)}")
    print(f"Methods: {methods}")
    print(f"Max trials per (problem, method): {max_trials_per_method}")
    print(f"Batch size (concurrent trials): {batch_size}")
    print(f"HMC budget: {hmc_budget_sec}s ({hmc_budget_sec / 3600:.1f}h)")
    print(f"Method budget: {method_budget_sec}s ({method_budget_sec / 60:.1f}min)")
    print(f"Artifacts: {artifacts_dir}, Results: {results_dir}")
    print()

    # ========================================================================
    # Stage 1: Compute HMC References
    # ========================================================================

    print("=== Stage 1: Computing HMC References ===")
    ref_futs = {}
    ref_meta = {}  # pid -> {llc_ref, se_ref, ...}

    for p in problems:
        # Normalize problem spec to dict
        problem_spec = OmegaConf.to_container(p, resolve=True)
        pid = problem_id(problem_spec)
        out_ref = artifacts_dir / "problems" / pid / "ref.json"
        out_ref.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Problem {pid}:")
        print(f"    Spec: {problem_spec}")

        # Check if reference already exists
        if out_ref.exists():
            print(f"    Reference exists, loading from {out_ref}")
            ref_meta[pid] = json.loads(out_ref.read_text())
        else:
            print("    Submitting HMC reference computation...")
            ref_futs[pid] = compute_hmc_reference(
                problem_spec, str(out_ref), budget_sec=hmc_budget_sec
            )

    # Wait for missing references to complete
    print(f"\n  Waiting for {len(ref_futs)} HMC references to complete...")
    for pid, fut in ref_futs.items():
        try:
            ref_meta[pid] = fut.result()
            print(f"    ✓ {pid}: LLC_ref = {ref_meta[pid]['llc_ref']:.4f}")
        except Exception as e:
            print(f"    ✗ {pid}: FAILED - {e}")
            raise

    print(f"\n  All {len(ref_meta)} HMC references ready")

    # ========================================================================
    # Stage 2: Optuna Optimization (ask-and-tell loop)
    # ========================================================================

    print("\n=== Stage 2: Hyperparameter Optimization ===")

    # Storage for Optuna studies (in-memory + periodic pickle)
    study_dir = results_dir / "studies" / "optuna_llc"
    study_dir.mkdir(parents=True, exist_ok=True)

    def save_study(study, path):
        """Save Optuna study to pickle (for resume)."""
        path.write_bytes(pickle.dumps(study))

    # Results accumulator
    all_rows = []

    # For each problem × method, run Optuna optimization
    for p in problems:
        problem_spec = OmegaConf.to_container(p, resolve=True)
        pid = problem_id(problem_spec)
        llc_ref = float(ref_meta[pid]["llc_ref"])

        print(f"\n  Problem {pid} (LLC_ref = {llc_ref:.4f})")

        for method_name in methods:
            print(f"    Method: {method_name}")

            # Create Optuna study
            study_name = f"{pid}:{method_name}"
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(seed=42),
            )

            # In-flight trials tracking
            inflight = {}  # future -> (trial, trial_id, run_dir)
            submitted = 0

            def submit_one():
                """Submit one trial to Parsl."""
                nonlocal submitted

                # Ask Optuna for hyperparameters
                t = study.ask()
                hp = suggest_method_params(t, method_name)

                # Create trial manifest
                manifest = {
                    "pid": pid,
                    "method": method_name,
                    "hyperparams": hp,
                    "seed": int(t.number),
                    "budget_sec": method_budget_sec,
                }
                tid = trial_id(manifest)

                # Prepare run directory
                run_dir = artifacts_dir / "runs" / pid / method_name / tid
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

                # Build method config
                method_cfg = {"name": method_name, **hp}

                # Submit Parsl app
                fut = run_method_trial(
                    problem_spec,
                    method_cfg,
                    llc_ref,
                    str(run_dir / "metrics.json"),
                    budget_sec=method_budget_sec,
                    seed=int(t.number),
                )

                inflight[fut] = (t, tid, run_dir)
                submitted += 1

                return fut

            # Prime the pump: fill initial batch
            print(
                f"      Submitting initial batch of {min(batch_size, max_trials_per_method)} trials..."
            )
            while submitted < min(batch_size, max_trials_per_method):
                submit_one()

            # Main loop: process completions and refill batch
            print(f"      Running optimization loop (max {max_trials_per_method} trials)...")
            while len(study.trials) < max_trials_per_method:
                # Check for completed futures
                done = [f for f in list(inflight.keys()) if f.done()]

                if not done:
                    time.sleep(1)
                    continue

                # Process completed trials
                for f in done:
                    t, tid, run_dir = inflight.pop(f)

                    try:
                        result = f.result()  # dict: {llc_hat, se_hat, runtime_sec, ...}
                    except Exception as e:
                        # Penalize crashed trials with large objective
                        print(f"        Trial {tid} FAILED: {e}")
                        study.tell(t, float("inf"))
                        continue

                    # Extract LLC and compute objective
                    llc_hat = float(result["llc_hat"])
                    obj = objective_from_metrics(llc_hat, llc_ref, huber_delta=None)

                    # Persist metrics with objective
                    metrics = {
                        **result,
                        "objective": obj,
                        "llc_ref": llc_ref,
                        "pid": pid,
                        "method": method_name,
                        "trial_id": tid,
                    }
                    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

                    # Tell Optuna
                    study.tell(t, obj)

                    # Add to results
                    all_rows.append(metrics)

                    # Log progress
                    error_pct = abs(llc_hat - llc_ref) / llc_ref * 100
                    print(
                        f"        Trial {len(study.trials)}/{max_trials_per_method}: "
                        f"LLC={llc_hat:.4f}, error={error_pct:.1f}%, obj={obj:.4f}"
                    )

                    # Refill batch if under budget
                    if submitted < max_trials_per_method:
                        submit_one()

                # Periodic study checkpoint
                if len(study.trials) % 10 == 0:
                    save_study(study, study_dir / f"{study_name}.pkl")

            # Final study save
            save_study(study, study_dir / f"{study_name}.pkl")

            # Report best trial
            best_trial = study.best_trial
            if best_trial.value == float("inf"):
                print("      ⚠ All trials failed - no valid hyperparameters found")
            else:
                print(f"      ✓ Best trial: obj={best_trial.value:.4f}")
                print(f"        Hyperparams: {best_trial.params}")

    # ========================================================================
    # Stage 3: Aggregate Results
    # ========================================================================

    print("\n=== Stage 3: Aggregating Results ===")
    df = pd.DataFrame(all_rows)
    output_path = results_dir / "optuna_trials.parquet"
    df.to_parquet(output_path, index=False)

    print(f"  Wrote {len(df)} trials to {output_path}")
    print("\n✓ Optuna workflow complete!")

    return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization with Parsl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config/optuna_demo.yaml",
        help="Path to Optuna experiment config (default: config/optuna_demo.yaml)",
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
        help="Override Parsl card values (OmegaConf dotlist), e.g.: --set walltime=04:00:00",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local ThreadPool executor (equivalent to --parsl-card config/parsl/local.yaml)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=200,
        help="Maximum trials per (problem, method) (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Concurrent trials per (problem, method) (default: 32)",
    )
    parser.add_argument(
        "--hmc-budget",
        type=int,
        default=36000,
        help="HMC reference time budget in seconds (default: 36000 = 10h)",
    )
    parser.add_argument(
        "--method-budget",
        type=int,
        default=6000,
        help="Method trial time budget in seconds (default: 6000 = 100min)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for artifacts (default: artifacts, relative to CWD)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for results (default: results, relative to CWD)",
    )

    args = parser.parse_args()

    # Resolve Parsl config
    if args.local and not args.parsl_card:
        print("Using Parsl mode: local (ThreadPool)")
        parsl_cfg = build_parsl_config_from_card(OmegaConf.create({"type": "local"}))
    elif args.parsl_card:
        card_path = Path(args.parsl_card)
        if not card_path.is_absolute():
            card_path = Path.cwd() / card_path
        if not card_path.exists():
            print(f"Error: Parsl card not found: {card_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Using Parsl card: {card_path}")
        if args.parsl_sets:
            print(f"  Overrides: {args.parsl_sets}")
        parsl_cfg = load_parsl_config_from_card(card_path, args.parsl_sets)
    else:
        print("Error: Must specify either --local or --parsl-card", file=sys.stderr)
        sys.exit(1)

    # Run workflow
    print(f"Using Optuna config: {args.config}\n")

    # Load Parsl config
    parsl.load(parsl_cfg)

    try:
        output_path = run_optuna_workflow(
            args.config,
            None,  # No longer pass parsl_config_path
            max_trials_per_method=args.max_trials,
            batch_size=args.batch_size,
            hmc_budget_sec=args.hmc_budget,
            method_budget_sec=args.method_budget,
            artifacts_dir=args.artifacts_dir,
            results_dir=args.results_dir,
        )
        print(f"\n✓ Results: {output_path}")
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}", file=sys.stderr)
        raise
    finally:
        parsl.clear()


if __name__ == "__main__":
    main()
