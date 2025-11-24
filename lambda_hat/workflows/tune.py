#!/usr/bin/env python3
"""Optuna + Parsl orchestrator for hyperparameter optimization.

Bayesian optimization workflow that:
1. Computes HMC reference LLC estimates for N problems
2. Optimizes method hyperparameters (SGLD/VI/MCLMC) to minimize |LLC - LLC_ref|
3. Uses Optuna's ask-and-tell API with Parsl for parallel trial execution
4. Aggregates all trials into a single results parquet file

Usage:
  # Local testing
  lambda-hat workflow optuna --config config/optuna/default.yaml --backend local

  # SLURM cluster
  lambda-hat workflow optuna --config config/optuna/default.yaml \\
      --parsl-card config/parsl/slurm/cpu.yaml

See plans/optuna.md for design details.
"""

import json
import logging
import pickle
import time

import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from lambda_hat.artifacts import Paths, RunContext
from lambda_hat.id_utils import problem_id, trial_id
from lambda_hat.runners.parsl_apps import compute_hmc_reference, run_method_trial

log = logging.getLogger(__name__)


def _suggest(trial, name: str, spec: dict):
    """Suggest a single hyperparameter from YAML spec.

    Args:
        trial: Optuna Trial object
        name: Parameter name
        spec: Parameter spec dict from YAML (dist, low, high, choices, etc.)

    Returns:
        Suggested value

    Raises:
        ValueError: If distribution type is unknown
    """
    dist = spec["dist"]

    if dist == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    elif dist == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step"))
    elif dist == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    else:
        raise ValueError(f"Unknown distribution type: {dist}")


def suggest_params_from_yaml(trial, space_dict: dict) -> dict:
    """Suggest all hyperparameters for a method from YAML search space.

    Args:
        trial: Optuna Trial object
        space_dict: Search space dict from cfg.search_space[method_name]

    Returns:
        Dict of hyperparameter name -> suggested value
    """
    return {name: _suggest(trial, name, dict(spec)) for name, spec in space_dict.items()}


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


def objective_from_metrics(llc_hat, llc_ref, objective_cfg):
    """Compute objective for Optuna from LLC estimates.

    Args:
        llc_hat: Estimated LLC from method
        llc_ref: Reference LLC from HMC
        objective_cfg: Objective config dict from cfg.optuna.objective

    Returns:
        float: Objective value to minimize
    """
    diff = llc_hat - llc_ref
    obj_type = objective_cfg.get("type", "abs")

    if obj_type == "abs":
        return abs(diff)
    elif obj_type == "huber":
        delta = objective_cfg.get("delta", 0.1)
        return huber_loss(diff, delta)
    else:
        raise ValueError(f"Unknown objective type: {obj_type}")


def run_optuna_workflow(cfg: DictConfig):
    """Execute Optuna hyperparameter optimization workflow.

    Args:
        cfg: Resolved Optuna configuration (from load_cfg)

    Returns:
        Path to results parquet file

    Side effects:
        - Creates RunContext directories under cfg.store.root
        - Writes resolved config to meta/resolved_config.yaml
        - Persists HMC references, trial manifests, metrics, study pickles
        - Aggregates trials to tables/optuna_trials.parquet
    """
    # Initialize artifact system with RunContext
    paths = Paths.from_env()
    paths.ensure()

    experiment = cfg.store.get("namespace", "optuna")
    ctx = RunContext.create(experiment=experiment, algo="optuna_workflow", paths=paths)

    log.info("Run dir: %s", ctx.run_dir)
    log.info("Artifacts: %s", ctx.artifacts_dir)
    log.info("Logs: %s", ctx.logs_dir)

    # Persist resolved config
    meta_dir = ctx.run_dir / "meta"
    meta_dir.mkdir(exist_ok=True, parents=True)
    resolved_cfg_path = meta_dir / "resolved_config.yaml"
    resolved_cfg_path.write_text(OmegaConf.to_yaml(cfg))
    log.info("Resolved config: %s", resolved_cfg_path)

    # Extract configuration
    problems = list(cfg.get("problems", []))
    methods = list(cfg.get("methods", []))
    max_trials_per_method = cfg.optuna.max_trials_per_method
    batch_size = cfg.optuna.concurrency.batch_size
    hmc_budget_sec = cfg.execution.budget.hmc_sec
    trial_budget_sec = cfg.execution.budget.trial_sec

    log.info("=== Optuna Workflow Configuration ===")
    log.info("Problems: %d", len(problems))
    log.info("Methods: %s", methods)
    log.info("Max trials per (problem, method): %d", max_trials_per_method)
    log.info("Batch size (concurrent trials): %d", batch_size)
    log.info("HMC budget: %ds (%.1fh)", hmc_budget_sec, hmc_budget_sec / 3600)
    log.info("Method budget: %ds (%.1fmin)", trial_budget_sec, trial_budget_sec / 60)

    # ========================================================================
    # Stage 1: Compute HMC References
    # ========================================================================

    log.info("=== Stage 1: Computing HMC References ===")
    ref_futs = {}
    ref_meta = {}  # pid -> {llc_ref, se_ref, ...}

    problems_dir = ctx.artifacts_dir / "problems"
    problems_dir.mkdir(exist_ok=True, parents=True)

    for p in problems:
        # Normalize problem spec to dict
        problem_spec = OmegaConf.to_container(p, resolve=True)
        pid = problem_id(problem_spec)
        out_ref = problems_dir / pid / "ref.json"
        out_ref.parent.mkdir(parents=True, exist_ok=True)

        log.info("  Problem %s:", pid)
        log.info("    Spec: %s", problem_spec)

        # Check if reference already exists
        if out_ref.exists():
            log.info("    Reference exists, loading from %s", out_ref)
            ref_meta[pid] = json.loads(out_ref.read_text())
        else:
            log.info("    Submitting HMC reference computation")
            ref_futs[pid] = compute_hmc_reference(
                problem_spec, str(out_ref), budget_sec=hmc_budget_sec
            )

    # Wait for missing references to complete
    log.info("  Waiting for %d HMC references to complete...", len(ref_futs))
    for pid, fut in ref_futs.items():
        try:
            ref_meta[pid] = fut.result()
            log.info("    ✓ %s: LLC_ref = %.4f", pid, ref_meta[pid]["llc_ref"])
        except Exception as e:
            log.error("    ✗ %s: FAILED - %s", pid, e)
            raise

    log.info("  All %d HMC references ready", len(ref_meta))

    # ========================================================================
    # Stage 2: Optuna Optimization (ask-and-tell loop)
    # ========================================================================

    log.info("=== Stage 2: Hyperparameter Optimization ===")

    # Storage for Optuna studies (in-memory + periodic pickle)
    study_dir = ctx.run_dir / "studies"
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

        log.info("  Problem %s (LLC_ref = %.4f)", pid, llc_ref)

        for method_name in methods:
            log.info("    Method: %s", method_name)

            # Get search space for this method
            method_space = cfg.search_space[method_name]

            # Create Optuna study
            study_name = f"{pid}:{method_name}"

            # Build sampler from config
            sampler_cfg = cfg.optuna.get("sampler", {})
            sampler_type = sampler_cfg.get("type", "tpe")
            if sampler_type == "tpe":
                sampler = optuna.samplers.TPESampler(
                    seed=sampler_cfg.get("seed", 42),
                    n_startup_trials=sampler_cfg.get("n_startup_trials", 20),
                )
            else:
                raise ValueError(f"Unknown sampler type: {sampler_type}")

            # Build pruner from config
            pruner_cfg = cfg.optuna.get("pruner", {})
            pruner_type = pruner_cfg.get("type", "median")
            if pruner_type == "median":
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=pruner_cfg.get("n_startup_trials", 10),
                    n_warmup_steps=pruner_cfg.get("n_warmup_steps", 1),
                )
            elif pruner_type == "none":
                pruner = optuna.pruners.NopPruner()
            else:
                raise ValueError(f"Unknown pruner type: {pruner_type}")

            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                sampler=sampler,
                pruner=pruner,
            )

            # In-flight trials tracking
            inflight = {}  # future -> (trial, trial_id, run_dir)
            submitted = 0

            def submit_one():
                """Submit one trial to Parsl."""
                nonlocal submitted

                # Ask Optuna for hyperparameters
                t = study.ask()
                hp = suggest_params_from_yaml(t, method_space)

                # Create trial manifest
                manifest = {
                    "pid": pid,
                    "method": method_name,
                    "hyperparams": hp,
                    "seed": int(t.number),
                    "budget_sec": trial_budget_sec,
                }
                tid = trial_id(manifest)

                # Prepare run directory
                trials_dir = ctx.artifacts_dir / "trials"
                run_dir = trials_dir / pid / method_name / tid
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
                    budget_sec=trial_budget_sec,
                    seed=int(t.number),
                )

                inflight[fut] = (t, tid, run_dir)
                submitted += 1

                return fut

            # Prime the pump: fill initial batch
            initial_batch = min(batch_size, max_trials_per_method)
            log.info("      Submitting initial batch of %d trials...", initial_batch)
            while submitted < initial_batch:
                submit_one()

            # Main loop: process completions and refill batch
            log.info("      Running optimization loop (max %d trials)...", max_trials_per_method)
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
                        log.error("        Trial %s FAILED: %s", tid, e)
                        study.tell(t, float("inf"))
                        continue

                    # Extract LLC and compute objective
                    llc_hat = float(result["llc_hat"])
                    obj = objective_from_metrics(llc_hat, llc_ref, cfg.optuna.objective)

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
                    log.info(
                        "        Trial %d/%d: LLC=%.4f, error=%.1f%%, obj=%.4f",
                        len(study.trials),
                        max_trials_per_method,
                        llc_hat,
                        error_pct,
                        obj,
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
                log.warning("      ⚠ All trials failed - no valid hyperparameters found")
            else:
                log.info("      ✓ Best trial: obj=%.4f", best_trial.value)
                log.info("        Hyperparams: %s", best_trial.params)

    # ========================================================================
    # Stage 3: Aggregate Results
    # ========================================================================

    log.info("=== Stage 3: Aggregating Results ===")

    tables_dir = ctx.run_dir / "tables"
    tables_dir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(all_rows)
    output_path = tables_dir / "optuna_trials.parquet"
    df.to_parquet(output_path, index=False)

    log.info("  Wrote %d trials to %s", len(df), output_path)
    log.info("✓ Optuna workflow complete!")

    return output_path
