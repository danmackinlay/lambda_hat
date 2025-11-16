"""Integration tests for Optuna hyperparameter optimization workflow."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent

from lambda_hat.entrypoints.parsl_optuna import run_optuna_workflow


@pytest.fixture
def optuna_test_config():
    """Path to minimal Optuna test configuration."""
    return ROOT / "tests" / "test_optuna_config.yaml"


@pytest.fixture
def parsl_config():
    """Path to local Parsl configuration."""
    return ROOT / "parsl_config_local.py"


@pytest.fixture
def cleanup_optuna_artifacts():
    """Cleanup Optuna test artifacts after test completes."""
    yield
    # Clear Parsl to allow next test to load config
    import parsl
    try:
        parsl.clear()
    except Exception:
        pass  # Ignore if not loaded

    # Cleanup paths
    cleanup_paths = [
        ROOT / "artifacts",
        ROOT / "results" / "optuna_trials.parquet",
        ROOT / "results" / "studies",
        ROOT / "logs",
        ROOT / "parsl_runinfo",
        ROOT / "temp_parsl_cfg",
    ]

    for path in cleanup_paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def test_optuna_workflow_integration(optuna_test_config, parsl_config, cleanup_optuna_artifacts):
    """Test end-to-end Optuna workflow with minimal config.

    This test:
    1. Runs HMC reference for test problem
    2. Runs 3 VI trials with Optuna optimization
    3. Validates results structure and artifacts
    """
    # Run workflow
    output_path = run_optuna_workflow(
        optuna_test_config,
        parsl_config,
        max_trials_per_method=3,
        batch_size=2,
        hmc_budget_sec=180,  # 3 minutes
        method_budget_sec=60,  # 1 minute
    )

    # ========================================================================
    # Verify results parquet
    # ========================================================================

    assert output_path.exists(), "Results parquet file should exist"
    assert output_path == ROOT / "results" / "optuna_trials.parquet"

    df = pd.read_parquet(output_path)

    # Check DataFrame structure
    required_columns = {
        "pid",
        "method",
        "trial_id",
        "llc_hat",
        "llc_ref",
        "objective",
        "error",
        "runtime_sec",
        "hyperparams",
        "diagnostics",
        "L0",
        "beta",
        "gamma",
    }
    assert required_columns.issubset(
        set(df.columns)
    ), f"Missing columns: {required_columns - set(df.columns)}"

    # Check number of rows (1 problem × 1 method × 3 trials = 3)
    assert len(df) == 3, f"Expected 3 trials, got {len(df)}"

    # Check all trials are for VI
    assert (df["method"] == "vi").all(), "All trials should be for VI method"

    # Check that at least one trial has finite objective
    finite_trials = df[df["objective"] != float("inf")]
    assert (
        len(finite_trials) > 0
    ), "At least one trial should have finite objective (not all failed)"

    # ========================================================================
    # Verify HMC reference artifacts
    # ========================================================================

    # Extract problem ID from results
    pid = df.iloc[0]["pid"]
    assert pid.startswith("p_"), f"Problem ID should start with 'p_', got {pid}"

    ref_path = ROOT / "artifacts" / "problems" / pid / "ref.json"
    assert ref_path.exists(), f"HMC reference should exist at {ref_path}"

    # Check reference structure
    with open(ref_path) as f:
        ref = json.load(f)

    required_ref_keys = {"llc_ref", "se_ref", "diagnostics", "runtime_sec", "L0", "beta", "gamma"}
    assert required_ref_keys.issubset(
        set(ref.keys())
    ), f"Missing reference keys: {required_ref_keys - set(ref.keys())}"

    assert isinstance(ref["llc_ref"], (int, float)), "llc_ref should be a number"
    assert ref["llc_ref"] > 0, "llc_ref should be positive"

    # ========================================================================
    # Verify trial artifacts
    # ========================================================================

    for _, row in df.iterrows():
        trial_id = row["trial_id"]
        method = row["method"]

        # Check manifest.json
        manifest_path = ROOT / "artifacts" / "runs" / pid / method / trial_id / "manifest.json"
        assert manifest_path.exists(), f"Manifest should exist for trial {trial_id}"

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["pid"] == pid
        assert manifest["method"] == method
        assert "hyperparams" in manifest
        assert "seed" in manifest
        assert "budget_sec" in manifest

        # Check metrics.json
        metrics_path = ROOT / "artifacts" / "runs" / pid / method / trial_id / "metrics.json"
        assert metrics_path.exists(), f"Metrics should exist for trial {trial_id}"

        with open(metrics_path) as f:
            metrics = json.load(f)

        assert "llc_hat" in metrics
        assert "objective" in metrics
        assert "llc_ref" in metrics
        assert metrics["llc_ref"] == ref["llc_ref"]  # Should match HMC reference

    # ========================================================================
    # Verify Optuna study
    # ========================================================================

    study_path = ROOT / "results" / "studies" / "optuna_llc" / f"{pid}:vi.pkl"
    assert study_path.exists(), f"Optuna study pickle should exist at {study_path}"

    # Study should be loadable
    import pickle

    with open(study_path, "rb") as f:
        study = pickle.load(f)

    assert len(study.trials) == 3, "Study should have 3 trials"
    assert study.direction.name == "MINIMIZE", "Study should minimize objective"

    # Best trial should have finite value (if any trials succeeded)
    if len(finite_trials) > 0:
        assert study.best_trial.value < float("inf"), "Best trial should have finite value"
        assert study.best_trial.value >= 0, "Best objective should be non-negative"


def test_optuna_hmc_caching(optuna_test_config, parsl_config, cleanup_optuna_artifacts):
    """Test that HMC reference is cached and reused across runs."""
    # First run
    output_path1 = run_optuna_workflow(
        optuna_test_config,
        parsl_config,
        max_trials_per_method=2,  # Fewer trials for speed
        batch_size=1,
        hmc_budget_sec=120,
        method_budget_sec=30,
    )

    df1 = pd.read_parquet(output_path1)
    pid = df1.iloc[0]["pid"]
    ref_path = ROOT / "artifacts" / "problems" / pid / "ref.json"

    # Get modification time of reference
    import time

    ref_mtime1 = ref_path.stat().st_mtime

    # Wait a bit to ensure different timestamp if recreated
    time.sleep(1)

    # Second run (should reuse cached reference)
    output_path2 = run_optuna_workflow(
        optuna_test_config,
        parsl_config,
        max_trials_per_method=2,
        batch_size=1,
        hmc_budget_sec=120,
        method_budget_sec=30,
    )

    # Reference should still exist
    assert ref_path.exists()

    # Modification time should be unchanged (reference was cached)
    ref_mtime2 = ref_path.stat().st_mtime
    assert (
        ref_mtime2 == ref_mtime1
    ), "HMC reference should be reused (same modification time), not recomputed"

    # Both runs should have same reference LLC
    df2 = pd.read_parquet(output_path2)
    assert df1.iloc[0]["llc_ref"] == df2.iloc[0]["llc_ref"], "Reference LLC should be identical"
