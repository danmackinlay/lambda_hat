"""Integration test for Parsl workflow.

Tests end-to-end pipeline: build target → run sampler → aggregate results.
Uses minimal config (1 target × 1 sampler) for fast execution.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from lambda_hat.workflows.parsl_llc import run_workflow

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def test_config():
    """Path to minimal test configuration."""
    return ROOT / "tests" / "test_parsl_workflow_config.yaml"


@pytest.fixture
def parsl_config():
    """Path to local Parsl configuration."""
    return ROOT / "parsl_config_local.py"


@pytest.fixture
def cleanup_test_runs():
    """Cleanup test artifacts after test completes."""
    yield
    # Cleanup after test
    test_runs = ROOT / "test_runs_parsl"
    if test_runs.exists():
        shutil.rmtree(test_runs)

    results_dir = ROOT / "results"
    results_file = results_dir / "llc_runs.parquet"
    if results_file.exists():
        results_file.unlink()

    # Cleanup temp config dir
    temp_cfg = ROOT / "temp_parsl_cfg"
    if temp_cfg.exists():
        shutil.rmtree(temp_cfg)

    # Cleanup log dirs
    logs_dir = ROOT / "logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)

    # Cleanup Parsl runinfo
    parsl_runinfo = ROOT / "parsl_runinfo"
    if parsl_runinfo.exists():
        shutil.rmtree(parsl_runinfo)


def test_parsl_workflow_integration(test_config, parsl_config, cleanup_test_runs):
    """Test full Parsl workflow with minimal config.

    Verifies:
    - Target build completes (meta.json created)
    - Sampler run completes (analysis.json created)
    - Result aggregation works (parquet file created)
    - Aggregated data contains expected metrics
    """
    import parsl
    from omegaconf import OmegaConf

    from lambda_hat.parsl_cards import load_parsl_config_from_card

    # Load Parsl config
    parsl_cfg = load_parsl_config_from_card(parsl_config, overrides=[])
    parsl.load(parsl_cfg)

    try:
        # Run workflow without promotion (faster)
        output_path = run_workflow(str(test_config), enable_promotion=False, promote_plots=None)

        # Verify aggregated results file exists
        assert Path(output_path).exists(), f"Aggregated results not found: {output_path}"

        # Load and verify parquet contents
        df = pd.read_parquet(output_path)
        assert len(df) == 1, f"Expected 1 row in results, got {len(df)}"

        # Verify required columns exist
        required_cols = ["target_id", "sampler", "run_id", "run_dir"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Verify metrics exist (from analysis.json)
        metrics_cols = ["llc_mean", "llc_std", "walltime_sec"]
        for col in metrics_cols:
            assert col in df.columns, f"Missing metric column: {col}"
    finally:
        parsl.clear()

    # Verify target artifacts exist
    row = df.iloc[0]
    target_id = row["target_id"]
    test_runs = ROOT / "test_runs_parsl"
    target_dir = test_runs / "targets" / target_id

    assert target_dir.exists(), f"Target directory not found: {target_dir}"
    assert (target_dir / "meta.json").exists(), "meta.json not found"
    assert (target_dir / "params.npz").exists(), "params.npz not found"
    assert (target_dir / "data.npz").exists(), "data.npz not found"

    # Verify run artifacts exist
    run_dir = Path(row["run_dir"])
    assert run_dir.exists(), f"Run directory not found: {run_dir}"
    assert (run_dir / "analysis.json").exists(), "analysis.json not found"
    assert (run_dir / "trace.nc").exists(), "trace.nc not found"

    print(f"✓ Integration test passed! Output: {output_path}")
