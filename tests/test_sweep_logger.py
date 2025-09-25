"""Test sweep command logger functionality."""

from lambda_hat.commands.sweep_cmd import _save_sweep_results


def test_save_sweep_no_crash_on_empty():
    """Test that _save_sweep_results doesn't crash on empty results with logging."""
    # Should not raise even with logging calls
    _save_sweep_results([])


def test_save_sweep_with_success_results():
    """Test that _save_sweep_results handles successful results."""
    results = [
        {
            "status": "ok",
            "run_dir": "runs/test_run_001",
            "run_id": "test_run_001",
            "sgld_llc_mean": 123.45,
            "hmc_llc_mean": 134.56,
            "family_id": "test_family_001",
        }
    ]

    # Should not raise and should create CSV file
    _save_sweep_results(results)


def test_save_sweep_with_error_results():
    """Test that _save_sweep_results handles error results with logging."""
    results = [
        {
            "status": "error",
            "run_id": "test_run_error",
            "stage": "run_experiment_task",
            "error_type": "RuntimeError",
            "duration_s": 12.34,
            "error": "Something went wrong during execution",
        }
    ]

    # Should not raise and should create error CSV
    _save_sweep_results(results)


def test_save_sweep_mixed_results():
    """Test that _save_sweep_results handles mixed success/error results."""
    results = [
        {
            "status": "ok",
            "run_dir": "runs/test_run_002",
            "sgld_llc_mean": 111.11,
            "family_id": "test_family_002",
        },
        {
            "status": "error",
            "run_id": "test_run_error_002",
            "error_type": "ValueError",
            "error": "Invalid configuration",
        },
    ]

    # Should handle both types without crashing
    _save_sweep_results(results)
