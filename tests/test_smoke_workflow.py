"""Fast smoke tests for complete workflow.

These tests exercise the full workflow (build + sample) with minimal parameters
to provide quick feedback (~10 seconds) that all samplers work end-to-end.

NOT intended for statistical validation or performance benchmarking.
"""

import os
import subprocess
from pathlib import Path

import pytest


def test_smoke_workflow_all_samplers(tmp_path):
    """Test 1 target × 4 samplers with minimal params.

    Expected runtime: ~8-12 seconds
    - Target build: ~1-2 sec (100 training steps on small network)
    - HMC: ~1-2 sec (10 draws × 5 integration steps)
    - MCLMC: ~1-2 sec (10 draws)
    - SGLD: ~2-3 sec (50 steps)
    - VI: ~2-3 sec (20 steps, M=2, r=1)

    This is a SMOKE TEST - minimal params for fast feedback.
    Tests that refactored flat-parameter interface works across all samplers.
    """
    config_path = Path(__file__).parent / "test_smoke_workflow_config.yaml"

    # Prepare environment with JAX_ENABLE_X64
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "1"

    # Run workflow with smoke config
    result = subprocess.run(
        [
            "uv",
            "run",
            "lambda-hat",
            "workflow",
            "llc",
            "--config",
            str(config_path),
            "--local",
        ],
        capture_output=True,
        text=True,
        timeout=60,  # Generous timeout; expect ~10 sec actual
        env=env,  # Use augmented environment
    )

    # Combine stdout and stderr for checking
    output = result.stdout + result.stderr

    # Basic success assertion
    assert result.returncode == 0, (
        f"Workflow failed with return code {result.returncode}:\n{output}"
    )

    # Verify all 4 samplers were submitted/ran
    # (workflow output includes "Submitting <sampler> for <target>")
    assert "hmc" in output.lower(), "HMC sampler not found in output"
    assert "mclmc" in output.lower(), "MCLMC sampler not found in output"
    assert "sgld" in output.lower(), "SGLD sampler not found in output"
    assert "vi" in output.lower(), "VI sampler not found in output"

    # Verify no failures
    # The workflow prints "✓" for success or "✗ FAILED" for failures
    assert "FAILED" not in output, f"Some samplers failed:\n{output}"

    # Check that target was built successfully
    assert "build_target" in output.lower() or "Build Target" in output

    # Optional: Check for specific success markers
    # The workflow aggregates results at the end
    assert "Aggregating Results" in output or "aggregating" in output.lower()

    print("\n✓ Smoke test passed - all 4 samplers completed successfully")
    print(f"  Check output for timing details")


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
