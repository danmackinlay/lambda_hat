"""Smoke test for normalizing flow VI algorithm.

Tests basic functionality. Skips if flowjax is not installed.
"""

import pytest

from lambda_hat.vi import get as get_vi_algo


def test_flow_registry():
    """Verify flow algorithm is registered (requires --extra flowvi)."""
    # Try to get the flow algorithm - will raise ValueError if flowvi not installed
    try:
        algo = get_vi_algo("flow")
        assert algo.name == "flow"
    except ValueError as e:
        if "requires optional dependencies" in str(e):
            pytest.skip("Flow VI requires --extra flowvi (not installed)")
        raise
