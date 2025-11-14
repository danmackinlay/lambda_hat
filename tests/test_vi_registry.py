"""Test VI registry for algorithm dispatch."""

import pytest

from lambda_hat.vi import get as get_vi_algo


def test_registry_mfa():
    """Verify registry returns MFA algorithm."""
    algo = get_vi_algo("mfa")
    assert algo.name == "mfa"


def test_registry_unknown_raises():
    """Verify registry raises on unknown algorithm."""
    with pytest.raises(ValueError, match="Unknown VI algorithm"):
        get_vi_algo("nonexistent")
