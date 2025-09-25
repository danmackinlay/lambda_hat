"""Test mixture data generation PRNG coupling."""

from lambda_hat.config import Config
from lambda_hat.data import sample_X
from jax import random
import jax.numpy as jnp
from dataclasses import replace


def test_mixture_prng_no_coupling():
    """Test that mixture PRNG uses proper key splits and gives reproducible results."""
    cfg = Config()
    cfg.data = replace(cfg.data, x_dist="mixture", mixture_k=2)
    key = random.PRNGKey(0)

    # Sample twice with same key - should get identical results
    X1 = sample_X(key, cfg, 16, 4)
    X2 = sample_X(key, cfg, 16, 4)

    assert jnp.allclose(X1, X2), "Same key should produce identical samples"


def test_mixture_different_keys_different_results():
    """Test that different keys produce different mixture samples."""
    cfg = Config()
    cfg.data = replace(cfg.data, x_dist="mixture", mixture_k=2)
    key1 = random.PRNGKey(0)
    key2 = random.PRNGKey(1)

    X1 = sample_X(key1, cfg, 16, 4)
    X2 = sample_X(key2, cfg, 16, 4)

    # Different keys should produce different results
    assert not jnp.allclose(X1, X2), "Different keys should produce different samples"


def test_mixture_shape_consistency():
    """Test that mixture sampling produces correct shapes."""
    cfg = Config()
    cfg.data = replace(cfg.data, x_dist="mixture", mixture_k=3, mixture_spread=2.0)
    key = random.PRNGKey(42)

    n, in_dim = 32, 6
    X = sample_X(key, cfg, n, in_dim)

    assert X.shape == (n, in_dim), f"Expected shape ({n}, {in_dim}), got {X.shape}"
    assert jnp.isfinite(X).all(), "All samples should be finite"
