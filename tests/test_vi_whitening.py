# tests/test_vi_whitening.py
"""Tests for VI whitening and stability features (Stage 1)."""

import jax
import jax.numpy as jnp

from lambda_hat.vi import mfa as vi


def test_softmax_with_temperature():
    """Test temperature-adjusted softmax."""
    logits = jnp.array([1.0, 2.0, 3.0])

    # Temperature = 1.0 should give standard softmax
    probs_t1 = vi.softmax_with_temperature(logits, temperature=1.0)
    probs_std = jax.nn.softmax(logits)
    assert jnp.allclose(probs_t1, probs_std)

    # Higher temperature should produce more uniform distribution
    probs_t2 = vi.softmax_with_temperature(logits, temperature=2.0)
    entropy_t1 = -jnp.sum(probs_t1 * jnp.log(probs_t1 + 1e-10))
    entropy_t2 = -jnp.sum(probs_t2 * jnp.log(probs_t2 + 1e-10))
    assert entropy_t2 > entropy_t1  # Higher temperature => higher entropy

    # Temperature < 0.5 should be clipped to 0.5
    probs_low = vi.softmax_with_temperature(logits, temperature=0.1)
    probs_half = vi.softmax_with_temperature(logits, temperature=0.5)
    assert jnp.allclose(probs_low, probs_half)


def test_whitener_identity():
    """Test identity whitening (A_diag=None)."""
    whitener = vi.make_whitener(None)

    v = jnp.array([1.0, 2.0, 3.0])
    tilde_v = whitener.to_tilde(v)
    v_reconstructed = whitener.from_tilde(tilde_v)

    assert jnp.allclose(tilde_v, v)  # Identity transformation
    assert jnp.allclose(v_reconstructed, v)


def test_whitener_diagonal():
    """Test diagonal whitening."""
    A_diag = jnp.array([1.0, 4.0, 9.0])  # Will be sqrt to get [1, 2, 3]
    whitener = vi.make_whitener(A_diag)

    v = jnp.array([1.0, 1.0, 1.0])
    tilde_v = whitener.to_tilde(v)
    v_reconstructed = whitener.from_tilde(tilde_v)

    # Whitening should scale by A^{1/2} = [1, 2, 3]
    expected_tilde_v = v * jnp.sqrt(A_diag)
    assert jnp.allclose(tilde_v, expected_tilde_v)
    assert jnp.allclose(v_reconstructed, v)


def test_whitener_numerical_stability():
    """Test whitening with very small diagonal values (numerical stability)."""
    A_diag = jnp.array([1e-12, 1.0, 1e-12])  # Very small values
    whitener = vi.make_whitener(A_diag, eps=1e-8)

    v = jnp.array([1.0, 1.0, 1.0])
    tilde_v = whitener.to_tilde(v)
    v_reconstructed = whitener.from_tilde(tilde_v)

    # Should not produce NaN or Inf
    assert not jnp.any(jnp.isnan(tilde_v))
    assert not jnp.any(jnp.isinf(tilde_v))
    assert not jnp.any(jnp.isnan(v_reconstructed))
    assert not jnp.any(jnp.isinf(v_reconstructed))

    # Reconstruction should be close (with some numerical error)
    assert jnp.allclose(v_reconstructed, v, rtol=1e-4)
