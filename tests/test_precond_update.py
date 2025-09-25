"""Test preconditioning update functions."""

import jax.numpy as jnp
from lambda_hat.samplers.utils import precond_update, DiagPrecondState


def test_adam_returns_mhat_for_drift():
    """Test that precond_update returns bias-corrected first moment for Adam drift."""
    g = jnp.array([1.0, -2.0])
    st = DiagPrecondState(m=jnp.zeros_like(g), v=jnp.zeros_like(g), t=jnp.zeros(()))

    inv_sqrt, newst, drift_m = precond_update(
        g, st, "adam", beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True
    )

    # With t->1, m_hat should be g/(1-β1) due to bias correction
    expected_drift = g / (1.0 - 0.9)
    assert jnp.allclose(drift_m, expected_drift)


def test_rmsprop_returns_raw_gradient():
    """Test that RMSProp mode returns raw gradient for drift."""
    g = jnp.array([1.0, -2.0])
    st = DiagPrecondState(m=jnp.zeros_like(g), v=jnp.zeros_like(g), t=jnp.zeros(()))

    inv_sqrt, newst, drift_m = precond_update(g, st, "rmsprop", beta2=0.999, eps=1e-8)

    # RMSProp should return raw gradient for drift
    assert jnp.allclose(drift_m, g)


def test_none_mode_returns_raw_gradient():
    """Test that none mode returns raw gradient for drift."""
    g = jnp.array([1.0, -2.0])
    st = DiagPrecondState(m=jnp.zeros_like(g), v=jnp.zeros_like(g), t=jnp.zeros(()))

    inv_sqrt, newst, drift_m = precond_update(g, st, "none")

    # None mode should return raw gradient for drift
    assert jnp.allclose(drift_m, g)
