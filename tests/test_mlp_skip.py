"""Test MLP skip/residual connections."""

import jax
import jax.numpy as jnp
import pytest

from lambda_hat.models import build_mlp_forward_fn


def test_mlp_with_skip_connections():
    """Test that MLP can be created with skip=True and runs a forward pass."""
    # Build model with skip connections
    model = build_mlp_forward_fn(
        in_dim=10,
        widths=[20, 20],
        out_dim=1,
        activation="relu",
        skip=True,
        residual_period=2,
    )

    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((5, 10))  # batch of 5, input dim 10
    params = model.init(rng, x)

    # Run forward pass
    y = model.apply(params, None, x)

    # Check output shape
    assert y.shape == (5, 1), f"Expected shape (5, 1), got {y.shape}"
    assert not jnp.any(jnp.isnan(y)), "Output contains NaN values"


def test_mlp_without_skip_connections():
    """Test that MLP still works without skip connections (default behavior)."""
    # Build model without skip connections (default)
    model = build_mlp_forward_fn(
        in_dim=10,
        widths=[20, 20],
        out_dim=1,
        activation="relu",
        skip=False,
    )

    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((5, 10))
    params = model.init(rng, x)

    # Run forward pass
    y = model.apply(params, None, x)

    # Check output shape
    assert y.shape == (5, 1), f"Expected shape (5, 1), got {y.shape}"
    assert not jnp.any(jnp.isnan(y)), "Output contains NaN values"


def test_mlp_skip_parameter_accepted():
    """Test that skip and residual_period parameters are accepted without error."""
    # This test ensures the parameters are wired through correctly
    try:
        model = build_mlp_forward_fn(
            in_dim=5,
            widths=[10],
            out_dim=1,
            skip=True,
            residual_period=3,
        )
        # If we get here, parameters were accepted
        assert True
    except TypeError as e:
        pytest.fail(f"skip/residual_period parameters not accepted: {e}")
