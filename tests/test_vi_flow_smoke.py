"""Smoke test for normalizing flow VI algorithm.

Tests basic functionality without requiring flowjax to be installed.
If flowjax is not installed, tests verify proper ImportError is raised.
"""

import pytest

from lambda_hat.vi import get as get_vi_algo


def test_flow_registry():
    """Verify flow algorithm is registered."""
    algo = get_vi_algo("flow")
    assert algo.name == "flow"


def test_flow_import_error_without_flowjax():
    """Verify flow raises ImportError if flowjax not installed."""
    # Try to get the flow algorithm
    algo = get_vi_algo("flow")

    # The algorithm should be registered, but calling it should fail
    # if flowjax is not installed. This tests the import guards.
    try:
        import flowjax  # noqa: F401

        # If flowjax is installed, skip this test
        pytest.skip("flowjax is installed, skipping import error test")
    except ImportError:
        # flowjax not installed - verify algo.run() raises ImportError
        import jax
        import jax.numpy as jnp

        # Create minimal dummy inputs
        key = jax.random.PRNGKey(0)
        d = 5
        wstar = jnp.zeros(d)

        def loss_fn(w, X, Y):
            return 0.0

        def loss_full_fn(w):
            return 0.0

        def unravel(flat):
            return {"w": flat}

        data = (jnp.zeros((10, d)), jnp.zeros((10, 1)))

        # Create minimal VIConfig mock
        class MockConfig:
            algo = "flow"
            d_latent = 2
            sigma_perp = 1e-3
            flow_type = "realnvp"
            flow_depth = 2
            flow_hidden = [16, 16]
            steps = 10
            batch_size = 5
            lr = 0.01
            eval_samples = 5
            dtype = "float32"
            clip_global_norm = None
            lr_schedule = None
            lr_warmup_frac = 0.0

        config = MockConfig()

        # Running algo.run() should raise ImportError about flowjax
        with pytest.raises(ImportError, match="flowjax|equinox"):
            algo.run(
                rng_key=key,
                loss_batch_fn=loss_fn,
                loss_full_fn=loss_full_fn,
                wstar_flat=wstar,
                unravel_fn=unravel,
                data=data,
                n_data=10,
                beta=0.1,
                gamma=1.0,
                vi_cfg=config,
            )
