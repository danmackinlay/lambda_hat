"""Tests for MCLMC API contract with BlackJAX 1.2.5."""

import inspect
import blackjax


def test_mclmc_tuner_signature_has_no_integrator():
    """Ensure mclmc_find_L_and_step_size doesn't accept integrator parameter."""
    sig = inspect.signature(blackjax.mclmc_find_L_and_step_size)
    assert "integrator" not in sig.parameters, "integrator should be specified on kernel creation, not tuner"


def test_mclmc_tuner_has_required_parameters():
    """Ensure mclmc_find_L_and_step_size has the expected parameters."""
    sig = inspect.signature(blackjax.mclmc_find_L_and_step_size)
    required_params = [
        "mclmc_kernel", "num_steps", "state", "rng_key",
        "frac_tune1", "frac_tune2", "frac_tune3"
    ]

    for name in required_params:
        assert name in sig.parameters, f"Required parameter '{name}' missing from tuner signature"


def test_mclmc_kernel_accepts_integrator():
    """Ensure MCLMC kernel creation accepts integrator parameter."""
    import jax.numpy as jnp

    def dummy_logdensity(x):
        return -0.5 * jnp.sum(x**2)

    # This should work - integrator goes on kernel creation
    try:
        integrator = blackjax.mcmc.integrators.isokinetic_mclachlan
        kernel = blackjax.mclmc(dummy_logdensity, L=1, step_size=0.1, integrator=integrator)
        assert kernel is not None
    except Exception as e:
        assert False, f"MCLMC kernel creation with integrator failed: {e}"