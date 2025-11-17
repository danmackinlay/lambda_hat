# tests/test_vi_integration.py
"""Integration tests for VI algorithms (MFA and Flow).

Focuses on high-signal end-to-end behavior rather than trivial unit tests.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from lambda_hat.equinox_adapter import vectorise_model
from lambda_hat.posterior import (
    make_grad_loss_minibatch_flat,
    make_loss_full_flat,
    make_loss_minibatch_flat,
    make_posterior,
)
from lambda_hat.samplers import run_vi

# Suppress JAX warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture
def tiny_problem():
    """Create a minimal regression problem for fast testing."""
    d = 10
    n = 100
    key = jax.random.PRNGKey(42)

    # Simple linear regression setup (just a dict with parameter array)
    params = {"w": jax.random.normal(key, (d,)).astype(jnp.float32)}
    X = jax.random.normal(key, (n, d)).astype(jnp.float32)
    Y = jax.random.normal(key, (n,)).astype(jnp.float32)

    # Loss functions that work on model (dict with 'w' key)
    def loss_batch(model, Xb, Yb):
        return jnp.mean((model["w"] @ Xb.T - Yb) ** 2)

    def loss_full(model):
        return jnp.mean((model["w"] @ X.T - Y) ** 2)

    # Vectorize the model
    vm, flat0 = vectorise_model(params, dtype=jnp.float32)

    # Create posterior
    beta = 1.0
    gamma = 0.001
    posterior = make_posterior(vm, flat0, loss_full, n, beta, gamma)

    # Create flat loss functions
    loss_minibatch_flat = make_loss_minibatch_flat(vm, loss_batch)
    grad_loss_minibatch = make_grad_loss_minibatch_flat(vm, loss_batch)
    loss_full_flat = make_loss_full_flat(vm, loss_full)

    return {
        "posterior": posterior,
        "loss_minibatch_flat": loss_minibatch_flat,
        "grad_loss_minibatch": grad_loss_minibatch,
        "loss_full_flat": loss_full_flat,
        "X": X,
        "Y": Y,
        "n_data": n,
        "beta": beta,
        "gamma": gamma,
        "key": key,
    }


@pytest.fixture
def mfa_config():
    """Minimal MFA configuration for testing."""
    return OmegaConf.create(
        {
            "algo": "mfa",
            "M": 2,
            "r": 1,
            "steps": 10,
            "batch_size": 25,  # n_data/4 for testing FGE
            "lr": 0.01,
            "eval_samples": 4,
            "clip_global_norm": 5.0,
            "alpha_temperature": 1.0,
            "entropy_bonus": 0.0,
            "alpha_dirichlet_prior": None,
            "r_per_component": None,
            "lr_schedule": None,
            "lr_warmup_frac": 0.05,
            "whitening_mode": "none",
            "whitening_decay": 0.9,
            "dtype": "float32",
            "eval_every": 10,
            "whitening_steps": 50,
        }
    )


def test_vi_algorithms_return_consistent_structure(tiny_problem, mfa_config):
    """Test that MFA returns SamplerRunResult with correct structure."""
    # Run MFA
    mfa_result = run_vi(
        key=tiny_problem["key"],
        posterior=tiny_problem["posterior"],
        data=(tiny_problem["X"], tiny_problem["Y"]),
        config=mfa_config,
        num_chains=1,
        loss_minibatch_flat=tiny_problem["loss_minibatch_flat"],
        grad_loss_minibatch=tiny_problem["grad_loss_minibatch"],
        loss_full_flat=tiny_problem["loss_full_flat"],
        n_data=tiny_problem["n_data"],
        beta=tiny_problem["beta"],
        gamma=tiny_problem["gamma"],
    )

    # Check MFA structure
    assert hasattr(mfa_result, "traces"), "MFA should return SamplerRunResult with traces"
    assert hasattr(mfa_result, "timings"), "MFA should have timings"
    assert hasattr(mfa_result, "work"), "MFA should have work metrics"

    # Check required trace keys
    assert "llc" in mfa_result.traces, "MFA should have llc trace"
    assert "cumulative_fge" in mfa_result.traces, "MFA should have cumulative_fge"

    # Check trace shapes
    assert mfa_result.traces["llc"].shape == (1, 10), (
        f"Expected (1, 10), got {mfa_result.traces['llc'].shape}"
    )
    assert mfa_result.traces["cumulative_fge"].shape == (1, 10), "cumulative_fge shape mismatch"

    # Check work dict keys
    assert "n_full_loss" in mfa_result.work, "Work should have n_full_loss"
    assert "n_minibatch_grads" in mfa_result.work, "Work should have n_minibatch_grads"
    assert "sampler_flavour" in mfa_result.work, "Work should have sampler_flavour"
    assert mfa_result.work["sampler_flavour"] == "mfa", "Wrong sampler flavour"


def test_fge_calculation_correctness_mfa(tiny_problem, mfa_config):
    """Test that FGE is calculated correctly for MFA: batch_size/n_data per step."""
    n_data = tiny_problem["n_data"]
    batch_size = mfa_config.batch_size
    steps = mfa_config.steps

    result = run_vi(
        key=tiny_problem["key"],
        posterior=tiny_problem["posterior"],
        data=(tiny_problem["X"], tiny_problem["Y"]),
        config=mfa_config,
        num_chains=1,
        loss_minibatch_flat=tiny_problem["loss_minibatch_flat"],
        grad_loss_minibatch=tiny_problem["grad_loss_minibatch"],
        loss_full_flat=tiny_problem["loss_full_flat"],
        n_data=n_data,
        beta=tiny_problem["beta"],
        gamma=tiny_problem["gamma"],
    )

    # Expected FGE per step
    expected_fge_per_step = batch_size / n_data  # 25/100 = 0.25
    expected_final_fge = steps * expected_fge_per_step  # 10 * 0.25 = 2.5

    # Check final cumulative FGE
    final_fge = float(result.traces["cumulative_fge"][0, -1])
    assert abs(final_fge - expected_final_fge) < 0.01, (
        f"Expected final FGE ≈ {expected_final_fge}, got {final_fge}"
    )


def test_whitener_integration_mfa(tiny_problem, mfa_config):
    """Test that MFA works with and without whitener (no crashes)."""
    # Test 1: Without whitener (whitening_mode='none')
    result_no_whitener = run_vi(
        key=tiny_problem["key"],
        posterior=tiny_problem["posterior"],
        data=(tiny_problem["X"], tiny_problem["Y"]),
        config=mfa_config,
        num_chains=1,
        loss_minibatch_flat=tiny_problem["loss_minibatch_flat"],
        grad_loss_minibatch=tiny_problem["grad_loss_minibatch"],
        loss_full_flat=tiny_problem["loss_full_flat"],
        n_data=tiny_problem["n_data"],
        beta=tiny_problem["beta"],
        gamma=tiny_problem["gamma"],
    )

    assert jnp.isfinite(result_no_whitener.traces["llc"]).all(), (
        "MFA without whitener produced non-finite values"
    )

    # Test 2: With whitener (adam)
    config_with_whitener = OmegaConf.create({**mfa_config, "whitening_mode": "adam"})
    result_with_whitener = run_vi(
        key=tiny_problem["key"],
        posterior=tiny_problem["posterior"],
        data=(tiny_problem["X"], tiny_problem["Y"]),
        config=config_with_whitener,
        num_chains=1,
        loss_minibatch_flat=tiny_problem["loss_minibatch_flat"],
        grad_loss_minibatch=tiny_problem["grad_loss_minibatch"],
        loss_full_flat=tiny_problem["loss_full_flat"],
        n_data=tiny_problem["n_data"],
        beta=tiny_problem["beta"],
        gamma=tiny_problem["gamma"],
    )

    assert jnp.isfinite(result_with_whitener.traces["llc"]).all(), (
        "MFA with whitener produced non-finite values"
    )

    # Results should differ (whitener affects optimization)
    # Note: We can't assert they're different because with such a simple problem
    # and random initialization, they might converge to similar values
    # Just verify both work without crashing


def test_work_metrics_structure(tiny_problem, mfa_config):
    """Test that work dict has expected structure across algorithms."""
    result = run_vi(
        key=tiny_problem["key"],
        posterior=tiny_problem["posterior"],
        data=(tiny_problem["X"], tiny_problem["Y"]),
        config=mfa_config,
        num_chains=1,
        loss_minibatch_flat=tiny_problem["loss_minibatch_flat"],
        grad_loss_minibatch=tiny_problem["grad_loss_minibatch"],
        loss_full_flat=tiny_problem["loss_full_flat"],
        n_data=tiny_problem["n_data"],
        beta=tiny_problem["beta"],
        gamma=tiny_problem["gamma"],
    )

    # Check work dict structure
    work = result.work
    assert isinstance(work, dict), "Work should be a dict"
    assert "n_full_loss" in work, "Missing n_full_loss"
    assert "n_minibatch_grads" in work, "Missing n_minibatch_grads"
    assert "sampler_flavour" in work, "Missing sampler_flavour"

    # Check values make sense
    assert work["n_minibatch_grads"] == mfa_config.steps, "n_minibatch_grads should equal steps"
    assert work["n_full_loss"] == mfa_config.eval_samples, (
        "n_full_loss should equal eval_samples (for VI)"
    )
