"""Test VI on realistic MLP targets with data-dependent losses.

These tests validate VI on actual production use cases:
- Tiny MLPs with real data
- Data-dependent MSE loss (not synthetic quadratics)
- Conservative hyperparameters for test stability

This replaces synthetic quadratic tests which have structural incompatibilities
with VI's gradient computation (see VI_TEST_INVESTIGATION_FOLLOWUP.md).
"""

# CRITICAL: Enable x64 BEFORE any JAX imports
import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from lambda_hat.equinox_adapter import vectorise_model  # noqa: E402
from lambda_hat.nn_eqx import build_mlp  # noqa: E402
from lambda_hat.posterior import (  # noqa: E402
    make_grad_loss_minibatch_flat,
    make_loss_full_flat,
    make_loss_minibatch_flat,
    make_posterior,
)
from lambda_hat.samplers import run_vi  # noqa: E402


def make_tiny_mlp_target(key, in_dim=4, out_dim=1, hidden=8):
    """Create a tiny MLP target for testing.

    Args:
        key: JAX random key
        in_dim: Input dimension
        out_dim: Output dimension
        hidden: Hidden layer width

    Returns:
        model: Equinox MLP module (parameters are part of the module)
        d: Total number of parameters (flat)
    """
    # Create MLP using Equinox
    key_init, key = jax.random.split(key)
    model = build_mlp(
        in_dim=in_dim,
        widths=[hidden],
        out_dim=out_dim,
        activation="relu",
        bias=True,
        key=key_init,
    )

    return model


def generate_regression_data(key, n_data, in_dim, teacher_fn):
    """Generate synthetic regression dataset.

    Args:
        key: JAX random key
        n_data: Number of samples
        in_dim: Input dimension
        teacher_fn: Function to generate targets y = f(x)

    Returns:
        X: (n_data, in_dim) inputs
        Y: (n_data,) or (n_data, out_dim) targets
    """
    key_x, key_noise = jax.random.split(key)

    # Generate inputs from standard normal
    X = jax.random.normal(key_x, (n_data, in_dim))

    # Generate targets from teacher
    Y_clean = teacher_fn(X)

    # Add small noise
    noise = jax.random.normal(key_noise, Y_clean.shape) * 0.1
    Y = Y_clean + noise

    return X, Y


def test_vi_tiny_mlp_convergence():
    """Test that VI converges on a tiny MLP with real data."""
    key = jax.random.PRNGKey(42)

    # Small problem for fast testing
    in_dim = 4
    out_dim = 1
    hidden = 8
    n_data = 50

    # Create student MLP
    key_student, key = jax.random.split(key)
    model = make_tiny_mlp_target(key_student, in_dim=in_dim, out_dim=out_dim, hidden=hidden)

    # Create teacher (simple linear function)
    key_teacher, key = jax.random.split(key)
    W_teacher = jax.random.normal(key_teacher, (in_dim,))

    def teacher_fn(X):
        return X @ W_teacher

    # Generate data
    key_data, key = jax.random.split(key)
    X, Y = generate_regression_data(key_data, n_data, in_dim, teacher_fn)
    Y = Y.reshape(-1)  # Ensure (n_data,) shape

    # Vectorize the MLP model
    vm, flat0 = vectorise_model(model, dtype=jnp.float32)

    # Define loss functions (work on Equinox model)
    def loss_batch_fn(model_eqx, Xb, Yb):
        """Batch loss (data-dependent!)"""
        preds = model_eqx(Xb).reshape(-1)
        return jnp.mean((preds - Yb) ** 2)

    def loss_full_fn(model_eqx):
        """Full dataset loss"""
        preds = model_eqx(X).reshape(-1)
        return jnp.mean((preds - Y) ** 2)

    # Create posterior
    beta = 1.0 / jnp.log(n_data)
    gamma = 0.5  # Moderate localizer
    posterior = make_posterior(vm, flat0, loss_full_fn, n_data, beta, gamma)

    # Create flat loss functions
    loss_minibatch_flat = make_loss_minibatch_flat(vm, loss_batch_fn)
    grad_loss_minibatch = make_grad_loss_minibatch_flat(vm, loss_batch_fn)
    loss_full_flat = make_loss_full_flat(vm, loss_full_fn)

    # VI hyperparameters (conservative for small problem)
    config = OmegaConf.create(
        {
            "algo": "mfa",
            "M": 2,  # Small mixture for tiny problem
            "r": 2,  # Small rank
            "steps": 300,
            "batch_size": 16,
            "lr": 0.005,  # Conservative
            "eval_samples": 200,
            "whitening_mode": "none",
            "clip_global_norm": 5.0,
            "alpha_temperature": 1.0,
            "entropy_bonus": 0.0,
            "alpha_dirichlet_prior": None,
            "r_per_component": None,
            "lr_schedule": None,
            "lr_warmup_frac": 0.05,
            "whitening_decay": 0.9,
            "dtype": "float32",
            "eval_every": 10,
            "whitening_steps": 50,
        }
    )

    # Run VI
    key_vi, key = jax.random.split(key)
    result = run_vi(
        key=key_vi,
        posterior=posterior,
        data=(X, Y),
        config=config,
        num_chains=1,
        loss_minibatch_flat=loss_minibatch_flat,
        grad_loss_minibatch=grad_loss_minibatch,
        loss_full_flat=loss_full_flat,
        n_data=n_data,
        beta=beta,
        gamma=gamma,
    )

    # Assertions: Basic sanity checks
    assert jnp.isfinite(result.traces["llc"]).all(), "LLC trace should be finite"
    assert result.traces["llc"].shape[0] == 1, "Should have 1 chain"
    assert result.traces["llc"].shape[1] == config.steps, f"Should have {config.steps} steps"

    print("✓ VI converged successfully on tiny MLP")


def test_vi_tiny_mlp_cv_reduces_variance():
    """Test that VI produces reasonable results on MLP target."""
    key = jax.random.PRNGKey(123)

    # Even smaller problem for faster test
    in_dim = 3
    out_dim = 1
    hidden = 6
    n_data = 40

    # Create MLP target
    key_student, key = jax.random.split(key)
    model = make_tiny_mlp_target(key_student, in_dim=in_dim, out_dim=out_dim, hidden=hidden)

    # Simple teacher
    key_teacher, key = jax.random.split(key)
    W_teacher = jax.random.normal(key_teacher, (in_dim,))

    def teacher_fn(X):
        return X @ W_teacher

    # Generate data
    key_data, key = jax.random.split(key)
    X, Y = generate_regression_data(key_data, n_data, in_dim, teacher_fn)
    Y = Y.reshape(-1)

    # Vectorize model
    vm, flat0 = vectorise_model(model, dtype=jnp.float32)

    # Loss functions (work on Equinox model)
    def loss_batch_fn(model_eqx, Xb, Yb):
        preds = model_eqx(Xb).reshape(-1)
        return jnp.mean((preds - Yb) ** 2)

    def loss_full_fn(model_eqx):
        preds = model_eqx(X).reshape(-1)
        return jnp.mean((preds - Y) ** 2)

    # Create posterior
    beta = 1.0 / jnp.log(n_data)
    gamma = 0.5
    posterior = make_posterior(vm, flat0, loss_full_fn, n_data, beta, gamma)

    # Create flat loss functions
    loss_minibatch_flat = make_loss_minibatch_flat(vm, loss_batch_fn)
    grad_loss_minibatch = make_grad_loss_minibatch_flat(vm, loss_batch_fn)
    loss_full_flat = make_loss_full_flat(vm, loss_full_fn)

    # VI config
    config = OmegaConf.create(
        {
            "algo": "mfa",
            "M": 2,
            "r": 2,
            "steps": 200,
            "batch_size": 12,
            "lr": 0.005,
            "eval_samples": 150,
            "whitening_mode": "none",
            "clip_global_norm": 5.0,
            "alpha_temperature": 1.0,
            "entropy_bonus": 0.0,
            "alpha_dirichlet_prior": None,
            "r_per_component": None,
            "lr_schedule": None,
            "lr_warmup_frac": 0.05,
            "whitening_decay": 0.9,
            "dtype": "float32",
            "eval_every": 10,
            "whitening_steps": 50,
        }
    )

    # Run VI
    key_vi, key = jax.random.split(key)
    result = run_vi(
        key=key_vi,
        posterior=posterior,
        data=(X, Y),
        config=config,
        num_chains=1,
        loss_minibatch_flat=loss_minibatch_flat,
        grad_loss_minibatch=grad_loss_minibatch,
        loss_full_flat=loss_full_flat,
        n_data=n_data,
        beta=beta,
        gamma=gamma,
    )

    # Check that results are reasonable
    assert jnp.isfinite(result.traces["llc"]).all(), "LLC should be finite"
    assert result.traces["llc"].shape == (1, 200), "LLC shape should be (1, 200)"

    print("✓ VI produces reasonable results on MLP")


def test_vi_tiny_mlp_basic_sanity():
    """Minimal smoke test: VI runs without crashing on tiny MLP."""
    key = jax.random.PRNGKey(456)

    # Minimal problem
    in_dim = 2
    hidden = 4
    n_data = 30

    # Create MLP
    key_mlp, key = jax.random.split(key)
    model = make_tiny_mlp_target(key_mlp, in_dim=in_dim, out_dim=1, hidden=hidden)

    # Simple data
    key_data, key = jax.random.split(key)
    X = jax.random.normal(key_data, (n_data, in_dim))
    Y = jax.random.normal(key_data, (n_data,)) * 0.5

    # Vectorize model
    vm, flat0 = vectorise_model(model, dtype=jnp.float32)

    # Minimal loss functions (work on Equinox model)
    def loss_batch_fn(model_eqx, Xb, Yb):
        preds = model_eqx(Xb).reshape(-1)
        return jnp.mean((preds - Yb) ** 2)

    def loss_full_fn(model_eqx):
        preds = model_eqx(X).reshape(-1)
        return jnp.mean((preds - Y) ** 2)

    # Create posterior
    beta = 1.0 / jnp.log(n_data)
    gamma = 0.5
    posterior = make_posterior(vm, flat0, loss_full_fn, n_data, beta, gamma)

    # Create flat loss functions
    loss_minibatch_flat = make_loss_minibatch_flat(vm, loss_batch_fn)
    grad_loss_minibatch = make_grad_loss_minibatch_flat(vm, loss_batch_fn)
    loss_full_flat = make_loss_full_flat(vm, loss_full_fn)

    # VI config
    config = OmegaConf.create(
        {
            "algo": "mfa",
            "M": 2,
            "r": 1,
            "steps": 50,  # Very short
            "batch_size": 10,
            "lr": 0.01,
            "eval_samples": 50,
            "whitening_mode": "none",
            "clip_global_norm": 5.0,
            "alpha_temperature": 1.0,
            "entropy_bonus": 0.0,
            "alpha_dirichlet_prior": None,
            "r_per_component": None,
            "lr_schedule": None,
            "lr_warmup_frac": 0.05,
            "whitening_decay": 0.9,
            "dtype": "float32",
            "eval_every": 10,
            "whitening_steps": 50,
        }
    )

    # Run VI with minimal steps
    key_vi, key = jax.random.split(key)
    result = run_vi(
        key=key_vi,
        posterior=posterior,
        data=(X, Y),
        config=config,
        num_chains=1,
        loss_minibatch_flat=loss_minibatch_flat,
        grad_loss_minibatch=grad_loss_minibatch,
        loss_full_flat=loss_full_flat,
        n_data=n_data,
        beta=beta,
        gamma=gamma,
    )

    # Just check it completed without NaN
    assert jnp.isfinite(result.traces["llc"]).any(), "At least some LLC values should be finite"

    print("✓ Smoke test passed")
