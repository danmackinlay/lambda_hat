"""Test Stage 3 advanced configuration features for VI sampler.

This test validates:
- Per-component rank budgets (r_per_component)
- Entropy bonus
- Dirichlet prior on mixture weights
- LR schedules (cosine, linear decay)
"""

# CRITICAL: Enable x64 BEFORE any JAX imports
import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from lambda_hat import variational as vi


def test_vi_per_component_ranks():
    """Test per-component rank budgets with masking."""
    d = 5
    M = 3
    r = 4  # Max rank
    rng = jax.random.PRNGKey(50)

    X = jax.random.normal(rng, (50, d))
    Y = jax.random.normal(jax.random.split(rng)[0], (50, 1))
    wstar_flat = jnp.zeros(d)

    def unravel_fn(flat):
        return {"w": flat}

    def loss_batch_fn(params, Xb, Yb):
        w = params["w"]
        pred = jnp.dot(Xb, w)
        return jnp.mean((pred[:, None] - Yb) ** 2)

    def loss_full_fn(params):
        return loss_batch_fn(params, X, Y)

    # Test with heterogeneous ranks: [2, 3, 4]
    r_per_component = [2, 3, 4]

    whitener = vi.make_whitener(None)
    lambda_hat, traces, extras = vi.fit_vi_and_estimate_lambda(
        rng_key=rng,
        loss_batch_fn=loss_batch_fn,
        loss_full_fn=loss_full_fn,
        wstar_flat=wstar_flat,
        unravel_fn=unravel_fn,
        data=(X, Y),
        n_data=50,
        beta=0.1,
        gamma=1.0,
        M=M,
        r=r,
        steps=100,
        batch_size=10,
        lr=0.01,
        eval_samples=10,
        whitener=whitener,
        clip_global_norm=5.0,
        alpha_temperature=1.0,
        r_per_component=r_per_component,
    )

    # Verify lambda_hat is finite
    assert jnp.isfinite(lambda_hat), "lambda_hat is not finite"
    # Verify traces are finite
    assert jnp.all(jnp.isfinite(traces["elbo"])), "ELBO has non-finite values"


def test_vi_entropy_bonus():
    """Test entropy bonus encourages more uniform mixture weights."""
    d = 5
    rng = jax.random.PRNGKey(51)

    X = jax.random.normal(rng, (50, d))
    Y = jax.random.normal(jax.random.split(rng)[0], (50, 1))
    wstar_flat = jnp.zeros(d)

    def unravel_fn(flat):
        return {"w": flat}

    def loss_batch_fn(params, Xb, Yb):
        w = params["w"]
        pred = jnp.dot(Xb, w)
        return jnp.mean((pred[:, None] - Yb) ** 2)

    def loss_full_fn(params):
        return loss_batch_fn(params, X, Y)

    # Test with entropy bonus
    whitener = vi.make_whitener(None)
    lambda_hat, traces, extras = vi.fit_vi_and_estimate_lambda(
        rng_key=rng,
        loss_batch_fn=loss_batch_fn,
        loss_full_fn=loss_full_fn,
        wstar_flat=wstar_flat,
        unravel_fn=unravel_fn,
        data=(X, Y),
        n_data=50,
        beta=0.1,
        gamma=1.0,
        M=4,
        r=2,
        steps=100,
        batch_size=10,
        lr=0.01,
        eval_samples=10,
        whitener=whitener,
        clip_global_norm=5.0,
        alpha_temperature=1.0,
        entropy_bonus=0.1,  # Encourage exploration
    )

    # Verify lambda_hat is finite
    assert jnp.isfinite(lambda_hat), "lambda_hat is not finite"
    # Verify pi_entropy is tracked and reasonable
    assert "pi_entropy" in traces, "pi_entropy not in traces"
    final_pi_entropy = traces["pi_entropy"][-1]
    assert jnp.isfinite(final_pi_entropy), "pi_entropy is not finite"
    assert final_pi_entropy >= 0.0, "pi_entropy should be non-negative"


def test_vi_dirichlet_prior():
    """Test Dirichlet prior on mixture weights."""
    d = 5
    rng = jax.random.PRNGKey(52)

    X = jax.random.normal(rng, (50, d))
    Y = jax.random.normal(jax.random.split(rng)[0], (50, 1))
    wstar_flat = jnp.zeros(d)

    def unravel_fn(flat):
        return {"w": flat}

    def loss_batch_fn(params, Xb, Yb):
        w = params["w"]
        pred = jnp.dot(Xb, w)
        return jnp.mean((pred[:, None] - Yb) ** 2)

    def loss_full_fn(params):
        return loss_batch_fn(params, X, Y)

    # Test with Dirichlet prior (symmetric with α₀ = 2.0)
    whitener = vi.make_whitener(None)
    lambda_hat, traces, extras = vi.fit_vi_and_estimate_lambda(
        rng_key=rng,
        loss_batch_fn=loss_batch_fn,
        loss_full_fn=loss_full_fn,
        wstar_flat=wstar_flat,
        unravel_fn=unravel_fn,
        data=(X, Y),
        n_data=50,
        beta=0.1,
        gamma=1.0,
        M=4,
        r=2,
        steps=100,
        batch_size=10,
        lr=0.01,
        eval_samples=10,
        whitener=whitener,
        clip_global_norm=5.0,
        alpha_temperature=1.0,
        alpha_dirichlet_prior=2.0,  # Symmetric Dirichlet prior
    )

    # Verify lambda_hat is finite
    assert jnp.isfinite(lambda_hat), "lambda_hat is not finite"
    # Verify traces are finite
    assert jnp.all(jnp.isfinite(traces["elbo"])), "ELBO has non-finite values"


def test_vi_lr_schedule_cosine():
    """Test cosine LR schedule with warmup."""
    d = 5
    rng = jax.random.PRNGKey(53)

    X = jax.random.normal(rng, (50, d))
    Y = jax.random.normal(jax.random.split(rng)[0], (50, 1))
    wstar_flat = jnp.zeros(d)

    def unravel_fn(flat):
        return {"w": flat}

    def loss_batch_fn(params, Xb, Yb):
        w = params["w"]
        pred = jnp.dot(Xb, w)
        return jnp.mean((pred[:, None] - Yb) ** 2)

    def loss_full_fn(params):
        return loss_batch_fn(params, X, Y)

    # Test with cosine schedule
    whitener = vi.make_whitener(None)
    lambda_hat, traces, extras = vi.fit_vi_and_estimate_lambda(
        rng_key=rng,
        loss_batch_fn=loss_batch_fn,
        loss_full_fn=loss_full_fn,
        wstar_flat=wstar_flat,
        unravel_fn=unravel_fn,
        data=(X, Y),
        n_data=50,
        beta=0.1,
        gamma=1.0,
        M=4,
        r=2,
        steps=100,
        batch_size=10,
        lr=0.01,
        eval_samples=10,
        whitener=whitener,
        clip_global_norm=5.0,
        alpha_temperature=1.0,
        lr_schedule="cosine",
        lr_warmup_frac=0.1,
    )

    # Verify lambda_hat is finite
    assert jnp.isfinite(lambda_hat), "lambda_hat is not finite"
    # Verify traces are finite
    assert jnp.all(jnp.isfinite(traces["elbo"])), "ELBO has non-finite values"


def test_vi_lr_schedule_linear_decay():
    """Test linear decay LR schedule."""
    d = 5
    rng = jax.random.PRNGKey(54)

    X = jax.random.normal(rng, (50, d))
    Y = jax.random.normal(jax.random.split(rng)[0], (50, 1))
    wstar_flat = jnp.zeros(d)

    def unravel_fn(flat):
        return {"w": flat}

    def loss_batch_fn(params, Xb, Yb):
        w = params["w"]
        pred = jnp.dot(Xb, w)
        return jnp.mean((pred[:, None] - Yb) ** 2)

    def loss_full_fn(params):
        return loss_batch_fn(params, X, Y)

    # Test with linear decay
    whitener = vi.make_whitener(None)
    lambda_hat, traces, extras = vi.fit_vi_and_estimate_lambda(
        rng_key=rng,
        loss_batch_fn=loss_batch_fn,
        loss_full_fn=loss_full_fn,
        wstar_flat=wstar_flat,
        unravel_fn=unravel_fn,
        data=(X, Y),
        n_data=50,
        beta=0.1,
        gamma=1.0,
        M=4,
        r=2,
        steps=100,
        batch_size=10,
        lr=0.01,
        eval_samples=10,
        whitener=whitener,
        clip_global_norm=5.0,
        alpha_temperature=1.0,
        lr_schedule="linear_decay",
        lr_warmup_frac=0.05,
    )

    # Verify lambda_hat is finite
    assert jnp.isfinite(lambda_hat), "lambda_hat is not finite"
    # Verify traces are finite
    assert jnp.all(jnp.isfinite(traces["elbo"])), "ELBO has non-finite values"
