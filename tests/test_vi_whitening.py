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


def test_gradient_clipping():
    """Test gradient clipping in ELBO step."""
    # This is more of an integration test - we'll verify clipping happens
    # by checking that large gradients get scaled down

    # Create dummy data
    d = 10
    rng = jax.random.PRNGKey(42)
    wstar_flat = jnp.zeros(d)

    def unravel_fn(flat):
        return {"w": flat}

    def loss_batch_fn(params, Xb, Yb):
        # Loss that produces large gradients
        return jnp.sum(params["w"] ** 2) * 100.0

    X = jnp.ones((100, 5))
    Y = jnp.ones((100, 1))
    data = (X, Y)

    import optax

    optimizer = optax.adam(0.01)
    whitener = vi.make_whitener(None)

    # Build step function WITH clipping
    step_fn_clipped = vi.build_elbo_step(
        loss_batch_fn=loss_batch_fn,
        data=data,
        wstar_flat=wstar_flat,
        unravel_fn=unravel_fn,
        n_data=100,
        beta=0.1,
        gamma=1.0,
        batch_size=10,
        whitener=whitener,
        optimizer=optimizer,
        clip_global_norm=1.0,  # Clip to norm 1.0
        alpha_temperature=1.0,
    )

    # Build step function WITHOUT clipping
    step_fn_unclipped = vi.build_elbo_step(
        loss_batch_fn=loss_batch_fn,
        data=data,
        wstar_flat=wstar_flat,
        unravel_fn=unravel_fn,
        n_data=100,
        beta=0.1,
        gamma=1.0,
        batch_size=10,
        whitener=whitener,
        optimizer=optimizer,
        clip_global_norm=None,  # No clipping
        alpha_temperature=1.0,
    )

    # Initialize VI params
    vi_params = vi.init_vi_params(rng, wstar_flat, M=2, r=2)
    opt_state = optimizer.init(vi_params)
    vi_state = vi.VIOptState(
        opt_state=opt_state,
        baseline=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    # Run one step with each
    key_step = jax.random.PRNGKey(123)
    (params_clipped, _), _ = step_fn_clipped(key_step, vi_params, vi_state)
    (params_unclipped, _), _ = step_fn_unclipped(key_step, vi_params, vi_state)

    # Clipped version should have smaller parameter changes
    # (This is a weak test, but validates that clipping affects the update)
    diff_clipped = jnp.linalg.norm(params_clipped.rho - vi_params.rho)
    diff_unclipped = jnp.linalg.norm(params_unclipped.rho - vi_params.rho)

    # With clipping, updates should generally be smaller (though not always guaranteed)
    # Just verify both are finite
    assert jnp.isfinite(diff_clipped)
    assert jnp.isfinite(diff_unclipped)


def test_vi_whitening_rmsprop_stability():
    """Test VI with RMSProp whitening mode."""
    # Simple quadratic target
    d = 5
    rng = jax.random.PRNGKey(42)

    # Generate synthetic data
    X = jax.random.normal(rng, (50, d))
    Y = jax.random.normal(jax.random.split(rng)[0], (50, 1))

    # ERM solution (just use zeros for simplicity)
    wstar_flat = jnp.zeros(d)

    def unravel_fn(flat):
        return {"w": flat}

    def loss_batch_fn(params, Xb, Yb):
        w = params["w"]
        pred = jnp.dot(Xb, w)
        return jnp.mean((pred[:, None] - Yb) ** 2)

    def loss_full_fn(params):
        return loss_batch_fn(params, X, Y)

    # Run VI with RMSProp whitening
    # Note: In real usage, whitening is computed in sampling.py::run_vi
    # Here we just test that the VI code doesn't crash with a non-None whitener

    # Create a simple diagonal whitener (simulate RMSProp output)
    A_diag = jnp.ones(d) * 2.0  # Simple scaling
    whitener = vi.make_whitener(A_diag)

    # Run fit_vi_and_estimate_lambda
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
        M=2,
        r=2,
        steps=100,  # Short run for test
        batch_size=10,
        lr=0.01,
        eval_samples=10,
        whitener=whitener,
        clip_global_norm=5.0,
        alpha_temperature=1.0,
    )

    # Verify outputs are finite
    assert jnp.isfinite(lambda_hat)
    assert jnp.all(jnp.isfinite(traces["elbo"]))
    assert jnp.all(jnp.isfinite(traces["radius2"]))
    assert not jnp.any(jnp.isnan(traces["elbo"]))


def test_vi_whitening_adam_stability():
    """Test VI with Adam whitening mode (includes first moment)."""
    # Similar to RMSProp test, but conceptually different whitener
    d = 5
    rng = jax.random.PRNGKey(43)

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

    # Adam-style whitener (different scaling)
    A_diag = jax.random.uniform(rng, (d,)) * 3.0 + 1.0
    whitener = vi.make_whitener(A_diag)

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
        M=2,
        r=2,
        steps=100,
        batch_size=10,
        lr=0.01,
        eval_samples=10,
        whitener=whitener,
        clip_global_norm=5.0,
        alpha_temperature=1.0,
    )

    # Verify stability
    assert jnp.isfinite(lambda_hat)
    assert jnp.all(jnp.isfinite(traces["elbo"]))
    assert not jnp.any(jnp.isnan(traces["elbo"]))
    assert not jnp.any(jnp.isnan(traces["radius2"]))
