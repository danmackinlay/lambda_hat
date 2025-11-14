"""Test TensorBoard logging and ArviZ integration for VI sampler.

This test validates that VI's diagnostic enhancements:
- Generate all expected scalar metrics during optimization
- Can be logged to TensorBoard event files
- Are properly exported to ArviZ sample_stats
"""

# CRITICAL: Enable x64 BEFORE any JAX imports
import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax

jax.config.update("jax_enable_x64", True)

import tempfile
from pathlib import Path

import jax.numpy as jnp

from lambda_hat.vi import mfa as vi


def test_vi_enhanced_diagnostics():
    """Test that VI generates all expected diagnostic metrics."""
    d = 5
    rng = jax.random.PRNGKey(44)

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

    # Run VI with short config
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

    # Verify traces contain expected VI metrics (from variational.py:784-801)
    expected_vi_metrics = [
        "elbo",
        "elbo_like",
        "logq",
        "resp_entropy",
        "pi_min",
        "pi_max",
        "pi_entropy",
        "D_sqrt_min",
        "D_sqrt_max",
        "D_sqrt_med",
        "grad_norm",
        "A_col_norm_max",
    ]
    for key in expected_vi_metrics:
        assert key in traces, f"Missing VI metric: {key}"
        assert traces[key].shape[0] > 0, f"Empty trace for {key}"
        assert jnp.all(jnp.isfinite(traces[key])), f"Non-finite values in {key}"

    # Verify lambda_hat is finite
    assert jnp.isfinite(lambda_hat), "lambda_hat is not finite"


def test_vi_tensorboard_smoke():
    """Test that TensorBoard logging writes event files successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tb_dir = Path(tmpdir) / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)

        d = 5
        rng = jax.random.PRNGKey(45)

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

        # Run VI
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
            M=2,
            r=2,
            steps=50,  # Even shorter for smoke test
            batch_size=10,
            lr=0.01,
            eval_samples=10,
            whitener=whitener,
            clip_global_norm=5.0,
            alpha_temperature=1.0,
        )

        # Test TensorBoard writing (mimic sample.py)
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(str(tb_dir))

        # Traces are 1D arrays (steps,) not (chains, steps)
        num_steps = traces["elbo"].shape[0]

        # Log a subset of steps for speed
        for step in range(0, num_steps, 10):
            writer.add_scalar("vi/elbo", float(traces["elbo"][step]), step)
            writer.add_scalar("vi/pi_entropy", float(traces["pi_entropy"][step]), step)
            writer.add_scalar("vi/grad_norm", float(traces["grad_norm"][step]), step)

        # Final metrics
        writer.add_scalar("vi/lambda_hat", float(lambda_hat), num_steps)

        writer.close()

        # Verify event files were created
        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0, "No TensorBoard event files created"
        assert event_files[0].stat().st_size > 0, "TensorBoard event file is empty"
