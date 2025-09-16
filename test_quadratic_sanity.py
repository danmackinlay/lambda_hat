#!/usr/bin/env python3
"""
Quadratic target sanity test to detect factor-of-2 bugs in samplers.

For a pure quadratic L_n(θ) = 0.5||θ||² with γ=0, β>0, the tempered
target is Gaussian and E[L_n] = d/(2nβ), so LLC = nβ(E[L_n] - 0) = d/2.

If any sampler reports LLC ≈ d instead of d/2, it suggests a hidden
factor-of-2 in how L_n is computed or scaled.
"""
import jax
import jax.numpy as jnp
import numpy as np
from llc.config import Config
from llc.pipeline import run_one

def make_quadratic_config(d: int = 4, beta: float = 1.0) -> Config:
    """Create config for pure quadratic test L_n(θ) = 0.5||θ||²"""
    return Config(
        # Model: just parameters (no actual network)
        in_dim=d,
        out_dim=1,
        depth=0,  # No hidden layers
        widths=[],
        target_params=d,  # d parameters total

        # Data: dummy (won't be used with our custom loss)
        n_data=100,

        # Posterior: tempering only, no spatial prior
        beta_mode="fixed",
        beta0=beta,
        gamma=0.0,  # No spatial localization

        # Quick sampling for test
        chains=2,
        sgld_steps=500,
        sgld_warmup=100,
        sgld_eval_every=10,
        hmc_draws=200,
        hmc_warmup=50,
        hmc_eval_every=2,
        mclmc_draws=300,
        mclmc_eval_every=3,

        # Use all samplers
        samplers=["sgld", "hmc", "mclmc"],

        # Save results
        save_plots=False,
        save_manifest=False,
        save_readme_snippet=False,
    )

def run_quadratic_test():
    """Run quadratic sanity test on all samplers"""
    d = 4  # 4-dimensional test
    expected_llc = d / 2.0  # Theoretical value: d/2

    print(f"Running quadratic sanity test with d={d}")
    print(f"Expected LLC = d/2 = {expected_llc}")
    print("=" * 50)

    cfg = make_quadratic_config(d)

    # We need to monkey-patch the loss function to use pure quadratic
    # This is a bit of a hack but allows us to test without major refactoring
    original_make_loss_fns = None

    try:
        from llc import losses
        original_make_loss_fns = losses.make_loss_fns

        def quadratic_loss_fns(unravel_fn, cfg, X, Y):
            """Pure quadratic loss: L_n(θ) = 0.5||θ||²"""
            def loss_full(theta_flat):
                return 0.5 * jnp.sum(theta_flat ** 2)

            def loss_batch(theta_flat, batch):
                # For quadratic, batch doesn't matter
                return loss_full(theta_flat)

            return loss_full, loss_batch

        # Monkey patch
        losses.make_loss_fns = quadratic_loss_fns

        # Run the test
        result = run_one(cfg, save_artifacts=False, skip_if_exists=False)

        print("\nResults:")
        print("-" * 30)

        results = {}
        for sampler in ["sgld", "hmc", "mclmc"]:
            llc_key = f"{sampler}_llc_mean"
            if llc_key in result.metrics:
                llc_value = result.metrics[llc_key]
                ratio = llc_value / expected_llc
                results[sampler] = (llc_value, ratio)

                status = "✓ GOOD" if 0.8 <= ratio <= 1.2 else "✗ BAD"
                print(f"{sampler.upper():6s}: LLC = {llc_value:.3f}, ratio = {ratio:.2f} {status}")
            else:
                print(f"{sampler.upper():6s}: No result found")

        print(f"\nExpected: LLC ≈ {expected_llc:.3f} (ratio ≈ 1.0)")
        print("If any sampler shows ratio ≈ 2.0, it has a factor-of-2 bug")

        # Check for factor-of-2 issues
        bad_samplers = []
        for sampler, (llc_value, ratio) in results.items():
            if ratio > 1.8:  # Close to 2x
                bad_samplers.append(sampler)

        if bad_samplers:
            print(f"\n⚠️  POTENTIAL FACTOR-OF-2 BUG in: {', '.join(bad_samplers)}")
            return False
        else:
            print(f"\n✅ All samplers passed the sanity test")
            return True

    finally:
        # Restore original function
        if original_make_loss_fns is not None:
            losses.make_loss_fns = original_make_loss_fns

if __name__ == "__main__":
    success = run_quadratic_test()
    exit(0 if success else 1)