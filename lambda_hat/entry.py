#!/usr/bin/env python3
"""
Main LLC estimation script using Hydra for configuration management.

Usage:
    lambda-hat                              # Use default config
    lambda-hat sampler=fast                 # Use fast sampler preset
    lambda-hat model=small data=small       # Use small presets
    lambda-hat model.target_params=1000     # Override specific values
    lambda-hat --multirun sampler=base,fast # Run multiple configurations
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import hydra
import jax
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from lambda_hat.config import Config, setup_config
from lambda_hat.targets import build_target
from lambda_hat.posterior import (
    make_grad_loss_minibatch,
    compute_beta_gamma,
    make_logpost_and_score,
    make_logdensity_for_mclmc,
)
from lambda_hat.sampling import run_hmc, run_sgld, run_mclmc
from lambda_hat.analysis import compute_llc_metrics
from lambda_hat.artifacts import save_run_artifacts

# Setup logging
log = logging.getLogger(__name__)


def setup_jax_environment():
    """Configure JAX environment"""
    # Enable 64-bit precision for HMC/MCLMC
    jax.config.update("jax_enable_x64", True)

    # Set platform
    if jax.default_backend() == "gpu":
        log.info("Using GPU backend")
    else:
        log.info("Using CPU backend")

    # Configure memory preallocation
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def run_sampler(
    sampler_name: str, cfg: Config, target, key: jax.random.PRNGKey
) -> Dict[str, Any]:
    """Run a specific sampler and return results"""
    log.info(f"Running {sampler_name} sampler...")

    # Compute beta and gamma (Updated signature)
    beta, gamma = compute_beta_gamma(cfg.posterior, target.d, cfg.data.n_data)
    log.info(f"Using beta={beta:.6f}, gamma={gamma:.6f}")

    if sampler_name == "hmc":
        # Setup HMC
        loss_full = target.loss_full_f64
        loss_mini = target.loss_minibatch_f64
        params0 = target.params0_f64

        logpost_and_grad, _ = make_logpost_and_score(
            loss_full, loss_mini, params0, cfg.data.n_data, beta, gamma
        )
        logdensity_fn = lambda params: logpost_and_grad(params)[0]

        # Run HMC
        traces = run_hmc(
            key,
            logdensity_fn,
            params0,
            num_samples=cfg.sampler.hmc.draws,
            num_chains=cfg.sampler.chains,
            step_size=cfg.sampler.hmc.step_size,
            num_integration_steps=cfg.sampler.hmc.num_integration_steps,
            adaptation_steps=cfg.sampler.hmc.warmup,
        )

    elif sampler_name == "sgld":
        # Setup SGLD/pSGLD
        loss_mini = target.loss_minibatch_f32
        params0 = target.params0_f32  # Center for localization (w_0)
        initial_params = params0  # Start sampling near w_0

        # Use the new make_grad_loss_minibatch
        grad_loss_fn = make_grad_loss_minibatch(loss_mini)

        # Run SGLD (Updated signature)
        traces = run_sgld(
            key,
            grad_loss_fn,
            initial_params=initial_params,
            params0=params0,
            data=(target.X_f32, target.Y_f32),
            config=cfg.sampler.sgld,
            num_chains=cfg.sampler.chains,
            beta=beta,
            gamma=gamma,
        )

    elif sampler_name == "mclmc":
        # Setup MCLMC
        loss_full = target.loss_full_f64
        params0 = target.params0_f64

        logdensity_fn = make_logdensity_for_mclmc(
            loss_full, params0, cfg.data.n_data, beta, gamma
        )

        # Run MCLMC (Updated signature)
        traces = run_mclmc(
            key,
            logdensity_fn,
            params0,
            num_samples=cfg.sampler.mclmc.draws,
            num_chains=cfg.sampler.chains,
            config=cfg.sampler.mclmc,  # Pass the config object
        )

    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    log.info(f"Completed {sampler_name} sampling")
    return {
        "traces": traces,
        "sampler_config": getattr(cfg.sampler, sampler_name),
        "beta": beta,
        "gamma": gamma,
    }


def main(cfg: Config) -> None:
    """Main LLC estimation function"""
    log.info("=== LLC Hydra Pipeline ===")
    log.info("Using CPU/GPU backend according to cfg (and JAX flags)")
    # Use OmegaConf just for printing the structured config
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Setup JAX
    setup_jax_environment()

    # Setup RNG
    key = jax.random.PRNGKey(cfg.seed)
    key, target_key = jax.random.split(key)

    # Build target
    log.info("Building target...")
    target = build_target(target_key, cfg)
    log.info(f"Target built: d={target.d}, L0={target.L0:.6f}")

    # Run samplers
    samplers_to_run = ["sgld", "hmc", "mclmc"]  # Default order
    results = {}

    for sampler_name in samplers_to_run:
        key, sampler_key = jax.random.split(key)
        start_time = time.time()

        try:
            sampler_results = run_sampler(sampler_name, cfg, target, sampler_key)
            elapsed = time.time() - start_time

            log.info(f"{sampler_name} completed in {elapsed:.2f}s")
            sampler_results["elapsed_time"] = elapsed
            results[sampler_name] = sampler_results

        except Exception as e:
            log.error(f"Failed to run {sampler_name}: {e}")
            continue

    # Analyze results
    log.info("Analyzing results...")
    analysis_results = {}

    for sampler_name, sampler_data in results.items():
        try:
            # Compute LLC metrics
            traces = sampler_data["traces"]
            loss_full = (
                target.loss_full_f64
                if sampler_name in ["hmc", "mclmc"]
                else target.loss_full_f32
            )

            # Determine warmup steps
            warmup_steps = 0
            if sampler_name == "sgld":
                # SGLD uses the configured warmup for burn-in
                warmup_steps = cfg.sampler.sgld.warmup
            # Note: HMC/MCLMC handle warmup/adaptation during the sampling phase.

            # Updated call to compute_llc_metrics
            metrics = compute_llc_metrics(
                traces,
                loss_full,
                target.L0,
                n_data=cfg.data.n_data,
                beta=sampler_data["beta"],
                warmup=warmup_steps,  # Pass warmup steps
            )
            analysis_results[sampler_name] = metrics

            # Update logging terminology
            log.info(f"{sampler_name} LLC (hat{{lambda}}) metrics:")
            log.info(f"  Mean: {metrics['llc_mean']:.6f}")
            log.info(f"  Std: {metrics['llc_std']:.6f}")
            if "ess" in metrics:
                log.info(f"  ESS: {metrics['ess']:.1f}")

        except Exception as e:
            log.error(f"Failed to analyze {sampler_name}: {e}")
            continue

    # Save artifacts
    if cfg.output.save_plots:
        log.info("Saving artifacts...")
        # Use Hydra's recorded output dir rather than assuming CWD
        hydra_output_dir = Path(HydraConfig.get().run.dir)

        # Optional guardrail: verify we're in the expected Hydra output directory
        try:
            assert Path.cwd().resolve() == hydra_output_dir.resolve()
        except AssertionError:
            log.warning(
                f"Working directory mismatch: cwd={Path.cwd()}, hydra_dir={hydra_output_dir}"
            )

        save_run_artifacts(
            results=results,
            analysis_results=analysis_results,
            target=target,
            cfg=cfg,
            output_dir=hydra_output_dir,
        )

    log.info("LLC estimation completed successfully!")


if __name__ == "__main__":
    # This allows running as: python -m lambda_hat.entry
    # Setup configuration and run with Hydra decorator
    import hydra

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def main_with_hydra(cfg: Config) -> None:
        return main(cfg)

    setup_config()
    main_with_hydra()
