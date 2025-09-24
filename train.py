#!/usr/bin/env python3
"""
Main training script using Hydra for configuration management.

Usage:
    python train.py                              # Use default config
    python train.py sampler=fast                 # Use fast sampler preset
    python train.py model=small data=small       # Use small presets
    python train.py model.target_params=1000     # Override specific values
    python train.py --multirun sampler=base,fast # Run multiple configurations
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import hydra
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from llc.config import Config, setup_config
from llc.targets import build_target
from llc.posterior import make_logpost_and_score, make_logdensity_for_mclmc, compute_beta_gamma
from llc.sampling import run_hmc, run_sgld, run_mclmc
from llc.analysis import compute_llc_metrics
from llc.artifacts import save_run_artifacts

# Setup logging
log = logging.getLogger(__name__)


def setup_jax_environment():
    """Configure JAX environment"""
    # Set platform
    if jax.default_backend() == "gpu":
        log.info("Using GPU backend")
    else:
        log.info("Using CPU backend")

    # Configure memory preallocation
    import os
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def run_sampler(
    sampler_name: str,
    cfg: Config,
    target,
    key: jax.random.PRNGKey
) -> Dict[str, Any]:
    """Run a specific sampler and return results"""
    log.info(f"Running {sampler_name} sampler...")

    # Compute beta and gamma
    beta, gamma = compute_beta_gamma(cfg.posterior, target.d)
    log.info(f"Using beta={beta:.6f}, gamma={gamma:.6f}")

    if sampler_name == "hmc":
        # Setup HMC
        loss_full = target.loss_full_f64
        loss_mini = target.loss_minibatch_f64
        params0 = target.params0_f64

        logpost_and_grad, _ = make_logpost_and_score(
            loss_full, loss_mini, params0,
            cfg.data.n_data, beta, gamma
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
        # Setup SGLD
        loss_full = target.loss_full_f32
        loss_mini = target.loss_minibatch_f32
        params0 = target.params0_f32

        _, grad_logpost_fn = make_logpost_and_score(
            loss_full, loss_mini, params0,
            cfg.data.n_data, beta, gamma
        )

        # Run SGLD
        traces = run_sgld(
            key,
            grad_logpost_fn,
            params0,
            data=(target.X_f32, target.Y_f32),
            num_samples=cfg.sampler.sgld.steps,
            num_chains=cfg.sampler.chains,
            step_size=cfg.sampler.sgld.step_size,
            batch_size=cfg.sampler.sgld.batch_size,
        )

    elif sampler_name == "mclmc":
        # Setup MCLMC
        loss_full = target.loss_full_f64
        params0 = target.params0_f64

        logdensity_fn = make_logdensity_for_mclmc(
            loss_full, params0,
            cfg.data.n_data, beta, gamma
        )

        # Run MCLMC
        traces = run_mclmc(
            key,
            logdensity_fn,
            params0,
            num_samples=cfg.sampler.mclmc.draws,
            num_chains=cfg.sampler.chains,
            L=cfg.sampler.mclmc.L,
            step_size=cfg.sampler.mclmc.step_size,
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    # Convert to structured config
    cfg = OmegaConf.structured(cfg)

    log.info("=== LLC Hydra Training ===")
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
            loss_full = target.loss_full_f64 if sampler_name in ["hmc", "mclmc"] else target.loss_full_f32

            metrics = compute_llc_metrics(traces, loss_full, target.L0)
            analysis_results[sampler_name] = metrics

            log.info(f"{sampler_name} LLC metrics:")
            log.info(f"  Mean: {metrics['llc_mean']:.6f}")
            log.info(f"  Std: {metrics['llc_std']:.6f}")
            if 'ess' in metrics:
                log.info(f"  ESS: {metrics['ess']:.1f}")

        except Exception as e:
            log.error(f"Failed to analyze {sampler_name}: {e}")
            continue

    # Save artifacts
    if cfg.output.save_plots:
        log.info("Saving artifacts...")
        save_run_artifacts(
            results=results,
            analysis_results=analysis_results,
            target=target,
            cfg=cfg,
            output_dir=Path.cwd()
        )

    log.info("Training completed successfully!")


if __name__ == "__main__":
    # Setup configuration
    setup_config()
    main()