#!/usr/bin/env python3
"""
Main LLC estimation function using optimized samplers.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import jax
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from lambda_hat.config import Config
from lambda_hat.targets import build_target, make_loss_full
from lambda_hat.posterior import (
    make_grad_loss_minibatch,
    compute_beta_gamma,
    make_logpost,
)
from lambda_hat.sampling import run_hmc, run_sgld, run_mclmc
from lambda_hat.analysis import analyze_traces
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


def run_sampler(
    sampler_name: str, cfg: Config, target, key: jax.random.PRNGKey
) -> Dict[str, Any]:
    """Run a specific sampler and return results"""
    log.info(f"Running {sampler_name} sampler...")

    # Compute beta and gamma
    beta, gamma = compute_beta_gamma(cfg.posterior, target.d, cfg.data.n_data)
    log.info(f"Using beta={beta:.6f}, gamma={gamma:.6f}")

    if sampler_name == "hmc":
        # Setup HMC
        loss_full = target.loss_full_f64
        loss_mini = target.loss_minibatch_f64
        params0 = target.params0_f64

        logdensity_fn = make_logpost(loss_full, params0, cfg.data.n_data, beta, gamma)

        # Create loss_full function for Ln recording
        loss_full_for_recording = make_loss_full(loss_full)

        # Run HMC (Updated call)
        run_result = run_hmc(
            key,
            logdensity_fn,
            params0,
            num_samples=cfg.sampler.hmc.draws,
            num_chains=cfg.sampler.chains,
            step_size=cfg.sampler.hmc.step_size,
            num_integration_steps=cfg.sampler.hmc.num_integration_steps,
            adaptation_steps=cfg.sampler.hmc.warmup,
            loss_full_fn=loss_full_for_recording,
        )

    elif sampler_name == "sgld":
        # Setup SGLD/pSGLD
        loss_mini = target.loss_minibatch_f32
        params0 = target.params0_f32  # Center for localization (w_0)
        initial_params = params0  # Start sampling near w_0

        # Use the new make_grad_loss_minibatch
        grad_loss_fn = make_grad_loss_minibatch(loss_mini)

        # Create loss_full function for Ln recording
        loss_full_for_recording = make_loss_full(target.loss_full_f32)

        # Run SGLD (Updated call)
        run_result = run_sgld(
            key,
            grad_loss_fn,
            initial_params=initial_params,
            params0=params0,
            data=(target.X, target.Y),
            config=cfg.sampler.sgld,
            num_chains=cfg.sampler.chains,
            beta=beta,
            gamma=gamma,
            loss_full_fn=loss_full_for_recording,
        )

    elif sampler_name == "mclmc":
        # Setup MCLMC
        loss_full = target.loss_full_f64
        params0 = target.params0_f64

        logdensity_fn = make_logpost(loss_full, params0, cfg.data.n_data, beta, gamma)

        # Create loss_full function for Ln recording
        loss_full_for_recording = make_loss_full(loss_full)

        # Run MCLMC (Updated call)
        run_result = run_mclmc(
            key,
            logdensity_fn,
            params0,
            num_samples=cfg.sampler.mclmc.draws,
            num_chains=cfg.sampler.chains,
            config=cfg.sampler.mclmc,
            loss_full_fn=loss_full_for_recording,
        )

    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    log.info(f"Completed {sampler_name} sampling")
    # Return the results including the new timings
    return {
        "traces": run_result.traces,
        "timings": run_result.timings,  # Add timings
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
    samplers_to_run = list(cfg.get("samplers_to_run", ["sgld", "hmc", "mclmc"]))
    results = {}

    for sampler_name in samplers_to_run:
        key, sampler_key = jax.random.split(key)
        start_time = time.time()

        try:
            sampler_results = run_sampler(sampler_name, cfg, target, sampler_key)
            elapsed_external = time.time() - start_time

            # Use internal precise timings if available
            if 'timings' in sampler_results and sampler_results['timings']:
                timings = sampler_results['timings']
                log.info(f"{sampler_name} completed. Total (Internal): {timings.get('total', 0.0):.2f}s | Adaptation: {timings.get('adaptation', 0.0):.2f}s | Sampling: {timings.get('sampling', 0.0):.2f}s")
                # We prioritize internal timing for the final result structure
                sampler_results["elapsed_time"] = timings.get('total', elapsed_external)
            else:
                log.info(f"{sampler_name} completed in {elapsed_external:.2f}s (external measurement)")
                sampler_results["elapsed_time"] = elapsed_external

            results[sampler_name] = sampler_results

        except Exception:
            log.exception(f"Failed to run {sampler_name}")
            continue

    # Analyze results
    log.info("Analyzing results...")
    analysis_results = {}
    inference_data = {} # Store InferenceData objects

    for sampler_name, sampler_data in results.items():
        try:
            traces = sampler_data["traces"]
            # Pass the timings dictionary to analyze_traces
            timings = sampler_data.get("timings", {})

            # Determine warmup draws to discard (burn-in)
            # HMC/MCLMC handle adaptation during sampling (warmup=0 for analysis).
            warmup_draws = 0
            if sampler_name == "sgld":
                # SGLD requires explicit burn-in. Convert warmup steps to recorded draws based on thinning.
                # Use the updated default (100) or the configured value
                eval_every = getattr(cfg.sampler.sgld, 'eval_every', 100)
                if eval_every > 0:
                    # cfg.sampler.sgld.warmup is the number of steps to discard.
                    warmup_draws = cfg.sampler.sgld.warmup // eval_every

            # Updated call to analyze_traces
            metrics, idata = analyze_traces(
                traces,
                target.L0,
                n_data=cfg.data.n_data,
                beta=sampler_data["beta"],
                warmup=warmup_draws,
                timings=timings,  # Pass timings
            )
            analysis_results[sampler_name] = metrics
            inference_data[sampler_name] = idata # Store idata

            # Update logging terminology
            log.info(f"{sampler_name} LLC (hat{{lambda}}) metrics:")
            log.info(f"  Mean: {metrics['llc_mean']:.6f}")
            log.info(f"  Std: {metrics['llc_std']:.6f}")
            if "ess" in metrics:
                log.info(f"  ESS: {metrics['ess']:.1f}")

        except Exception:
            log.exception(f"Failed to analyze {sampler_name}")
            continue

    # Save artifacts
    if cfg.output.save_plots:
        log.info("Saving artifacts...")
        # Use Hydra's recorded output dir
        hydra_output_dir = Path(HydraConfig.get().run.dir)

        # Updated call to save_run_artifacts
        save_run_artifacts(
            results=results,
            analysis_results=analysis_results,
            inference_data=inference_data, # Pass the new inference_data
            target=target,
            cfg=cfg,
            output_dir=hydra_output_dir,
        )

    log.info("LLC estimation completed successfully!")
