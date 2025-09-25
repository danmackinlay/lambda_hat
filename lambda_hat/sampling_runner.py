"""
Sampling runner module for executing MCMC samplers.

Extracted from the one-phase workflow to be used by the two-stage sampling process.
"""

import logging
from typing import Dict, Any

import jax
from omegaconf import DictConfig

from lambda_hat.posterior import (
    make_grad_loss_minibatch,
    compute_beta_gamma,
    make_logpost_and_score,
    make_logdensity_for_mclmc,
)
from lambda_hat.sampling import run_hmc, run_sgld, run_mclmc

log = logging.getLogger(__name__)


def run_sampler(
    sampler_name: str, cfg: DictConfig, target, key: jax.random.PRNGKey
) -> Dict[str, Any]:
    """Run a specific sampler and return results

    Args:
        sampler_name: Name of the sampler to run ('sgld', 'hmc', 'mclmc')
        cfg: Sample-stage configuration (from conf/sample/)
        target: TargetBundle with loss functions and parameters
        key: JAX PRNG key

    Returns:
        Dictionary with traces, beta, gamma, and sampler config
    """
    log.info(f"Running {sampler_name} sampler...")

    # Compute beta and gamma from sample config
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

        # Run SGLD
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

        # Run MCLMC
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