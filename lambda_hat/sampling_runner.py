"""
Sampling runner module for executing MCMC samplers.

Extracted from the one-phase workflow to be used by the two-stage sampling process.
"""

import logging
from typing import Dict, Any

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from lambda_hat.posterior import (
    make_grad_loss_minibatch,
    compute_beta_gamma,
    make_logpost,
)
from lambda_hat.sampling import run_hmc, run_sgld, run_mclmc
from lambda_hat.losses import make_loss_fns, as_dtype

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

    # Derive n_data from simplified TargetBundle
    n_data = target.X.shape[0]

    # Compute beta and gamma using explicit n_data (removes cfg.data.n_data dependency)
    beta, gamma = compute_beta_gamma(cfg.posterior, target.d, n_data)
    log.info(f"Using beta={beta:.6f}, gamma={gamma:.6f} (n={n_data})")

    if sampler_name == "hmc":
        # Setup HMC - cast to f64 for precision
        X_f64, Y_f64 = as_dtype(target.X, "float64"), as_dtype(target.Y, "float64")
        params0_f64 = jax.tree.map(lambda x: x.astype(jnp.float64), target.params0)

        # Create f64 loss function
        loss_full, _ = make_loss_fns(
            target.model.apply,
            X_f64,
            Y_f64,
            loss_type=cfg.posterior.loss,
            noise_scale=cfg.data.noise_scale if hasattr(cfg, "data") else 0.1,
            student_df=cfg.data.student_df if hasattr(cfg, "data") else 4.0,
        )

        # Use the modern make_logpost function
        logdensity_fn = make_logpost(loss_full, params0_f64, n_data, beta, gamma)

        # Run HMC
        traces = run_hmc(
            key,
            logdensity_fn,
            params0_f64,
            num_samples=cfg.sampler.hmc.draws,
            num_chains=cfg.sampler.chains,
            step_size=cfg.sampler.hmc.step_size,
            num_integration_steps=cfg.sampler.hmc.num_integration_steps,
            adaptation_steps=cfg.sampler.hmc.warmup,
            loss_full=loss_full,  # Pass loss function for Ln recording
        )

    elif sampler_name == "sgld":
        # Setup SGLD/pSGLD - data already in f32, params may need casting
        params0_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), target.params0)
        initial_params = params0_f32  # Start sampling near w_0

        # Create f32 loss function (minibatch version)
        _, loss_mini = make_loss_fns(
            target.model.apply,
            target.X,
            target.Y,
            loss_type=cfg.posterior.loss,
            noise_scale=cfg.data.noise_scale if hasattr(cfg, "data") else 0.1,
            student_df=cfg.data.student_df if hasattr(cfg, "data") else 4.0,
        )

        # Use the new make_grad_loss_minibatch
        grad_loss_fn = make_grad_loss_minibatch(loss_mini)

        # Run SGLD
        traces = run_sgld(
            key,
            grad_loss_fn,
            initial_params=initial_params,
            params0=params0_f32,
            data=(target.X, target.Y),
            config=cfg.sampler.sgld,
            num_chains=cfg.sampler.chains,
            beta=beta,
            gamma=gamma,
            loss_full=target.loss_full,  # Pass loss function for Ln recording
        )

    elif sampler_name == "mclmc":
        # Setup MCLMC - cast to f64 for precision
        X_f64, Y_f64 = as_dtype(target.X, "float64"), as_dtype(target.Y, "float64")
        params0_f64 = jax.tree.map(lambda x: x.astype(jnp.float64), target.params0)

        # Create f64 loss function
        loss_full, _ = make_loss_fns(
            target.model.apply,
            X_f64,
            Y_f64,
            loss_type=cfg.posterior.loss,
            noise_scale=cfg.data.noise_scale if hasattr(cfg, "data") else 0.1,
            student_df=cfg.data.student_df if hasattr(cfg, "data") else 4.0,
        )

        # Use the modern make_logpost function
        logdensity_fn = make_logpost(loss_full, params0_f64, n_data, beta, gamma)

        # Run MCLMC
        traces = run_mclmc(
            key,
            logdensity_fn,
            params0_f64,
            num_samples=cfg.sampler.mclmc.draws,
            num_chains=cfg.sampler.chains,
            config=cfg.sampler.mclmc,  # Pass the config object
            loss_full=loss_full,  # Pass loss function for Ln recording
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
