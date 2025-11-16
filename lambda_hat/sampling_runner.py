"""
Sampling runner module for executing MCMC samplers.

NEW DESIGN: All samplers work with flat R^D vectors via Posterior interface.
"""

import logging
from typing import Any, Dict

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from lambda_hat.equinox_adapter import ensure_dtype, vectorise_model
from lambda_hat.losses import make_loss_fns
from lambda_hat.posterior import (
    compute_beta_gamma,
    make_grad_loss_minibatch_flat,
    make_posterior,
)
from lambda_hat.samplers import run_hmc, run_mclmc, run_sgld, run_vi

log = logging.getLogger(__name__)


def run_sampler(
    sampler_name: str, cfg: DictConfig, target, key: jax.random.PRNGKey
) -> Dict[str, Any]:
    """Run a specific sampler and return results.

    Args:
        sampler_name: Name of the sampler to run ('sgld', 'hmc', 'mclmc', 'vi')
        cfg: Sample-stage configuration
        target: TargetBundle with loss functions and parameters
        key: JAX PRNG key

    Returns:
        Dictionary with traces, beta, gamma, and sampler config
    """
    log.info(f"Running {sampler_name} sampler...")

    # Derive n_data from TargetBundle
    n_data = target.X.shape[0]

    # Compute beta and gamma
    beta, gamma = compute_beta_gamma(cfg.posterior, target.d, n_data)
    log.info(f"Using beta={beta:.6f}, gamma={gamma:.6f} (n={n_data})")

    # Determine dtype for this sampler
    dtype_str = None
    if sampler_name == "hmc" or sampler_name == "mclmc":
        dtype_str = getattr(cfg.sampler.get(sampler_name, {}), "dtype", "float64")
    elif sampler_name == "sgld":
        dtype_str = getattr(cfg.sampler.sgld, "dtype", "float32")
    elif sampler_name == "vi":
        dtype_str = cfg.sampler.vi.dtype

    dtype = jnp.float32 if dtype_str == "float32" else jnp.float64

    # Cast model and data to sampler's dtype
    model = ensure_dtype(target.params0, dtype)
    X = target.X.astype(dtype)
    Y = target.Y.astype(dtype)

    # Vectorise model to flat space
    vm, flat0 = vectorise_model(model, dtype=dtype)
    log.info(f"Vectorised model: D={vm.size}, dtype={vm.dtype}")

    # Create loss function (model -> scalar)
    def predict_fn(m, x):
        return m(x)

    loss_full, loss_mini = make_loss_fns(
        predict_fn,
        X,
        Y,
        loss_type=cfg.posterior.loss,
        noise_scale=cfg.data.noise_scale if hasattr(cfg, "data") else 0.1,
        student_df=cfg.data.student_df if hasattr(cfg, "data") else 4.0,
    )

    # Create Posterior in flat space
    posterior = make_posterior(vm, flat0, loss_full, n_data, beta, gamma)

    # Dispatch to sampler with flat interface
    if sampler_name == "hmc":
        run_result = run_hmc(
            key=key,
            posterior=posterior,
            num_samples=cfg.sampler.hmc.draws,
            num_chains=cfg.sampler.chains,
            step_size=cfg.sampler.hmc.step_size,
            num_integration_steps=cfg.sampler.hmc.num_integration_steps,
            adaptation_steps=cfg.sampler.hmc.warmup,
            target_acceptance=cfg.sampler.hmc.target_acceptance,
            loss_full_fn=loss_full,
            n_data=n_data,
            beta=beta,
            L0=target.L0,
        )

    elif sampler_name == "mclmc":
        run_result = run_mclmc(
            key=key,
            posterior=posterior,
            num_samples=cfg.sampler.mclmc.draws,
            num_chains=cfg.sampler.chains,
            config=cfg.sampler.mclmc,
            loss_full_fn=loss_full,
            n_data=n_data,
            beta=beta,
            L0=target.L0,
        )

    elif sampler_name == "sgld":
        grad_loss_minibatch_flat = make_grad_loss_minibatch_flat(vm, loss_mini)

        run_result = run_sgld(
            key=key,
            posterior=posterior,
            data=(X, Y),
            config=cfg.sampler.sgld,
            num_chains=cfg.sampler.chains,
            grad_loss_minibatch=grad_loss_minibatch_flat,
            loss_full_fn=loss_full,
            n_data=n_data,
            beta=beta,
            gamma=gamma,
            L0=target.L0,
        )

    elif sampler_name == "vi":
        grad_loss_minibatch_flat = make_grad_loss_minibatch_flat(vm, loss_mini)

        run_result = run_vi(
            key=key,
            posterior=posterior,
            data=(X, Y),
            config=cfg.sampler.vi,
            num_chains=cfg.sampler.chains,
            grad_loss_minibatch=grad_loss_minibatch_flat,
            loss_full_fn=loss_full,
        )

    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    log.info(f"Completed {sampler_name} sampling")
    return {
        "traces": run_result.traces,
        "timings": run_result.timings,
        "work": run_result.work,
        "sampler_config": getattr(cfg.sampler, sampler_name),
        "beta": beta,
        "gamma": gamma,
    }
