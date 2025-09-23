# llc/pipeline.py
"""
Batched-only pipeline. Uses registry-based sampler loop with centralized metrics
via analysis.py. Saves only {sampler}.nc, metrics.json, config.json, and L0.txt.
No legacy saves, manifests, or galleries.
"""

from __future__ import annotations
import logging
import os

import jax.numpy as jnp
from jax import random, jit, device_put
import jax
import numpy as np

from .config import Config
from .cache import run_id, run_family_id, load_cached_outputs
from .types import RunOutputs

# Set up logger
logger = logging.getLogger(__name__)
from .targets import build_target
from .posterior import (
    compute_beta_gamma,
    make_logpost_and_score,
    make_logdensity_for_mclmc,
)
from .runners import (
    run_sgld_online_batched,
    run_sghmc_online_batched,
    run_hmc_online_batched,
    run_mclmc_online_batched,
)
from .convert import to_idata
import arviz as az
from .samplers.base import prepare_diag_targets
from .analysis import llc_point_se, efficiency_metrics
from .artifacts import (
    save_L0,
    save_config,
    save_metrics,
)


def run_one(
    cfg: Config, *, save_artifacts: bool = True, skip_if_exists: bool = True
) -> RunOutputs:
    """
    Run one complete experiment: build data → train ERM → run all samplers → compute metrics.

    This is the single source of truth that replaces the scattered logic in main.py,
    experiments.py, and tasks.py.

    Args:
        cfg: Experiment configuration
        save_artifacts: Whether to save plots, data files, and HTML gallery
        skip_if_exists: Whether to skip if results already exist

    Returns:
        RunOutputs with metrics, histories, run_dir, and L0
    """
    logger = logging.getLogger(__name__)

    # Compute deterministic run ID and family ID
    rid = run_id(cfg)
    fid = run_family_id(cfg)
    # Use canonical runs/ directory everywhere (local and Modal)
    if hasattr(cfg, "runs_dir") and cfg.runs_dir.endswith("/runs"):
        # Modal path: cfg.runs_dir="/runs" -> run_dir="/runs/rid"
        run_dir = os.path.join(cfg.runs_dir, rid)
    else:
        # Default: use runs/ under current directory
        run_dir = os.path.join("runs", rid)

    # Check if we should skip (results already exist)
    if skip_if_exists and os.path.exists(os.path.join(run_dir, "metrics.json")):
        cached = load_cached_outputs(run_dir)
        if cached:
            logger.info(f"Skipping run {rid} - results already exist in {run_dir}")
            return RunOutputs(
                run_dir=run_dir,
                metrics=cached["metrics"],
                histories={},  # Don't reload full histories for cache hits
                L0=cached["L0"],
            )

    # Create run directory and save config immediately
    if save_artifacts:
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Run ID: {rid}")
        logger.info(f"Run will be saved to: {run_dir}")
        # Save config up-front so it's available even if we crash
        save_config(run_dir, cfg)
    else:
        run_dir = ""  # Don't save if not requested

    logger.info("Building target")
    key = random.PRNGKey(cfg.seed)

    # Build a self-contained target (NN, quadratic, …)
    bundle = build_target(key, cfg)

    theta0_f32 = bundle.theta0_f32
    theta0_f64 = bundle.theta0_f64
    X_f32, Y_f32, X_f64, Y_f64 = bundle.X_f32, bundle.Y_f32, bundle.X_f64, bundle.Y_f64
    dim = bundle.d
    logger.info(f"Parameter dimension: {dim:,d}")

    # Move data to GPU if CUDA is enabled
    theta0_f32 = device_put(theta0_f32)
    theta0_f64 = device_put(theta0_f64)
    X_f32 = device_put(X_f32)
    Y_f32 = device_put(Y_f32)
    X_f64 = device_put(X_f64)
    Y_f64 = device_put(Y_f64)

    beta, gamma = compute_beta_gamma(cfg, dim)
    logger.info(f"beta={beta:.6g} gamma={gamma:.6g}")

    # Loss functions supplied by the target
    loss_full_f32, loss_minibatch_f32 = bundle.loss_full_f32, bundle.loss_minibatch_f32
    loss_full_f64, loss_minibatch_f64 = bundle.loss_full_f64, bundle.loss_minibatch_f64

    # log posterior & gradient factories for each dtype
    logpost_and_grad_f32, grad_logpost_minibatch_f32 = make_logpost_and_score(
        loss_full_f32, loss_minibatch_f32, theta0_f32, cfg.n_data, beta, gamma
    )
    logpost_and_grad_f64, grad_logpost_minibatch_f64 = make_logpost_and_score(
        loss_full_f64, loss_minibatch_f64, theta0_f64, cfg.n_data, beta, gamma
    )

    # L0 at the empirical minimizer (from target bundle)
    L0 = float(bundle.L0)
    logger.info(f"L0 at empirical minimizer: {L0:.6f}")

    # JIT compile the loss evaluator for LLC computation (scalar θ)
    Ln_full64 = jit(loss_full_f64)
    # Provide a batched evaluator over chains: theta[C, d] -> Ln[C]
    Ln_full64_vmapped = jit(jax.vmap(loss_full_f64))

    # Prepare diagnostic targets based on config
    diag_targets = prepare_diag_targets(dim, cfg)

    # Sampler registry with init and run functions
    SAMPLERS = {
        "sgld": {
            "init": lambda key, d, theta0_f32, cfg: (
                theta0_f32
                + 0.01 * random.normal(key, (cfg.chains, d)).astype(jnp.float32)
            ),
            "run": lambda key, init, env: run_sgld_online_batched(
                key,
                init,
                env["grad_logpost_minibatch_f32"],
                env["X_f32"],
                env["Y_f32"],
                cfg.n_data,
                cfg.sgld_step_size,
                cfg.sgld_steps,
                cfg.sgld_warmup,
                cfg.sgld_batch_size,
                cfg.sgld_eval_every,
                cfg.sgld_thin,
                env["Ln_full64_vmapped"],
                precond_mode=getattr(cfg, "sgld_precond", "none"),
                beta1=getattr(cfg, "sgld_beta1", 0.9),
                beta2=getattr(cfg, "sgld_beta2", 0.999),
                eps=getattr(cfg, "sgld_eps", 1e-8),
                bias_correction=getattr(cfg, "sgld_bias_correction", True),
                **env["diag_targets"],
            ),
        },
        "sghmc": {
            "init": lambda key, d, theta0_f32, cfg: (
                theta0_f32
                + 0.01 * random.normal(key, (cfg.chains, d)).astype(jnp.float32)
            ),
            "run": lambda key, init, env: run_sghmc_online_batched(
                key,
                init,
                env["grad_logpost_minibatch_f32"],
                env["X_f32"],
                env["Y_f32"],
                cfg.n_data,
                cfg.sghmc_step_size,
                cfg.sghmc_temperature,
                cfg.sghmc_steps,
                cfg.sghmc_eval_every,
                cfg.sghmc_thin,
                cfg.sghmc_batch_size,
                env["Ln_full64_vmapped"],
                **env["diag_targets"],
            ),
        },
        "hmc": {
            "init": lambda key, d, theta0_f64, cfg: (
                theta0_f64 + 0.01 * random.normal(key, (cfg.chains, d))
            ),
            "run": lambda key, init, env: run_hmc_online_batched(
                key,
                init,
                env["logpost_and_grad_f64"],
                cfg.hmc_draws,
                cfg.hmc_warmup,
                cfg.hmc_num_integration_steps,
                cfg.hmc_eval_every,
                cfg.hmc_thin,
                env["Ln_full64_vmapped"],
                **env["diag_targets"],
            ),
        },
        "mclmc": {
            "init": lambda key, d, theta0_f64, cfg: (
                theta0_f64 + 0.01 * random.normal(key, (cfg.chains, d))
            ),
            "run": lambda key, init, env: run_mclmc_online_batched(
                key,
                init,
                env["logdensity_mclmc"],
                cfg.mclmc_draws,
                cfg.mclmc_eval_every,
                cfg.mclmc_thin,
                env["Ln_full64_vmapped"],
                tuner_steps=cfg.mclmc_tune_steps,
                diagonal_preconditioning=cfg.mclmc_diagonal_preconditioning,
                desired_energy_var=cfg.mclmc_desired_energy_var,
                integrator_name=cfg.mclmc_integrator,
                **env["diag_targets"],
            ),
        },
    }

    # Environment dictionary with all shared data
    env = dict(
        Ln_full64=Ln_full64,
        Ln_full64_vmapped=Ln_full64_vmapped,
        grad_logpost_minibatch_f32=grad_logpost_minibatch_f32,
        logpost_and_grad_f64=logpost_and_grad_f64,
        logdensity_mclmc=make_logdensity_for_mclmc(
            loss_full_f64, theta0_f64, cfg.n_data, beta, gamma
        ),
        X_f32=X_f32,
        Y_f32=Y_f32,
        diag_targets=diag_targets,
    )

    # Storage for results
    all_metrics = {}
    histories = {}

    # Run samplers using registry-based loop
    for name in cfg.samplers:
        if name not in SAMPLERS:
            continue

        # Optional: Skip if already completed (micro-resume for partial runs)
        nc_path = f"{run_dir}/{name}.nc" if run_dir else None
        if save_artifacts and nc_path and os.path.exists(nc_path):
            try:
                logger.info(f"Found existing {name}.nc, loading results...")
                idata = az.from_netcdf(nc_path)
                # Compute metrics from loaded data
                m_core = llc_point_se(idata)
                m_eff = efficiency_metrics(
                    idata=idata,
                    timings={"sampling": np.nan, "warmup": np.nan},  # Unknown on reload
                    work={"n_full_loss": np.nan},  # Optional on reload
                    n_data=cfg.n_data,
                    sgld_batch=cfg.sgld_batch_size if name == "sgld" else (cfg.sghmc_batch_size if name == "sghmc" else None),
                )
                # Merge metrics and continue to next sampler
                for k2, v in {**m_core, **m_eff}.items():
                    all_metrics[f"{name}_{k2}"] = v
                continue
            except Exception as e:
                logger.info(f"Failed to load {name}.nc: {e}, re-running sampler")

        logger.info(f"Running {name.upper()} (BlackJAX, online)")
        k = random.fold_in(key, hash(name) & 0xFFFF)
        init = SAMPLERS[name]["init"](
            k, dim, theta0_f32 if name == "sgld" else theta0_f64, cfg
        )
        init = device_put(init)
        res = SAMPLERS[name]["run"](k, init, env)

        # Build idata (once; reuse for metrics + later save)
        idata = to_idata(
            Ln_histories=res.Ln_histories,
            theta_thin=res.theta_thin,
            acceptance=res.acceptance,
            energy=res.energy,
            n=cfg.n_data,
            beta=beta,
            L0=L0,
        )

        # Metrics
        m_core = llc_point_se(idata)
        m_eff = efficiency_metrics(
            idata=idata,
            timings=res.timings,
            work=res.work,
            n_data=cfg.n_data,
            sgld_batch=(cfg.sgld_batch_size if name == "sgld" else (cfg.sghmc_batch_size if name == "sghmc" else None)),
        )

        logger.info(
            f"{name.upper()} LLC: {m_core['llc_mean']:.4f} ± {m_core['llc_se']:.4f} (ESS: {int(m_core['ess_bulk']):.1f})"
        )

        # Optional: HMC mean acceptance (scalar)
        if name == "hmc" and res.acceptance:
            try:
                acc_scalar = float(
                    np.nanmean([a.mean() for a in res.acceptance if a.size])
                )
                m_eff["mean_acceptance"] = acc_scalar
                logger.info(f"HMC acceptance rate (mean over chains/draws): {acc_scalar:.3f}")
            except Exception:
                pass

        # prefix + collect
        for k2, v in {**m_core, **m_eff}.items():
            all_metrics[f"{name}_{k2}"] = v
        histories[name] = res.Ln_histories

        # Save idata and metrics incrementally (allows partial recovery)
        if save_artifacts and run_dir:
            idata.attrs.update(
                {
                    "n_data": int(cfg.n_data),
                    "beta": float(beta),
                    "L0": float(L0),
                    "sampler": name,
                }
            )
            az.to_netcdf(idata, f"{run_dir}/{name}.nc")
            # Save metrics after each sampler completes
            all_metrics_out = dict(all_metrics)
            all_metrics_out["family_id"] = fid
            save_metrics(run_dir, all_metrics_out)

    # Save final artifacts if requested
    if save_artifacts and run_dir:
        logger.info("Saving final run outputs")
        save_L0(run_dir, L0)
        all_metrics_out = dict(all_metrics)
        all_metrics_out["family_id"] = fid
        save_metrics(run_dir, all_metrics_out)  # Final save with all samplers

    return RunOutputs(
        run_dir=run_dir,
        metrics=all_metrics,
        histories=histories,
        L0=L0,
    )
