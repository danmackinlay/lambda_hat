# llc/pipeline.py
"""
Lean pipeline. Always uses batched runners. Saves `{sampler}.nc`, `metrics.json`,
`config.json`, and `L0.txt`.
"""

from __future__ import annotations
from typing import Optional
import logging
import os

import jax.numpy as jnp
from jax import random, jit, device_put
import numpy as np

from .config import Config
from .cache import run_id, load_cached_outputs
from .types import RunOutputs, RunStats

# Set up logger
logger = logging.getLogger(__name__)
from .targets import build_target
from .posterior import (
    compute_beta_gamma,
    make_logpost_and_score,
    make_logdensity_for_mclmc,
)
from .runners import (
    RunStats,
    tic,
    toc,
    run_sgld_online_batched,
    run_hmc_online_batched,
    run_mclmc_online_batched,
)
from .convert import to_idata
import arviz as az
from .samplers.base import prepare_diag_targets
from llc.analysis import llc_point_se_from_histories as llc_mean_and_se_from_histories
from llc.analysis import llc_point_se, efficiency_metrics, generate_diagnostics
from .artifacts import (
    save_L0,
    save_config,
    save_metrics,
    generate_gallery_html,
)
from datetime import datetime



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
    # Compute deterministic run ID
    rid = run_id(cfg)
    # Use canonical runs/ directory, fallback to artifacts_dir for backward compatibility
    if hasattr(cfg, "artifacts_dir") and cfg.artifacts_dir.endswith("/artifacts"):
        base_dir = cfg.artifacts_dir.replace("/artifacts", "")
        run_dir = os.path.join(base_dir, "runs", rid)
    elif cfg.artifacts_dir == "artifacts":
        run_dir = os.path.join("runs", rid)
    else:
        run_dir = os.path.join(cfg.artifacts_dir, rid)

    # Check if we should skip (results already exist)
    if skip_if_exists and os.path.exists(os.path.join(run_dir, "metrics.json")):
        cached = load_cached_outputs(run_dir)
        if cached:
            print(f"Skipping run {rid} - results already exist in {run_dir}")
            return RunOutputs(
                run_dir=run_dir,
                metrics=cached["metrics"],
                histories={},  # Don't reload full histories for cache hits
                L0=cached["L0"],
            )

    # Create run directory
    if save_artifacts:
        os.makedirs(run_dir, exist_ok=True)
        print(f"Run ID: {rid}")
        print(f"Artifacts will be saved to: {run_dir}")
    else:
        run_dir = ""  # Don't save if not requested

    logger.info("Building target")
    stats = RunStats()

    # Build timing
    t0 = tic()
    key = random.PRNGKey(cfg.seed)

    # Build a self-contained target (NN, quadratic, …)
    bundle = build_target(key, cfg)
    stats.t_build = toc(t0)

    theta0_f32 = bundle.theta0_f32
    theta0_f64 = bundle.theta0_f64
    X_f32, Y_f32, X_f64, Y_f64 = bundle.X_f32, bundle.Y_f32, bundle.X_f64, bundle.Y_f64
    dim = bundle.d
    print(f"Parameter dimension: {dim:,d}")

    # Move data to GPU if CUDA is enabled
    theta0_f32 = device_put(theta0_f32)
    theta0_f64 = device_put(theta0_f64)
    X_f32 = device_put(X_f32)
    Y_f32 = device_put(Y_f32)
    X_f64 = device_put(X_f64)
    Y_f64 = device_put(Y_f64)

    beta, gamma = compute_beta_gamma(cfg, dim)
    print(f"beta={beta:.6g} gamma={gamma:.6g}")

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
    print(f"L0 at empirical minimizer: {L0:.6f}")
    target_se = 1.0  # used for time/fde-to-target across samplers

    # JIT compile the loss evaluator for LLC computation
    Ln_full64 = jit(loss_full_f64)

    # Prepare diagnostic targets based on config
    diag_targets = prepare_diag_targets(dim, cfg)

    # Storage for results
    all_metrics = {}
    histories = {}

    # ===== SGLD (Online) =====
    if "sgld" in getattr(cfg, "samplers", ["sgld"]):
        logger.info("Running SGLD (BlackJAX, online)")
        k_sgld = random.split(key, 1)[0]
        # simple overdispersed inits around w0
        init_thetas_sgld = theta0_f32 + 0.01 * random.normal(
            k_sgld, (cfg.chains, dim)
        ).astype(jnp.float32)
        init_thetas_sgld = device_put(init_thetas_sgld)

        # Always use batched SGLD execution
        res_sgld = run_sgld_online_batched(
            k_sgld,
            init_thetas_sgld,
            grad_logpost_minibatch_f32,
            X_f32,
            Y_f32,
            cfg.n_data,
            cfg.sgld_step_size,
            cfg.sgld_steps,
            cfg.sgld_warmup,
            cfg.sgld_batch_size,
            cfg.sgld_eval_every,
            cfg.sgld_thin,
            Ln_full64,
            use_tqdm=cfg.use_tqdm,
            progress_update_every=cfg.progress_update_every,
            **diag_targets,
            # NEW: optional preconditioning (works even if cfg lacks these fields)
            precond_mode=getattr(cfg, "sgld_precond", "none"),
            beta1=getattr(cfg, "sgld_beta1", 0.9),
            beta2=getattr(cfg, "sgld_beta2", 0.999),
            eps=getattr(cfg, "sgld_eps", 1e-8),
            bias_correction=getattr(cfg, "sgld_bias_correction", True),
        )

        # Build idata and compute centralized metrics
        idata_sgld = to_idata(
            Ln_histories=res_sgld.Ln_histories,
            theta_thin=res_sgld.theta_thin,
            acceptance=res_sgld.acceptance,
            energy=res_sgld.energy,
            n=cfg.n_data,
            beta=beta,
            L0=L0,
        )

        # Core LLC metrics
        m_core = llc_point_se(idata_sgld)
        print(f"SGLD LLC: {m_core['llc_mean']:.4f} ± {m_core['llc_se']:.4f} (ESS: {int(m_core['ess_bulk']):.1f})")

        # Efficiency metrics
        m_eff = efficiency_metrics(
            idata=idata_sgld,
            timings=res_sgld.timings,
            work=res_sgld.work,
            n_data=cfg.n_data,
            sgld_batch=cfg.sgld_batch_size,
        )

        # Store SGLD results with prefixed keys
        sgld_metrics = {}
        for k, v in m_core.items():
            sgld_metrics[f"sgld_{k}"] = v
        for k, v in m_eff.items():
            sgld_metrics[f"sgld_{k}"] = v
        # Add timing and work details
        sgld_metrics.update({
            "sgld_timing_warmup": res_sgld.timings.get("warmup", 0.0),
            "sgld_timing_sampling": res_sgld.timings.get("sampling", 0.0),
            "sgld_n_steps": res_sgld.work.get("n_minibatch_grads", 0),
            "sgld_n_full_loss": res_sgld.work.get("n_full_loss", 0),
        })
        all_metrics.update(sgld_metrics)
        histories["sgld"] = res_sgld.Ln_histories

    # ===== HMC (Online) =====
    if "hmc" in getattr(cfg, "samplers", ["sgld"]):
        logger.info("Running HMC (BlackJAX, online)")
        k_hmc = random.fold_in(key, 123)
        init_thetas_hmc = theta0_f64 + 0.01 * random.normal(k_hmc, (cfg.chains, dim))
        init_thetas_hmc = device_put(init_thetas_hmc)

        # Always use batched HMC execution
        res_hmc = run_hmc_online_batched(
            k_hmc,
            init_thetas_hmc,
            logpost_and_grad_f64,
            cfg.hmc_draws,
            cfg.hmc_warmup,
            cfg.hmc_num_integration_steps,
            cfg.hmc_eval_every,
            cfg.hmc_thin,
            Ln_full64,
            **diag_targets,
        )

        # Build idata and compute centralized metrics
        idata_hmc = to_idata(
            Ln_histories=res_hmc.Ln_histories,
            theta_thin=res_hmc.theta_thin,
            acceptance=res_hmc.acceptance,
            energy=res_hmc.energy,
            n=cfg.n_data,
            beta=beta,
            L0=L0,
        )

        # Core LLC metrics
        m_core = llc_point_se(idata_hmc)

        # Acceptance rate calculation
        vals = (
            [np.nanmean(a) for a in res_hmc.acceptance if a.size]
            if res_hmc.acceptance
            else []
        )
        mean_acc = float(np.nanmean(vals)) if vals else float("nan")

        print(f"HMC LLC: {m_core['llc_mean']:.4f} ± {m_core['llc_se']:.4f} (ESS: {int(m_core['ess_bulk']):.1f})")
        print(f"HMC acceptance rate (mean over chains/draws): {mean_acc:.3f}")

        # Efficiency metrics
        m_eff = efficiency_metrics(
            idata=idata_hmc,
            timings=res_hmc.timings,
            work=res_hmc.work,
            n_data=cfg.n_data,
            sgld_batch=None,  # HMC doesn't use minibatches
        )

        # Store HMC results with prefixed keys
        hmc_metrics = {}
        for k, v in m_core.items():
            hmc_metrics[f"hmc_{k}"] = v
        for k, v in m_eff.items():
            hmc_metrics[f"hmc_{k}"] = v
        # Add timing, work, and acceptance details
        hmc_metrics.update({
            "hmc_timing_warmup": res_hmc.timings.get("warmup", 0.0),
            "hmc_timing_sampling": res_hmc.timings.get("sampling", 0.0),
            "hmc_n_leapfrog_grads": res_hmc.work.get("n_leapfrog_grads", 0),
            "hmc_n_full_loss": res_hmc.work.get("n_full_loss", 0),
            "hmc_mean_acceptance": mean_acc,
        })
        all_metrics.update(hmc_metrics)
        histories["hmc"] = res_hmc.Ln_histories

    # ===== MCLMC (Online) =====
    if "mclmc" in getattr(cfg, "samplers", ["sgld"]):
        logger.info("Running MCLMC (BlackJAX, online)")
        k_mclmc = random.fold_in(key, 456)
        init_thetas_mclmc = theta0_f64 + 0.01 * random.normal(
            k_mclmc, (cfg.chains, dim)
        )
        init_thetas_mclmc = device_put(init_thetas_mclmc)

        # Create logdensity for MCLMC
        logdensity_mclmc = make_logdensity_for_mclmc(
            loss_full_f64, theta0_f64, cfg.n_data, beta, gamma
        )

        # Always use batched MCLMC execution
        res_mclmc = run_mclmc_online_batched(
            k_mclmc,
            init_thetas_mclmc,
            logdensity_mclmc,
            cfg.mclmc_draws,
            cfg.mclmc_eval_every,
            cfg.mclmc_thin,
            Ln_full64,
            tuner_steps=cfg.mclmc_tune_steps,
            diagonal_preconditioning=cfg.mclmc_diagonal_preconditioning,
            desired_energy_var=cfg.mclmc_desired_energy_var,
            integrator_name=cfg.mclmc_integrator,
            use_tqdm=cfg.use_tqdm,
            progress_update_every=cfg.progress_update_every,
            **diag_targets,
        )

        # Build idata and compute centralized metrics
        idata_mclmc = to_idata(
            Ln_histories=res_mclmc.Ln_histories,
            theta_thin=res_mclmc.theta_thin,
            acceptance=res_mclmc.acceptance,
            energy=res_mclmc.energy,
            n=cfg.n_data,
            beta=beta,
            L0=L0,
        )

        # Core LLC metrics
        m_core = llc_point_se(idata_mclmc)
        print(f"MCLMC LLC: {m_core['llc_mean']:.4f} ± {m_core['llc_se']:.4f} (ESS: {int(m_core['ess_bulk']):.1f})")

        # Efficiency metrics
        m_eff = efficiency_metrics(
            idata=idata_mclmc,
            timings=res_mclmc.timings,
            work=res_mclmc.work,
            n_data=cfg.n_data,
            sgld_batch=None,
        )

        # Store MCLMC results with unified keys
        all_metrics.update({f"mclmc_{k}": v for k, v in m_core.items()})
        all_metrics.update({f"mclmc_{k}": v for k, v in m_eff.items()})
        histories["mclmc"] = res_mclmc.Ln_histories

    print(f"\nTotal Runtime: {toc(t0):.2f}s")

    # Save artifacts if requested
    if save_artifacts and run_dir:
        logger.info("Saving artifacts")

        # Save L0 for running LLC reconstruction
        save_L0(run_dir, L0)

        # Save unified InferenceData files per sampler (new format)
        if "sgld" in getattr(cfg, "samplers", []) and "res_sgld" in locals():
            idata_sgld = to_idata(
                Ln_histories=res_sgld.Ln_histories,
                theta_thin=res_sgld.theta_thin,
                acceptance=res_sgld.acceptance,
                energy=res_sgld.energy,
                n=cfg.n_data,
                beta=beta,
                L0=L0,
            )
            idata_sgld.attrs.update(
                {
                    "n_data": int(cfg.n_data),
                    "beta": float(beta),
                    "L0": float(L0),
                    "sampler": "sgld",
                }
            )
            az.to_netcdf(idata_sgld, f"{run_dir}/sgld.nc")

        if "hmc" in getattr(cfg, "samplers", []) and "res_hmc" in locals():
            idata_hmc = to_idata(
                Ln_histories=res_hmc.Ln_histories,
                theta_thin=res_hmc.theta_thin,
                acceptance=res_hmc.acceptance,
                energy=res_hmc.energy,
                n=cfg.n_data,
                beta=beta,
                L0=L0,
            )
            idata_hmc.attrs.update(
                {
                    "n_data": int(cfg.n_data),
                    "beta": float(beta),
                    "L0": float(L0),
                    "sampler": "hmc",
                }
            )
            az.to_netcdf(idata_hmc, f"{run_dir}/hmc.nc")

        if "mclmc" in getattr(cfg, "samplers", []) and "res_mclmc" in locals():
            idata_mclmc = to_idata(
                Ln_histories=res_mclmc.Ln_histories,
                theta_thin=res_mclmc.theta_thin,
                acceptance=res_mclmc.acceptance,
                energy=res_mclmc.energy,
                n=cfg.n_data,
                beta=beta,
                L0=L0,
            )
            idata_mclmc.attrs.update(
                {
                    "n_data": int(cfg.n_data),
                    "beta": float(beta),
                    "L0": float(L0),
                    "sampler": "mclmc",
                }
            )
            az.to_netcdf(idata_mclmc, f"{run_dir}/mclmc.nc")

        # Save all metrics
        save_metrics(run_dir, all_metrics)

        # Save configuration
        save_config(run_dir, cfg)

        # Generate diagnostic plots if enabled
        print(f"save_plots={cfg.save_plots} diag_mode={cfg.diag_mode}")
        if cfg.diag_mode != "none" and cfg.save_plots:
            logger.info("Generating diagnostic plots...")

            # Use generate_diagnostics with existing idata objects
            if "idata_sgld" in locals():
                generate_diagnostics(idata_sgld, "sgld", run_dir)

            if "idata_hmc" in locals():
                generate_diagnostics(idata_hmc, "hmc", run_dir)

            if "idata_mclmc" in locals():
                generate_diagnostics(idata_mclmc, "mclmc", run_dir)

        # Generate lightweight HTML preview (no manifest)
        gallery_path = generate_gallery_html(run_dir, cfg, all_metrics)
        print(f"HTML preview: {gallery_path}")
        print(f"Artifacts saved to: {run_dir}")

    return RunOutputs(
        run_dir=run_dir,
        metrics=all_metrics,
        histories=histories,
        L0=L0,
    )
