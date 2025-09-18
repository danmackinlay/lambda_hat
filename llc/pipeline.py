# llc/pipeline.py
"""
Single source of truth for running one complete experiment.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging
import os

import jax.numpy as jnp
from jax import random, jit
import numpy as np

from .config import Config
from .cache import run_id, load_cached_outputs

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
    run_sgld_online,
    run_hmc_online_with_adaptation,
    run_mclmc_online,
    run_sgld_online_batched,
    run_hmc_online_batched,
)
from .convert import to_idata
import arviz as az
from .samplers.base import prepare_diag_targets
from llc.analysis import llc_point_se_from_histories as llc_mean_and_se_from_histories
from llc.analysis import generate_diagnostics
from .artifacts import (
    save_L0,
    save_config,
    save_metrics,
    create_manifest,
    generate_gallery_html,
)
from datetime import datetime


@dataclass
class RunOutputs:
    """Results from running one complete experiment"""

    run_dir: Optional[str]
    metrics: Dict[str, Any]
    histories: Dict[str, Any]  # {"sgld": [...], "hmc": [...], "mclmc": [...]}
    L0: float


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

        # Choose batched or sequential SGLD execution
        sgld_runner = (
            run_sgld_online_batched if cfg.use_batched_chains else run_sgld_online
        )
        res_sgld = sgld_runner(
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
            stats=stats,
            **diag_targets,
            # NEW: optional preconditioning (works even if cfg lacks these fields)
            precond_mode=getattr(cfg, "sgld_precond", "none"),
            beta1=getattr(cfg, "sgld_beta1", 0.9),
            beta2=getattr(cfg, "sgld_beta2", 0.999),
            eps=getattr(cfg, "sgld_eps", 1e-8),
            bias_correction=getattr(cfg, "sgld_bias_correction", True),
        )

        # Compute LLC with ESS-based uncertainty
        llc_sgld_mean, se_sgld, ess_sgld = llc_mean_and_se_from_histories(
            res_sgld.Ln_histories, cfg.n_data, beta, L0
        )
        print(f"SGLD LLC: {llc_sgld_mean:.4f} ± {se_sgld:.4f} (ESS: {ess_sgld:.1f})")

        # Efficiency (wall-clock and gradient-normalized) + Work-Normalized Variance
        # Full-data-equivalent gradients (FDE): minibatch grads scaled by b/n
        b = getattr(cfg, "sgld_batch_size", 1)
        n = cfg.n_data
        fde_sgld = float(stats.n_sgld_full_loss) + float(
            stats.n_sgld_minibatch_grads
        ) * (b / n)
        ess_per_sec_sgld = float(ess_sgld) / max(1e-9, stats.t_sgld_sampling)
        ess_per_fde_sgld = float(ess_sgld) / max(1e-12, fde_sgld)
        # Back out MC std s from se = s / sqrt(ess)
        s_hat_sgld = (
            float(se_sgld) * np.sqrt(max(1.0, ess_sgld))
            if np.isfinite(se_sgld)
            else np.nan
        )
        # True WNV: Var(estimator) × cost = (s²/ESS) × cost
        wnv_time_sgld = (
            (s_hat_sgld**2 / ess_sgld) * stats.t_sgld_sampling
            if np.isfinite(s_hat_sgld) and ess_sgld > 0
            else np.nan
        )
        wnv_fde_sgld = (
            (s_hat_sgld**2 / ess_sgld) * fde_sgld
            if np.isfinite(s_hat_sgld) and ess_sgld > 0
            else np.nan
        )
        ess_target = (
            (s_hat_sgld / target_se) ** 2
            if np.isfinite(s_hat_sgld) and s_hat_sgld > 0
            else np.nan
        )
        time_to_target_sgld = (
            stats.t_sgld_sampling * (ess_target / ess_sgld)
            if np.isfinite(ess_target) and ess_sgld > 0
            else np.nan
        )
        fde_to_target_sgld = (
            fde_sgld * (ess_target / ess_sgld)
            if np.isfinite(ess_target) and ess_sgld > 0
            else np.nan
        )

        # Store SGLD results
        all_metrics.update(
            {
                "sgld_llc_mean": float(llc_sgld_mean),
                "sgld_llc_se": float(se_sgld),
                "sgld_ess": float(ess_sgld),
                "sgld_ess_per_sec": ess_per_sec_sgld,
                "sgld_ess_per_fde": ess_per_fde_sgld,
                "sgld_wnv_time": float(wnv_time_sgld),
                "sgld_wnv_fde": float(wnv_fde_sgld),
                "sgld_time_to_se1": float(time_to_target_sgld),
                "sgld_fde_to_se1": float(fde_to_target_sgld),
                "sgld_timing_warmup": float(stats.t_sgld_warmup),
                "sgld_timing_sampling": float(stats.t_sgld_sampling),
                "sgld_n_steps": int(stats.n_sgld_minibatch_grads),
                "sgld_n_full_loss": int(stats.n_sgld_full_loss),
            }
        )
        histories["sgld"] = res_sgld.Ln_histories

    # ===== HMC (Online) =====
    if "hmc" in getattr(cfg, "samplers", ["sgld"]):
        logger.info("Running HMC (BlackJAX, online)")
        k_hmc = random.fold_in(key, 123)
        init_thetas_hmc = theta0_f64 + 0.01 * random.normal(k_hmc, (cfg.chains, dim))

        # Choose batched or sequential HMC execution
        if cfg.use_batched_chains:
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
        else:
            res_hmc = run_hmc_online_with_adaptation(
                k_hmc,
                init_thetas_hmc,
                logpost_and_grad_f64,
                cfg.hmc_draws,
                cfg.hmc_warmup,
                cfg.hmc_num_integration_steps,
                cfg.hmc_eval_every,
                cfg.hmc_thin,
                Ln_full64,
                use_tqdm=cfg.use_tqdm,
                progress_update_every=cfg.progress_update_every,
                stats=stats,
                **diag_targets,
            )

        # Compute LLC with ESS-based uncertainty
        llc_hmc_mean, se_hmc, ess_hmc = llc_mean_and_se_from_histories(
            res_hmc.Ln_histories, cfg.n_data, beta, L0
        )
        vals = (
            [np.nanmean(a) for a in res_hmc.acceptance if a.size]
            if res_hmc.acceptance
            else []
        )
        mean_acc = float(np.nanmean(vals)) if vals else float("nan")

        print(f"HMC LLC: {llc_hmc_mean:.4f} ± {se_hmc:.4f} (ESS: {ess_hmc:.1f})")
        print(f"HMC acceptance rate (mean over chains/draws): {mean_acc:.3f}")

        # Efficiency
        fde_hmc = float(stats.n_hmc_full_loss) + float(
            stats.n_hmc_leapfrog_grads
        )  # leapfrog grads are full
        ess_per_sec_hmc = float(ess_hmc) / max(1e-9, stats.t_hmc_sampling)
        ess_per_fde_hmc = float(ess_hmc) / max(1e-12, fde_hmc)
        s_hat_hmc = (
            float(se_hmc) * np.sqrt(max(1.0, ess_hmc))
            if np.isfinite(se_hmc)
            else np.nan
        )
        # True WNV: Var(estimator) × cost = (s²/ESS) × cost
        wnv_time_hmc = (
            (s_hat_hmc**2 / ess_hmc) * stats.t_hmc_sampling
            if np.isfinite(s_hat_hmc) and ess_hmc > 0
            else np.nan
        )
        wnv_fde_hmc = (
            (s_hat_hmc**2 / ess_hmc) * fde_hmc
            if np.isfinite(s_hat_hmc) and ess_hmc > 0
            else np.nan
        )
        ess_target = (
            (s_hat_hmc / target_se) ** 2
            if np.isfinite(s_hat_hmc) and s_hat_hmc > 0
            else np.nan
        )
        time_to_target_hmc = (
            stats.t_hmc_sampling * (ess_target / ess_hmc)
            if np.isfinite(ess_target) and ess_hmc > 0
            else np.nan
        )
        fde_to_target_hmc = (
            fde_hmc * (ess_target / ess_hmc)
            if np.isfinite(ess_target) and ess_hmc > 0
            else np.nan
        )

        # Store HMC results
        all_metrics.update(
            {
                "hmc_llc_mean": float(llc_hmc_mean),
                "hmc_llc_se": float(se_hmc),
                "hmc_ess": float(ess_hmc),
                "hmc_ess_per_sec": ess_per_sec_hmc,
                "hmc_ess_per_fde": ess_per_fde_hmc,
                "hmc_wnv_time": float(wnv_time_hmc),
                "hmc_wnv_fde": float(wnv_fde_hmc),
                "hmc_time_to_se1": float(time_to_target_hmc),
                "hmc_fde_to_se1": float(fde_to_target_hmc),
                "hmc_timing_warmup": float(stats.t_hmc_warmup),
                "hmc_timing_sampling": float(stats.t_hmc_sampling),
                "hmc_n_leapfrog_grads": int(stats.n_hmc_leapfrog_grads),
                "hmc_n_full_loss": int(stats.n_hmc_full_loss),
                "hmc_mean_acceptance": float(
                    np.nanmean([np.nanmean(a) for a in res_hmc.acceptance if a.size])
                )
                if res_hmc.acceptance and any(a.size for a in res_hmc.acceptance)
                else float("nan"),
            }
        )
        histories["hmc"] = res_hmc.Ln_histories

    # ===== MCLMC (Online) =====
    if "mclmc" in getattr(cfg, "samplers", ["sgld"]):
        logger.info("Running MCLMC (BlackJAX, online)")
        k_mclmc = random.fold_in(key, 456)
        init_thetas_mclmc = theta0_f64 + 0.01 * random.normal(
            k_mclmc, (cfg.chains, dim)
        )

        # Create logdensity for MCLMC
        logdensity_mclmc = make_logdensity_for_mclmc(
            loss_full_f64, theta0_f64, cfg.n_data, beta, gamma
        )

        if cfg.use_batched_chains:
            from llc.runners import run_mclmc_online_batched

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
                stats=stats,
                **diag_targets,
            )
        else:
            res_mclmc = run_mclmc_online(
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
                stats=stats,
                **diag_targets,
            )

        # Compute LLC with ESS-based uncertainty
        llc_mclmc_mean, se_mclmc, ess_mclmc = llc_mean_and_se_from_histories(
            res_mclmc.Ln_histories, cfg.n_data, beta, L0
        )
        print(
            f"MCLMC LLC: {llc_mclmc_mean:.4f} ± {se_mclmc:.4f} (ESS: {ess_mclmc:.1f})"
        )

        # Efficiency (adapt to your bookkeeping)
        fde_mclmc = (
            float(stats.n_mclmc_full_loss)
            if hasattr(stats, "n_mclmc_full_loss")
            else 0.0
        )
        ess_per_sec_mclmc = float(ess_mclmc) / max(1e-9, stats.t_mclmc_sampling)
        ess_per_fde_mclmc = (
            float(ess_mclmc) / max(1e-12, fde_mclmc) if fde_mclmc > 0 else np.nan
        )
        s_hat_mclmc = (
            float(se_mclmc) * np.sqrt(max(1.0, ess_mclmc))
            if np.isfinite(se_mclmc)
            else np.nan
        )
        ess_target = (
            (s_hat_mclmc / target_se) ** 2
            if np.isfinite(s_hat_mclmc) and s_hat_mclmc > 0
            else np.nan
        )
        time_to_target_mclmc = (
            stats.t_mclmc_sampling * (ess_target / ess_mclmc)
            if np.isfinite(ess_target) and ess_mclmc > 0
            else np.nan
        )
        fde_to_target_mclmc = (
            fde_mclmc * (ess_target / ess_mclmc)
            if np.isfinite(ess_target) and ess_mclmc > 0
            else np.nan
        )

        # True WNV: Var(estimator) × cost = (s²/ESS) × cost
        wnv_time_mclmc = (
            (s_hat_mclmc**2 / ess_mclmc) * stats.t_mclmc_sampling
            if np.isfinite(s_hat_mclmc) and ess_mclmc > 0
            else np.nan
        )
        wnv_fde_mclmc = (
            (s_hat_mclmc**2 / ess_mclmc) * fde_mclmc
            if np.isfinite(s_hat_mclmc) and ess_mclmc > 0
            else np.nan
        )

        # Store MCLMC results
        all_metrics.update(
            {
                "mclmc_llc_mean": float(llc_mclmc_mean),
                "mclmc_llc_se": float(se_mclmc),
                "mclmc_ess": float(ess_mclmc),
                "mclmc_ess_per_sec": ess_per_sec_mclmc,
                "mclmc_ess_per_fde": ess_per_fde_mclmc,
                "mclmc_time_to_se1": float(time_to_target_mclmc),
                "mclmc_fde_to_se1": float(fde_to_target_mclmc),
                "mclmc_timing_warmup": float(stats.t_mclmc_warmup),
                "mclmc_timing_sampling": float(stats.t_mclmc_sampling),
                "mclmc_n_steps": int(stats.n_mclmc_steps),
                "mclmc_n_full_loss": int(stats.n_mclmc_full_loss),
                "mclmc_wnv_time": wnv_time_mclmc,
                "mclmc_wnv_fde": wnv_fde_mclmc,
            }
        )
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

        # Create manifest
        from pathlib import Path

        pngs = [p.name for p in Path(run_dir).glob("*.png")]
        artifact_files = [
            "config.json",
            "metrics.json",
            "L0.txt",
            "sgld_L.nc",
            "hmc_L.nc",
            "mclmc_L.nc",
            "sgld_theta.nc",
            "hmc_theta.nc",
            "mclmc_theta.nc",
            *pngs,
        ]
        create_manifest(run_dir, cfg, all_metrics, artifact_files)

        # Generate HTML gallery
        gallery_path = generate_gallery_html(run_dir, cfg, all_metrics)
        print(f"HTML gallery: {gallery_path}")

        # Create convenience timestamp symlink in artifacts/ for completed runs
        try:
            from llc.manifest import create_timestamp_symlink

            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            # Determine artifacts directory from run_dir
            if "/runs/" in run_dir:
                artifacts_dir = Path(run_dir).parent.parent / "artifacts"
            else:
                artifacts_dir = Path(cfg.artifacts_dir)
            artifacts_dir.mkdir(exist_ok=True)
            create_timestamp_symlink(artifacts_dir, ts, rid)
            print(f"Created timestamp symlink: {ts} -> runs/{rid}")
        except Exception as e:
            logger.warning(f"Could not create timestamp symlink: {e}")

        print(f"Artifacts saved to: {run_dir}")

    return RunOutputs(
        run_dir=run_dir,
        metrics=all_metrics,
        histories=histories,
        L0=L0,
    )
