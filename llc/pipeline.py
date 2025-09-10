# llc/pipeline.py
"""
Single source of truth for running one complete experiment.
Replaces duplicated logic between main.py, experiments.py, and tasks.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import os

import jax
import jax.numpy as jnp
from jax import random, jit
from jax.flatten_util import ravel_pytree
import numpy as np

from .config import Config
from .models import infer_widths, init_mlp_params
from .data import make_dataset
from .losses import as_dtype, make_loss_fns
from .posterior import compute_beta_gamma, make_logpost_and_score, make_logdensity_for_mclmc
from .runners import (
    RunStats,
    tic,
    toc,
    run_sgld_online,
    run_hmc_online_with_adaptation,
    run_mclmc_online,
)
from .samplers.base import prepare_diag_targets
from .diagnostics import llc_mean_and_se_from_histories, plot_diagnostics
from .artifacts import (
    create_run_directory,
    save_L0,
    save_idata_L,
    save_idata_theta,
    save_config,
    save_metrics,
    create_manifest,
    generate_gallery_html,
)
from .experiments import train_erm


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
        skip_if_exists: Whether to skip if results already exist (future: PR2)
        
    Returns:
        RunOutputs with metrics, histories, run_dir, and L0
    """
    print("=== Building teacher and data ===")
    stats = RunStats()

    # Create run directory for artifacts (for now, use original logic)
    run_dir = create_run_directory(cfg) if cfg.auto_create_run_dir else ""
    if run_dir:
        print(f"Artifacts will be saved to: {run_dir}")

    # Build timing
    t0 = tic()
    key = random.PRNGKey(cfg.seed)

    X, Y, teacher_params, teacher_forward = make_dataset(key, cfg)

    # Initialize student network parameters
    key, subkey = random.split(key)
    widths = cfg.widths or infer_widths(
        cfg.in_dim, cfg.out_dim, cfg.depth, cfg.target_params, fallback_width=cfg.hidden
    )
    w0_pytree = init_mlp_params(
        subkey, cfg.in_dim, widths, cfg.out_dim, cfg.activation, cfg.bias, cfg.init
    )

    stats.t_build = toc(t0)

    # Train to empirical minimizer (ERM) - center the local prior there
    print("Training to empirical minimizer...")
    t1 = tic()
    theta_star_f64, unravel_star_f64 = train_erm(
        w0_pytree, cfg, X.astype(jnp.float64), Y.astype(jnp.float64)
    )
    stats.t_train = toc(t1)

    # Create proper f32 unravel function (rebuild around f32 params)
    params_star_f64 = unravel_star_f64(theta_star_f64)
    params_star_f32 = jax.tree_util.tree_map(
        lambda a: a.astype(jnp.float32), params_star_f64
    )
    theta_star_f32, unravel_star_f32 = ravel_pytree(params_star_f32)

    # Center the local prior at θ⋆, not at the teacher
    theta0_f64, unravel_f64 = theta_star_f64, unravel_star_f64
    theta0_f32, unravel_f32 = theta_star_f32, unravel_star_f32

    # Create dtype-specific data versions
    X_f32, Y_f32 = as_dtype(X, cfg.sgld_dtype), as_dtype(Y, cfg.sgld_dtype)
    X_f64, Y_f64 = as_dtype(X, cfg.hmc_dtype), as_dtype(Y, cfg.hmc_dtype)

    dim = theta0_f32.size
    print(f"Parameter dimension: {dim:,d}")

    beta, gamma = compute_beta_gamma(cfg, dim)
    print(f"beta={beta:.6g} gamma={gamma:.6g}")

    # Create loss functions for each dtype
    loss_full_f32, loss_minibatch_f32 = make_loss_fns(unravel_f32, cfg, X_f32, Y_f32)
    loss_full_f64, loss_minibatch_f64 = make_loss_fns(unravel_f64, cfg, X_f64, Y_f64)

    # log posterior & gradient factories for each dtype
    logpost_and_grad_f32, grad_logpost_minibatch_f32 = make_logpost_and_score(
        loss_full_f32, loss_minibatch_f32, theta0_f32, cfg.n_data, beta, gamma
    )
    logpost_and_grad_f64, grad_logpost_minibatch_f64 = make_logpost_and_score(
        loss_full_f64, loss_minibatch_f64, theta0_f64, cfg.n_data, beta, gamma
    )

    # Recompute L0 at empirical minimizer (do this in float64 for both samplers)
    L0 = float(loss_full_f64(theta0_f64))
    print(f"L0 at empirical minimizer: {L0:.6f}")

    # JIT compile the loss evaluator for LLC computation
    Ln_full64 = jit(loss_full_f64)

    # Prepare diagnostic targets based on config
    diag_targets = prepare_diag_targets(dim, cfg)

    # Storage for results
    all_metrics = {}
    histories = {}
    
    # ===== SGLD (Online) =====
    print("\n=== SGLD (BlackJAX, online) ===")
    k_sgld = random.split(key, 1)[0]
    # simple overdispersed inits around w0
    init_thetas_sgld = theta0_f32 + 0.01 * random.normal(
        k_sgld, (cfg.chains, dim)
    ).astype(jnp.float32)

    sgld_samples_thin, sgld_Es, sgld_Vars, sgld_Ns, Ln_histories_sgld = run_sgld_online(
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
    )

    # Compute LLC with ESS-based uncertainty
    llc_sgld_mean, se_sgld, ess_sgld = llc_mean_and_se_from_histories(
        Ln_histories_sgld, cfg.n_data, beta, L0
    )
    print(f"SGLD LLC: {llc_sgld_mean:.4f} ± {se_sgld:.4f} (ESS: {ess_sgld:.1f})")

    # Store SGLD results
    all_metrics.update({
        "sgld_llc_mean": float(llc_sgld_mean),
        "sgld_llc_se": float(se_sgld),
        "sgld_ess": float(ess_sgld),
        "sgld_timing_warmup": float(stats.t_sgld_warmup),
        "sgld_timing_sampling": float(stats.t_sgld_sampling),
        "sgld_n_steps": int(stats.n_sgld_minibatch_grads),
        "sgld_n_full_loss": int(stats.n_sgld_full_loss),
    })
    histories["sgld"] = Ln_histories_sgld

    # ===== HMC (Online) =====
    print("\n=== HMC (BlackJAX, online) ===")
    k_hmc = random.fold_in(key, 123)
    init_thetas_hmc = theta0_f64 + 0.01 * random.normal(k_hmc, (cfg.chains, dim))

    hmc_samples_thin, hmc_Es, hmc_Vars, hmc_Ns, accs_hmc, Ln_histories_hmc = (
        run_hmc_online_with_adaptation(
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
    )

    # Compute LLC with ESS-based uncertainty
    llc_hmc_mean, se_hmc, ess_hmc = llc_mean_and_se_from_histories(
        Ln_histories_hmc, cfg.n_data, beta, L0
    )
    vals = [np.nanmean(a) for a in accs_hmc if a.size]
    mean_acc = float(np.nanmean(vals)) if vals else float("nan")

    print(f"HMC LLC: {llc_hmc_mean:.4f} ± {se_hmc:.4f} (ESS: {ess_hmc:.1f})")
    print(f"HMC acceptance rate (mean over chains/draws): {mean_acc:.3f}")

    # Store HMC results
    all_metrics.update({
        "hmc_llc_mean": float(llc_hmc_mean),
        "hmc_llc_se": float(se_hmc),
        "hmc_ess": float(ess_hmc),
        "hmc_timing_warmup": float(stats.t_hmc_warmup),
        "hmc_timing_sampling": float(stats.t_hmc_sampling),
        "hmc_n_leapfrog_grads": int(stats.n_hmc_leapfrog_grads),
        "hmc_n_full_loss": int(stats.n_hmc_full_loss),
        "hmc_mean_acceptance": float(
            np.nanmean([np.nanmean(a) for a in accs_hmc if a.size])
        )
        if any(a.size for a in accs_hmc)
        else float("nan"),
    })
    histories["hmc"] = Ln_histories_hmc

    # ===== MCLMC (Online) =====
    print("\n=== MCLMC (BlackJAX, online) ===")
    k_mclmc = random.fold_in(key, 456)
    init_thetas_mclmc = theta0_f64 + 0.01 * random.normal(k_mclmc, (cfg.chains, dim))

    # Create logdensity for MCLMC
    logdensity_mclmc = make_logdensity_for_mclmc(
        loss_full_f64, theta0_f64, cfg.n_data, beta, gamma
    )

    (
        mclmc_samples_thin,
        mclmc_Es,
        mclmc_Vars,
        mclmc_Ns,
        energy_deltas_mclmc,
        Ln_histories_mclmc,
    ) = run_mclmc_online(
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
        Ln_histories_mclmc, cfg.n_data, beta, L0
    )
    print(f"MCLMC LLC: {llc_mclmc_mean:.4f} ± {se_mclmc:.4f} (ESS: {ess_mclmc:.1f})")

    # Store MCLMC results
    all_metrics.update({
        "mclmc_llc_mean": float(llc_mclmc_mean),
        "mclmc_llc_se": float(se_mclmc),
        "mclmc_ess": float(ess_mclmc),
        "mclmc_timing_warmup": float(stats.t_mclmc_warmup),
        "mclmc_timing_sampling": float(stats.t_mclmc_sampling),
        "mclmc_n_steps": int(stats.n_mclmc_steps),
        "mclmc_n_full_loss": int(stats.n_mclmc_full_loss),
    })
    histories["mclmc"] = Ln_histories_mclmc

    print(f"\nTotal Runtime: {toc(t0):.2f}s")

    # Save artifacts if requested
    if save_artifacts and run_dir:
        print("\n=== Saving Artifacts ===")
        
        # Save L0 for running LLC reconstruction
        save_L0(run_dir, L0)

        # Save L_n histories as ArviZ InferenceData (NetCDF)
        save_idata_L(run_dir, "sgld", Ln_histories_sgld)
        save_idata_L(run_dir, "hmc", Ln_histories_hmc)
        save_idata_L(run_dir, "mclmc", Ln_histories_mclmc)

        # Save thinned theta samples as ArviZ InferenceData
        save_idata_theta(run_dir, "sgld", sgld_samples_thin)
        save_idata_theta(run_dir, "hmc", hmc_samples_thin)
        save_idata_theta(run_dir, "mclmc", mclmc_samples_thin)

        # Save all metrics
        save_metrics(run_dir, all_metrics)

        # Save configuration
        save_config(run_dir, cfg)

        # Generate diagnostic plots if enabled
        if cfg.diag_mode != "none" and cfg.save_plots:
            print("Generating diagnostic plots...")

            # Call single-sampler plot_diagnostics for each sampler
            if sgld_samples_thin.size > 0:
                plot_diagnostics(
                    run_dir=run_dir,
                    sampler_name="sgld",
                    Ln_histories=Ln_histories_sgld,
                    samples_thin=sgld_samples_thin,
                    n=cfg.n_data,
                    beta=beta,
                    L0=L0,
                    save_plots=cfg.save_plots,
                )

            if hmc_samples_thin.size > 0:
                plot_diagnostics(
                    run_dir=run_dir,
                    sampler_name="hmc",
                    Ln_histories=Ln_histories_hmc,
                    samples_thin=hmc_samples_thin,
                    acceptance_rates=accs_hmc,
                    n=cfg.n_data,
                    beta=beta,
                    L0=L0,
                    save_plots=cfg.save_plots,
                )

            if mclmc_samples_thin.size > 0:
                plot_diagnostics(
                    run_dir=run_dir,
                    sampler_name="mclmc",
                    Ln_histories=Ln_histories_mclmc,
                    samples_thin=mclmc_samples_thin,
                    energy_deltas=energy_deltas_mclmc,
                    n=cfg.n_data,
                    beta=beta,
                    L0=L0,
                    save_plots=cfg.save_plots,
                )

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

        print(f"Artifacts saved to: {run_dir}")

    return RunOutputs(
        run_dir=run_dir,
        metrics=all_metrics,
        histories=histories,
        L0=L0,
    )