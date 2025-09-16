# llc/pipeline.py
"""
Single source of truth for running one complete experiment.
Replaces duplicated logic between main.py, experiments.py, and tasks.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging
import os

import jax
import jax.numpy as jnp
from jax import random, jit
from jax.flatten_util import ravel_pytree
import numpy as np

from .config import Config
from .cache import run_id, load_cached_outputs

# Set up logger
logger = logging.getLogger(__name__)
from .models import infer_widths, init_mlp_params
from .data import make_dataset
from .losses import as_dtype, make_loss_fns
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
from .samplers.base import prepare_diag_targets
from .diagnostics import llc_mean_and_se_from_histories, plot_diagnostics
from .artifacts import (
    save_L0,
    save_idata_L,
    save_idata_theta,
    save_config,
    save_metrics,
    create_manifest,
    generate_gallery_html,
)
from .experiments import train_erm
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
    if hasattr(cfg, 'artifacts_dir') and cfg.artifacts_dir.endswith('/artifacts'):
        base_dir = cfg.artifacts_dir.replace('/artifacts', '')
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

    logger.info("Building teacher and data")
    stats = RunStats()

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
    logger.info("Training to empirical minimizer...")
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
        sgld_samples_thin, sgld_Es, sgld_Vars, sgld_Ns, Ln_histories_sgld = sgld_runner(
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
            Ln_histories_sgld, cfg.n_data, beta, L0
        )
        print(f"SGLD LLC: {llc_sgld_mean:.4f} ± {se_sgld:.4f} (ESS: {ess_sgld:.1f})")

        # Store SGLD results
        all_metrics.update(
            {
                "sgld_llc_mean": float(llc_sgld_mean),
                "sgld_llc_se": float(se_sgld),
                "sgld_ess": float(ess_sgld),
                "sgld_timing_warmup": float(stats.t_sgld_warmup),
                "sgld_timing_sampling": float(stats.t_sgld_sampling),
                "sgld_n_steps": int(stats.n_sgld_minibatch_grads),
                "sgld_n_full_loss": int(stats.n_sgld_full_loss),
            }
        )
        histories["sgld"] = Ln_histories_sgld

    # ===== HMC (Online) =====
    if "hmc" in getattr(cfg, "samplers", ["sgld"]):
        logger.info("Running HMC (BlackJAX, online)")
        k_hmc = random.fold_in(key, 123)
        init_thetas_hmc = theta0_f64 + 0.01 * random.normal(k_hmc, (cfg.chains, dim))

        # Choose batched or sequential HMC execution
        if cfg.use_batched_chains:
            hmc_samples_thin, hmc_Es, hmc_Vars, hmc_Ns, accs_hmc, Ln_histories_hmc = (
                run_hmc_online_batched(
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
            )
        else:
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
        all_metrics.update(
            {
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
            }
        )
        histories["hmc"] = Ln_histories_hmc

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

            (
                mclmc_samples_thin,
                mclmc_Es,
                mclmc_Vars,
                mclmc_Ns,
                energy_deltas_mclmc,
                Ln_histories_mclmc,
            ) = run_mclmc_online_batched(
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
        print(
            f"MCLMC LLC: {llc_mclmc_mean:.4f} ± {se_mclmc:.4f} (ESS: {ess_mclmc:.1f})"
        )

        # Store MCLMC results
        all_metrics.update(
            {
                "mclmc_llc_mean": float(llc_mclmc_mean),
                "mclmc_llc_se": float(se_mclmc),
                "mclmc_ess": float(ess_mclmc),
                "mclmc_timing_warmup": float(stats.t_mclmc_warmup),
                "mclmc_timing_sampling": float(stats.t_mclmc_sampling),
                "mclmc_n_steps": int(stats.n_mclmc_steps),
                "mclmc_n_full_loss": int(stats.n_mclmc_full_loss),
            }
        )
        histories["mclmc"] = Ln_histories_mclmc

    print(f"\nTotal Runtime: {toc(t0):.2f}s")

    # Save artifacts if requested
    if save_artifacts and run_dir:
        logger.info("Saving artifacts")

        # Save L0 for running LLC reconstruction
        save_L0(run_dir, L0)

        # Save L_n histories as ArviZ InferenceData (NetCDF)
        if "sgld" in getattr(cfg, "samplers", []):
            save_idata_L(run_dir, "sgld", histories.get("sgld", []))
        if "hmc" in getattr(cfg, "samplers", []):
            save_idata_L(run_dir, "hmc", histories.get("hmc", []))
        if "mclmc" in getattr(cfg, "samplers", []):
            save_idata_L(run_dir, "mclmc", histories.get("mclmc", []))

        # Save thinned theta samples as ArviZ InferenceData
        if "sgld" in getattr(cfg, "samplers", []):
            save_idata_theta(run_dir, "sgld", locals().get("sgld_samples_thin", []))
        if "hmc" in getattr(cfg, "samplers", []):
            save_idata_theta(run_dir, "hmc", locals().get("hmc_samples_thin", []))
        if "mclmc" in getattr(cfg, "samplers", []):
            save_idata_theta(run_dir, "mclmc", locals().get("mclmc_samples_thin", []))

        # Save all metrics
        save_metrics(run_dir, all_metrics)

        # Save configuration
        save_config(run_dir, cfg)

        # Generate diagnostic plots if enabled
        if cfg.diag_mode != "none" and cfg.save_plots:
            logger.info("Generating diagnostic plots...")

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
