# main.py
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")  # HMC benefits from float64
os.environ.setdefault("MPLBACKEND", "Agg")  # Headless rendering - no GUI windows

import argparse
from dataclasses import replace

import arviz as az
import jax
import jax.numpy as jnp
from jax import random, jit
from jax.flatten_util import ravel_pytree
import scipy.stats as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import new sampler adapters and utility modules
from llc.samplers.base import prepare_diag_targets
from llc.diagnostics import (
    llc_mean_and_se_from_histories,
    llc_ci_from_histories,
    plot_diagnostics,
)
from llc.artifacts import (
    create_run_directory,
    save_config,
    save_idata_L,
    save_idata_theta,
    save_metrics,
    create_manifest,
    generate_gallery_html,
    save_L0,
)
from llc.models import infer_widths, init_mlp_params
from llc.data import make_dataset
from llc.losses import as_dtype, make_loss_fns
from llc.posterior import (
    compute_beta_gamma,
    make_logpost_and_score,
    make_logdensity_for_mclmc,
)
from llc.runners import (
    RunStats,
    tic,
    toc,
    run_sgld_online,
    run_hmc_online_with_adaptation,
    run_mclmc_online,
)
from llc.config import Config, CFG
from llc.experiments import train_erm, sweep_space, build_sweep_worklist


plt.switch_backend("Agg")  # Ensure headless backend even if pyplot was already imported


def get_accept(info):
    """Robust accessor for HMC acceptance rate across BlackJAX versions

    BlackJAX >=1.2: HMCInfo.acceptance_rate (float)
    Some versions expose nested 'acceptance.rate'
    This function handles both patterns.
    """
    if hasattr(info, "acceptance_rate"):
        return float(info.acceptance_rate)
    acc = getattr(info, "acceptance", None)
    return float(getattr(acc, "rate", np.nan)) if acc is not None else np.nan


def work_normalized_variance(se, time_seconds: float, grad_work: int):
    """Compute WNV in both time and gradient units"""
    return dict(
        WNV_seconds=float(se**2 * max(1e-12, time_seconds)),
        WNV_grads=float(se**2 * max(1.0, grad_work)),
    )


def scalar_chain_diagnostics(series_per_chain, name="L"):
    """Compute ESS and R-hat for a scalar quantity across chains"""
    # Truncate to common length
    valid = [np.asarray(s) for s in series_per_chain if len(s) > 1]
    if not valid:
        return dict(ess=np.nan, rhat=np.nan)
    m = min(len(s) for s in valid)
    H = np.stack([s[:m] for s in valid], axis=0)  # (chains, m)
    idata = az.from_dict(posterior={name: H})
    ess_result = az.ess(idata, var_names=[name])
    rhat_result = az.rhat(idata, var_names=[name])
    ess = float(np.nanmedian(ess_result[name].values))
    rhat = float(np.nanmax(rhat_result[name].values))
    return dict(ess=ess, rhat=rhat)


# ----------------------------
# CLI Argument Parsing
# ----------------------------
def parse_args():
    """Parse command line arguments to override Config defaults"""
    parser = argparse.ArgumentParser(description="Local Learning Coefficient Analysis")

    # Sampler selection
    parser.add_argument(
        "--samplers",
        type=str,
        default=None,
        help="Comma-separated list of samplers (sgld,hmc,mclmc)",
    )

    # Data parameters
    parser.add_argument(
        "--n-data", type=int, default=None, help="Number of data points"
    )

    # Sampling parameters
    parser.add_argument(
        "--chains", type=int, default=None, help="Number of chains to run"
    )

    # SGLD parameters
    parser.add_argument(
        "--sgld-steps", type=int, default=None, help="Number of SGLD steps"
    )
    parser.add_argument(
        "--sgld-warmup", type=int, default=None, help="SGLD warmup steps"
    )
    parser.add_argument(
        "--sgld-step-size", type=float, default=None, help="SGLD step size"
    )

    # HMC parameters
    parser.add_argument(
        "--hmc-draws", type=int, default=None, help="Number of HMC draws"
    )
    parser.add_argument("--hmc-warmup", type=int, default=None, help="HMC warmup steps")
    parser.add_argument(
        "--hmc-steps", type=int, default=None, help="HMC integration steps"
    )

    # MCLMC parameters
    parser.add_argument(
        "--mclmc-draws", type=int, default=None, help="Number of MCLMC draws"
    )

    # Output control
    parser.add_argument(
        "--save-plots", action="store_true", default=None, help="Save diagnostic plots"
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        default=None,
        help="Don't save diagnostic plots",
    )

    # Presets
    parser.add_argument(
        "--preset",
        choices=["quick", "full"],
        default=None,
        help="Use quick or full preset",
    )

    # Model parameters
    parser.add_argument(
        "--target-params",
        type=int,
        default=None,
        help="Target parameter count for model",
    )

    return parser.parse_args()


def apply_preset(cfg: Config, preset: str) -> Config:
    """Apply quick or full preset configurations"""
    if preset == "quick":
        # Quick preset: fewer steps, larger eval_every, more thinning
        return replace(
            cfg,
            sgld_steps=1000,
            sgld_warmup=200,
            sgld_eval_every=20,
            sgld_thin=10,
            hmc_draws=200,
            hmc_warmup=200,
            hmc_eval_every=5,
            hmc_thin=5,
            mclmc_draws=500,
            mclmc_eval_every=10,
            mclmc_thin=5,
            progress_update_every=100,
        )
    elif preset == "full":
        # Full preset: current defaults (no changes)
        return cfg
    else:
        return cfg


def override_config(cfg: Config, args) -> Config:
    """Override config with command line arguments"""
    overrides = {}

    # Handle samplers list
    if args.samplers:
        samplers = [s.strip() for s in args.samplers.split(",")]
        # For now, just set the primary sampler to the first one
        if samplers:
            overrides["sampler"] = samplers[0]

    # Simple parameter overrides
    if args.n_data is not None:
        overrides["n_data"] = args.n_data
    if args.chains is not None:
        overrides["chains"] = args.chains
    if args.sgld_steps is not None:
        overrides["sgld_steps"] = args.sgld_steps
    if args.sgld_warmup is not None:
        overrides["sgld_warmup"] = args.sgld_warmup
    if args.sgld_step_size is not None:
        overrides["sgld_step_size"] = args.sgld_step_size
    if args.hmc_draws is not None:
        overrides["hmc_draws"] = args.hmc_draws
    if args.hmc_warmup is not None:
        overrides["hmc_warmup"] = args.hmc_warmup
    if args.hmc_steps is not None:
        overrides["hmc_num_integration_steps"] = args.hmc_steps
    if args.mclmc_draws is not None:
        overrides["mclmc_draws"] = args.mclmc_draws
    if args.target_params is not None:
        overrides["target_params"] = args.target_params

    # Handle save plots
    if args.save_plots:
        overrides["save_plots"] = True
    elif args.no_save_plots:
        overrides["save_plots"] = False

    return replace(cfg, **overrides)


# ----------------------------
# Main
# ----------------------------
def main(cfg: Config = CFG):
    print("=== Building teacher and data ===")
    stats = RunStats()

    # Create run directory for artifacts
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

    # Compute LLC with proper CI using ESS
    llc_sgld, ci_sgld = llc_ci_from_histories(Ln_histories_sgld, cfg.n_data, beta, L0)
    print(f"SGLD LLC: {llc_sgld:.4f}  95% CI: [{ci_sgld[0]:.4f}, {ci_sgld[1]:.4f}]")

    # Scalar diagnostics on L_n histories (relevant to LLC estimand)
    diag_L_sgld = scalar_chain_diagnostics(Ln_histories_sgld, name="L")
    print("SGLD diagnostics (L_n histories):")
    print(
        f"  ESS(L_n): {diag_L_sgld['ess']:.1f}  R-hat(L_n): {diag_L_sgld['rhat']:.3f}"
    )

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

    # Compute LLC with proper CI using ESS
    llc_hmc, ci_hmc = llc_ci_from_histories(Ln_histories_hmc, cfg.n_data, beta, L0)
    vals = [np.nanmean(a) for a in accs_hmc if a.size]
    mean_acc = float(np.nanmean(vals)) if vals else float("nan")

    print(f"HMC LLC: {llc_hmc:.4f}  95% CI: [{ci_hmc[0]:.4f}, {ci_hmc[1]:.4f}]")
    print(f"HMC acceptance rate (mean over chains/draws): {mean_acc:.3f}")

    # Scalar diagnostics on L_n histories (relevant to LLC estimand)
    diag_L_hmc = scalar_chain_diagnostics(Ln_histories_hmc, name="L")
    print("HMC diagnostics (L_n histories):")
    print(f"  ESS(L_n): {diag_L_hmc['ess']:.1f}  R-hat(L_n): {diag_L_hmc['rhat']:.3f}")

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

    # Compute LLC with proper CI using ESS
    llc_mclmc, ci_mclmc = llc_ci_from_histories(
        Ln_histories_mclmc, cfg.n_data, beta, L0
    )
    print(f"MCLMC LLC: {llc_mclmc:.4f}  95% CI: [{ci_mclmc[0]:.4f}, {ci_mclmc[1]:.4f}]")

    # Scalar diagnostics on L_n histories (relevant to LLC estimand)
    diag_L_mclmc = scalar_chain_diagnostics(Ln_histories_mclmc, name="L")
    print("MCLMC diagnostics (L_n histories):")
    print(
        f"  ESS(L_n): {diag_L_mclmc['ess']:.1f}  R-hat(L_n): {diag_L_mclmc['rhat']:.3f}"
    )

    # LLC confidence interval from running statistics
    def llc_ci_from_running(L_means, L_vars, L_ns, n, beta, L0, alpha=0.05):
        # combine chain means via simple average; use within-chain SE pooled / n_chains
        llc_chain = n * beta * (L_means - L0)
        se_chain = n * beta * np.sqrt(L_vars / np.maximum(1, (L_ns - 1)))
        # conservative: combine via mean of variances / sqrt(C)
        se = float(np.sqrt(np.nanmean(se_chain**2) / max(1, len(se_chain))))
        z = st.norm.ppf(1 - alpha / 2)
        return float(np.nanmean(llc_chain)), (
            llc_chain.mean() - z * se,
            llc_chain.mean() + z * se,
        )

    # ==============================================
    # Work-Normalized Variance (WNV) Analysis
    # ==============================================
    print("\n=== Work-Normalized Variance (WNV) Analysis ===")

    # Compute LLC estimates with standard errors from histories
    llc_sgld_mean, se_sgld, ess_sgld = llc_mean_and_se_from_histories(
        Ln_histories_sgld, cfg.n_data, beta, L0
    )
    llc_hmc_mean, se_hmc, ess_hmc = llc_mean_and_se_from_histories(
        Ln_histories_hmc, cfg.n_data, beta, L0
    )
    llc_mclmc_mean, se_mclmc, ess_mclmc = llc_mean_and_se_from_histories(
        Ln_histories_mclmc, cfg.n_data, beta, L0
    )

    # Separate gradient work from loss evaluations (for fair WNV comparison)
    sgld_grad_work = stats.n_sgld_minibatch_grads  # Only gradient operations
    hmc_grad_work = stats.n_hmc_leapfrog_grads  # Only gradient operations
    # MCLMC work: use override if provided, otherwise default to draws count
    mclmc_grad_work = int(cfg.mclmc_draws * (cfg.mclmc_grad_per_step_override or 1.0))

    # Compute WNV using gradient work only (loss evals are for LLC estimation, not sampling cost)
    wnv_sgld = work_normalized_variance(se_sgld, stats.t_sgld_sampling, sgld_grad_work)
    wnv_hmc = work_normalized_variance(se_hmc, stats.t_hmc_sampling, hmc_grad_work)
    wnv_mclmc = work_normalized_variance(
        se_mclmc, stats.t_mclmc_sampling, mclmc_grad_work
    )

    print(f"SGLD: λ̂={llc_sgld_mean:.4f}, SE={se_sgld:.4f}, ESS={ess_sgld:.1f}")
    print(
        f"      Time: {stats.t_sgld_sampling:.2f}s, WNV-time: {wnv_sgld['WNV_seconds']:.6f}"
    )
    print(f"      Grad work: {sgld_grad_work}, WNV-grad: {wnv_sgld['WNV_grads']:.6f}")
    print(f"      Loss evals: {stats.n_sgld_full_loss} (for LLC estimation)")

    print(f"HMC:  λ̂={llc_hmc_mean:.4f}, SE={se_hmc:.4f}, ESS={ess_hmc:.1f}")
    print(
        f"      Time: {stats.t_hmc_sampling:.2f}s, WNV-time: {wnv_hmc['WNV_seconds']:.6f}"
    )
    print(f"      Grad work: {hmc_grad_work}, WNV-grad: {wnv_hmc['WNV_grads']:.6f}")
    print(f"      Loss evals: {stats.n_hmc_full_loss} (for LLC estimation)")

    print(f"MCLMC: λ̂={llc_mclmc_mean:.4f}, SE={se_mclmc:.4f}, ESS={ess_mclmc:.1f}")
    print(
        f"      Time: {stats.t_mclmc_sampling:.2f}s, WNV-time: {wnv_mclmc['WNV_seconds']:.6f}"
    )
    print(f"      Grad work: {mclmc_grad_work}, WNV-grad: {wnv_mclmc['WNV_grads']:.6f}")
    print(f"      Loss evals: {stats.n_mclmc_full_loss} (for LLC estimation)")

    # WNV efficiency ratios
    wnv_ratio_hmc_sgld_time = (
        wnv_hmc["WNV_seconds"] / wnv_sgld["WNV_seconds"]
        if wnv_sgld["WNV_seconds"] > 0
        else float("inf")
    )
    wnv_ratio_hmc_sgld_grad = (
        wnv_hmc["WNV_grads"] / wnv_sgld["WNV_grads"]
        if wnv_sgld["WNV_grads"] > 0
        else float("inf")
    )
    wnv_ratio_mclmc_sgld_time = (
        wnv_mclmc["WNV_seconds"] / wnv_sgld["WNV_seconds"]
        if wnv_sgld["WNV_seconds"] > 0
        else float("inf")
    )
    wnv_ratio_mclmc_sgld_grad = (
        wnv_mclmc["WNV_grads"] / wnv_sgld["WNV_grads"]
        if wnv_sgld["WNV_grads"] > 0
        else float("inf")
    )

    print("WNV Efficiency Ratios (vs SGLD):")
    print(
        f"  HMC   - Time-normalized: {wnv_ratio_hmc_sgld_time:.3f}, Grad-normalized: {wnv_ratio_hmc_sgld_grad:.3f}"
    )
    print(
        f"  MCLMC - Time-normalized: {wnv_ratio_mclmc_sgld_time:.3f}, Grad-normalized: {wnv_ratio_mclmc_sgld_grad:.3f}"
    )

    print("\n=== Timing Summary (seconds) ===")
    print(f"Build & Data:     {stats.t_build:.2f}")
    print(f"ERM Training:     {stats.t_train:.2f}")
    print(f"SGLD Warmup:      {stats.t_sgld_warmup:.2f}")
    print(f"SGLD Sampling:    {stats.t_sgld_sampling:.2f}")
    print(f"HMC Warmup:       {stats.t_hmc_warmup:.2f}")
    print(f"HMC Sampling:     {stats.t_hmc_sampling:.2f}")
    print(f"MCLMC Warmup:     {stats.t_mclmc_warmup:.2f}")
    print(f"MCLMC Sampling:   {stats.t_mclmc_sampling:.2f}")
    print(f"Total Runtime:    {toc(t0):.2f}")

    print("\n=== Work Summary ===")
    print(f"SGLD - Minibatch grads: {stats.n_sgld_minibatch_grads}")
    print(f"SGLD - Full loss evals: {stats.n_sgld_full_loss}")
    print(f"HMC - Leapfrog grads:   {stats.n_hmc_leapfrog_grads}")
    print(f"HMC - Full loss evals:  {stats.n_hmc_full_loss}")
    print(f"HMC - Warmup grads:     {stats.n_hmc_warmup_leapfrog_grads}")
    print(f"MCLMC - Steps:          {stats.n_mclmc_steps}")
    print(f"MCLMC - Full loss evals: {stats.n_mclmc_full_loss}")

    # Plot diagnostics if enabled
    if cfg.diag_mode != "none":
        if cfg.save_plots:
            print("\n=== Generating Diagnostic Plots ===")

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

    # Save run manifest and README snippet
    # Save data artifacts if enabled
    if run_dir:
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

        # Collect all metrics for consolidated saving
        all_metrics = {
            "sgld_llc_mean": float(llc_sgld_mean),
            "sgld_llc_se": float(se_sgld),
            "sgld_ess": float(ess_sgld),
            "sgld_timing_warmup": float(stats.t_sgld_warmup),
            "sgld_timing_sampling": float(stats.t_sgld_sampling),
            "sgld_n_steps": int(stats.n_sgld_minibatch_grads),
            "sgld_n_full_loss": int(stats.n_sgld_full_loss),
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
            "mclmc_llc_mean": float(llc_mclmc_mean),
            "mclmc_llc_se": float(se_mclmc),
            "mclmc_ess": float(ess_mclmc),
            "mclmc_timing_warmup": float(stats.t_mclmc_warmup),
            "mclmc_timing_sampling": float(stats.t_mclmc_sampling),
            "mclmc_n_steps": int(stats.n_mclmc_steps),
            "mclmc_n_full_loss": int(stats.n_mclmc_full_loss),
        }

        # Save all metrics
        save_metrics(run_dir, all_metrics)

        # Save configuration
        save_config(run_dir, cfg)

        # Create manifest (replaces save_run_manifest)
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

        # Generate HTML gallery (replaces save_readme_snippet functionality)
        gallery_path = generate_gallery_html(run_dir, cfg, all_metrics)
        print(f"HTML gallery: {gallery_path}")

        print(f"Artifacts saved to: {run_dir}")

    print(f"\nDone in {toc(t0):.1f}s.")
    return run_dir


# ----------------------------
# Experiment runner for parameter sweeps
# ----------------------------


if __name__ == "__main__":
    import sys
    import argparse
    import pandas as pd

    # Create main parser
    parser = argparse.ArgumentParser(description="Local Learning Coefficient Analysis")
    sub = parser.add_subparsers(dest="cmd")

    # Single run (default) - inherit from existing CLI
    single_parser = sub.add_parser(
        "run", help="Run single experiment (default)", add_help=False
    )

    # Sweep mode with parallel backends
    sweep_parser = sub.add_parser(
        "sweep", help="Run parameter sweep (optionally parallel)"
    )
    sweep_parser.add_argument(
        "--backend", choices=["local", "submitit", "modal"], default="local"
    )
    sweep_parser.add_argument(
        "--workers", type=int, default=0, help="Local workers (0/1=serial)"
    )
    sweep_parser.add_argument("--n-seeds", type=int, default=2)

    # submitit params
    sweep_parser.add_argument("--partition", type=str, default=None)
    sweep_parser.add_argument("--timeout-min", type=int, default=60)
    sweep_parser.add_argument("--gpus", type=int, default=0)
    sweep_parser.add_argument("--cpus", type=int, default=4)
    sweep_parser.add_argument("--mem-gb", type=int, default=16)
    sweep_parser.add_argument("--account", type=str, default=None)
    sweep_parser.add_argument("--qos", type=str, default=None)
    sweep_parser.add_argument("--constraint", type=str, default=None)

    # timeout and artifact control
    sweep_parser.add_argument(
        "--timeout-s", type=int, default=None, help="Local executor timeout in seconds"
    )
    sweep_parser.add_argument(
        "--modal-timeout-s", type=int, default=3600, help="Modal timeout in seconds"
    )
    sweep_parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save full artifacts (plots, data, HTML)",
    )
    sweep_parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Base artifacts directory",
    )

    # If no subcommand, default to single run behavior
    if len(sys.argv) == 1 or (
        len(sys.argv) > 1 and sys.argv[1] not in ["sweep", "run"]
    ):
        # Parse command line arguments and run main (existing behavior)
        args = parse_args()
        cfg = CFG  # Start with default config

        # Apply preset if specified
        if args.preset:
            cfg = apply_preset(cfg, args.preset)

        # Apply command line overrides
        cfg = override_config(cfg, args)

        # Run main with configured settings
        main(cfg)
    else:
        args, unknown = parser.parse_known_args()

        if args.cmd == "sweep":
            from llc.tasks import run_experiment_task
            from llc.execution import get_executor

            sw = sweep_space()
            work = build_sweep_worklist(sw, n_seeds=args.n_seeds)

            # Transform to the minimal payload the executors expect
            items = []
            for name, param, val, seed, cfg in work:
                cfg_dict = cfg.__dict__.copy()
                # Add artifact configuration
                if args.save_artifacts:
                    cfg_dict["save_artifacts"] = True
                    cfg_dict["artifacts_dir"] = args.artifacts_dir

                items.append(
                    {
                        "cfg": cfg_dict,
                        "tag": {
                            "sweep": name,
                            "param": param,
                            "value": val,
                            "seed": seed,
                        },
                    }
                )

            # Pick executor and run
            if args.backend == "local":
                ex = get_executor(
                    "local", workers=args.workers, timeout_s=args.timeout_s
                )
                results = ex.map(lambda it: run_experiment_task(it["cfg"]), items)
            elif args.backend == "submitit":
                slurm_additional = {}
                if args.account:
                    slurm_additional["account"] = args.account
                if args.qos:
                    slurm_additional["qos"] = args.qos
                if args.constraint:
                    slurm_additional["constraint"] = args.constraint

                ex = get_executor(
                    "submitit",
                    folder="slurm_logs",
                    timeout_min=args.timeout_min,
                    slurm_partition=args.partition,
                    gpus_per_node=args.gpus,
                    cpus_per_task=args.cpus,
                    mem_gb=args.mem_gb,
                    name="llc",
                    slurm_additional_parameters=slurm_additional
                    if slurm_additional
                    else None,
                )
                results = ex.map(lambda it: run_experiment_task(it["cfg"]), items)
            elif args.backend == "modal":
                try:
                    from modal_app import run_experiment_remote
                except ImportError:
                    raise RuntimeError(
                        "Modal app not available. Ensure modal_app.py is present and modal is installed."
                    )

                # Configure Modal function with timeout
                modal_options = {}
                if args.modal_timeout_s:
                    modal_options["timeout"] = args.modal_timeout_s

                ex = get_executor(
                    "modal", remote_fn=run_experiment_remote, options=modal_options
                )
                # pass only the cfg dict (modal function signature matches)
                results = ex.map(None, [it["cfg"] for it in items])
            else:
                raise SystemExit(f"Unknown backend: {args.backend}")

            # Fold results into a DataFrame
            rows = []
            for it, r in zip(items, results):
                rows.append(
                    {
                        "sweep": it["tag"]["sweep"],
                        "param": it["tag"]["param"],
                        "value": it["tag"]["value"],
                        "seed": it["tag"]["seed"],
                        "llc_sgld": r.get("llc_sgld"),
                        "llc_hmc": r.get("llc_hmc"),
                    }
                )

            df = pd.DataFrame(rows)
            df.to_csv("llc_sweep_results.csv", index=False)
            print("\n=== Sweep Results ===")
            print(
                df.groupby(["sweep", "param", "value"]).agg(
                    {
                        "llc_sgld": ["mean", "std"],
                        "llc_hmc": ["mean", "std"],
                        "seed": "count",
                    }
                )
            )
            print("\nResults saved to llc_sweep_results.csv")
        else:
            # Single run mode - use existing behavior
            args = parse_args()
            cfg = CFG

            if args.preset:
                cfg = apply_preset(cfg, args.preset)

            cfg = override_config(cfg, args)
            main(cfg)
