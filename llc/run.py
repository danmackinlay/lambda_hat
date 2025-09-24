import os
import logging
import jax
import jax.numpy as jnp
import arviz as az
import matplotlib.pyplot as plt
from jax import random

from .config import Config
from .cache import run_id, run_family_id
from .targets import build_target
from .posterior import compute_beta_gamma, make_logpost_and_score, make_logdensity_for_mclmc
from .samplers.utils import build_tiny_store
from .samplers.sgld import run_sgld_batched
from .samplers.sgnht import run_sgnht_batched
from .samplers.hmc import run_hmc_batched
from .samplers.mclmc import run_mclmc_batched
from .analysis import to_idata, llc_point_se, efficiency_metrics, fig_running_llc
from .artifacts import save_config, save_metrics, save_L0, generate_gallery_html

logger = logging.getLogger(__name__)

def run_one(cfg: Config, *, save_artifacts=True, skip_if_exists=True):
    """Atomic run: exactly one sampler; returns {run_dir, metrics}."""
    if not cfg.samplers or len(cfg.samplers) != 1:
        raise SystemExit("run_one requires exactly one sampler; use sweep for many.")

    rid = run_id(cfg)
    run_dir = os.path.join(cfg.runs_dir, rid) if save_artifacts else ""

    if save_artifacts:
        os.makedirs(run_dir, exist_ok=True)
        save_config(run_dir, cfg)

    key = random.PRNGKey(cfg.seed)
    bundle = build_target(key, cfg)
    beta, gamma = compute_beta_gamma(cfg, bundle.d)

    logpost_and_grad_f32, grad_minibatch_f32 = make_logpost_and_score(
        bundle.loss_full_f32, bundle.loss_minibatch_f32, bundle.theta0_f32, cfg.n_data, beta, gamma
    )
    logpost_and_grad_f64, _ = make_logpost_and_score(
        bundle.loss_full_f64, bundle.loss_minibatch_f64, bundle.theta0_f64, cfg.n_data, beta, gamma
    )
    logdensity_mclmc = make_logdensity_for_mclmc(bundle.loss_full_f64, bundle.theta0_f64, cfg.n_data, beta, gamma)
    Ln_full64_vmapped = jax.jit(jax.vmap(bundle.loss_full_f64))

    # tiny-store diagnostics: first k dimensions for theta traces
    k = min(8, bundle.d)
    tiny = build_tiny_store(diag_dims=list(range(k)), Rproj=None)

    sampler = cfg.samplers[0]
    if sampler == "sgld":
        # init near θ*
        init = bundle.theta0_f32 + 0.01*random.normal(key, (cfg.chains, bundle.d)).astype(jnp.float32)
        res = run_sgld_batched(
            key=key, init_thetas=init, grad_logpost_minibatch=grad_minibatch_f32,
            X=bundle.X_f32, Y=bundle.Y_f32, n=cfg.n_data, step_size=cfg.sgld_step_size,
            num_steps=cfg.sgld_steps, warmup=cfg.sgld_warmup, batch_size=cfg.sgld_batch_size,
            eval_every=cfg.sgld_eval_every, thin=cfg.sgld_thin, Ln_full64_vmapped=Ln_full64_vmapped,
            tiny_store_fn=tiny, precond_mode=cfg.sgld_precond, beta1=cfg.sgld_beta1,
            beta2=cfg.sgld_beta2, eps=cfg.sgld_eps, bias_correction=cfg.sgld_bias_correction
        )
    elif sampler == "sgnht":
        init = bundle.theta0_f32 + 0.01*random.normal(key, (cfg.chains, bundle.d)).astype(jnp.float32)
        res = run_sgnht_batched(
            key=key, init_thetas=init, grad_logpost_minibatch=grad_minibatch_f32,
            X=bundle.X_f32, Y=bundle.Y_f32, n=cfg.n_data, step_size=cfg.sgnht_step_size,
            num_steps=cfg.sgnht_steps, warmup=cfg.sgnht_warmup, batch_size=cfg.sgnht_batch_size,
            eval_every=cfg.sgnht_eval_every, thin=cfg.sgnht_thin, Ln_full64_vmapped=Ln_full64_vmapped,
            tiny_store_fn=tiny, alpha0=cfg.sgnht_alpha0
        )
    elif sampler == "hmc":
        init = bundle.theta0_f64 + 0.01*random.normal(key, (cfg.chains, bundle.d))
        res = run_hmc_batched(
            key=key, init_thetas=init, logpost_and_grad=logpost_and_grad_f64,
            draws=cfg.hmc_draws, warmup=cfg.hmc_warmup, L=cfg.hmc_num_integration_steps,
            eval_every=cfg.hmc_eval_every, thin=cfg.hmc_thin, Ln_full64_vmapped=Ln_full64_vmapped,
            tiny_store_fn=tiny, tuned_dir=(run_dir if save_artifacts else None)
        )
    elif sampler == "mclmc":
        init = bundle.theta0_f64 + 0.01*random.normal(key, (cfg.chains, bundle.d))
        res = run_mclmc_batched(
            key=key, init_thetas=init, logdensity_fn=logdensity_mclmc, draws=cfg.mclmc_draws,
            eval_every=cfg.mclmc_eval_every, thin=cfg.mclmc_thin, Ln_full64_vmapped=Ln_full64_vmapped,
            tiny_store_fn=tiny, num_steps=cfg.mclmc_num_steps, frac_tune1=cfg.mclmc_frac_tune1,
            frac_tune2=cfg.mclmc_frac_tune2, frac_tune3=cfg.mclmc_frac_tune3,
            diagonal_preconditioning=cfg.mclmc_diagonal_preconditioning,
            desired_energy_var=cfg.mclmc_desired_energy_var, trust_in_estimate=cfg.mclmc_trust_in_estimate,
            num_effective_samples=cfg.mclmc_num_effective_samples, integrator_name=cfg.mclmc_integrator,
            tuned_dir=(run_dir if save_artifacts else None)
        )
    else:
        raise SystemExit(f"unknown sampler {sampler}")

    # res = (Ln_histories, theta_thin, acceptance, energy, timings, work)
    Ln_histories, theta_thin, acceptance, energy, timings, work = res

    # Create ArviZ InferenceData
    idata = to_idata(
        Ln_histories=Ln_histories,
        theta_thin=theta_thin,
        acceptance=acceptance,
        energy=energy,
        n=cfg.n_data,
        beta=beta,
        L0=bundle.L0
    )
    idata.attrs.update({"n_data": int(cfg.n_data), "beta": float(beta), "L0": float(bundle.L0)})

    if save_artifacts:
        az.to_netcdf(idata, os.path.join(run_dir, f"{sampler}.nc"))

    # Compute metrics
    core = llc_point_se(idata)
    eff = efficiency_metrics(
        idata=idata,
        timings=timings,
        work=work,
        n_data=cfg.n_data,
        sgld_batch=(cfg.sgld_batch_size if sampler == "sgld" else
                   cfg.sgnht_batch_size if sampler == "sgnht" else None)
    )
    metrics = {f"{sampler}_{k}": v for k, v in {**core, **eff}.items()}
    metrics["family_id"] = run_family_id(cfg)

    if save_artifacts:
        save_L0(run_dir, bundle.L0)
        save_metrics(run_dir, metrics)

        # Save plots if requested
        if cfg.save_plots:
            fig = fig_running_llc(idata, cfg.n_data, beta, bundle.L0, f"{sampler.upper()} Running LLC")
            fig.savefig(os.path.join(run_dir, f"{sampler}_running_llc.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Generate HTML gallery
        generate_gallery_html(run_dir, sampler)

    return {"run_dir": run_dir, "metrics": metrics}