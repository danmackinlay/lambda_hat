# llc/experiments.py
"""Experiment sweep utilities and training functions"""

from typing import TYPE_CHECKING
from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.flatten_util import ravel_pytree
import optax
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import Config
else:
    # Runtime import to avoid circular dependency
    Config = "Config"

from .config import Config
from .models import infer_widths, init_mlp_params
from .data import make_dataset
from .losses import make_loss_fns
from .posterior import compute_beta_gamma, make_logpost_and_score
from .runners import run_sgld_online, run_hmc_online_with_adaptation
from .diagnostics import llc_ci_from_histories


def train_erm(w_init_pytree, cfg: Config, X, Y, steps=2000, lr=1e-2):
    """Train to empirical risk minimizer using new flexible architecture"""
    theta, unravel = ravel_pytree(w_init_pytree)
    loss_full, _ = make_loss_fns(unravel, cfg, X, Y)
    opt = optax.adam(lr)
    opt_state = opt.init(theta)

    @jit
    def step(theta, opt_state):
        loss, g = value_and_grad(loss_full)(theta)
        updates, opt_state = opt.update(g, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss

    for _ in range(steps):
        theta, opt_state, _ = step(theta, opt_state)

    return theta, unravel  # θ⋆, unravel tied to θ⋆


def sweep_space():
    """Define experiment sweep space"""
    return {
        "base": Config(
            in_dim=32,
            out_dim=1,
            n_data=5000,
            sgld_steps=2000,
            sgld_warmup=500,
            hmc_draws=500,
            hmc_warmup=200,
        ),
        "sweeps": [
            # Architecture sweeps
            {"name": "depth", "param": "depth", "values": [1, 2, 3, 4]},
            {"name": "width", "param": "hidden", "values": [50, 100, 200, 400]},
            {
                "name": "activation",
                "param": "activation",
                "values": ["relu", "tanh", "gelu"],
            },
            # Data sweeps
            {
                "name": "x_dist",
                "param": "x_dist",
                "values": ["gauss_iso", "gauss_aniso", "mixture", "lowdim_manifold"],
            },
            {
                "name": "noise",
                "param": "noise_model",
                "values": ["gauss", "hetero", "student_t"],
            },
            # Loss sweeps
            {"name": "loss", "param": "loss", "values": ["mse", "t_regression"]},
        ],
    }


def build_sweep_worklist(sweep_config, n_seeds=3):
    """Build worklist of configs for parallel execution"""
    base = sweep_config["base"]
    work = []
    for sweep in sweep_config["sweeps"]:
        param, values = sweep["param"], sweep["values"]
        for val in values:
            for seed in range(n_seeds):
                cfg = replace(base, **{param: val, "seed": seed})
                # emit as dict for cross-process pickling safety
                work.append((sweep["name"], param, val, seed, cfg))
    return work


def run_experiment(cfg: Config, verbose=True):
    """Run single experiment and return LLCs"""
    key = random.PRNGKey(cfg.seed)

    # Generate data
    X, Y, _, _ = make_dataset(key, cfg)

    # Initialize network
    key, subkey = random.split(key)
    widths = cfg.widths or infer_widths(
        cfg.in_dim, cfg.out_dim, cfg.depth, cfg.target_params, fallback_width=cfg.hidden
    )
    w0 = init_mlp_params(
        subkey, cfg.in_dim, widths, cfg.out_dim, cfg.activation, cfg.bias, cfg.init
    )

    # Train ERM
    theta_star, unravel = train_erm(w0, cfg, X, Y)

    # Setup sampling
    dim = theta_star.size
    beta, gamma = compute_beta_gamma(cfg, dim)

    # Create loss functions (default dtype for training)
    loss_full, loss_minibatch = make_loss_fns(unravel, cfg, X, Y)

    # Create f64 loss functions for consistent LLC evaluation
    X64, Y64 = X.astype(jnp.float64), Y.astype(jnp.float64)
    params64 = jax.tree_util.tree_map(
        lambda a: a.astype(jnp.float64), unravel(theta_star)
    )
    theta_star64, unravel64 = ravel_pytree(params64)
    loss_full64, loss_minibatch64 = make_loss_fns(unravel64, cfg, X64, Y64)
    L0 = float(loss_full64(theta_star64))
    Ln_full64 = jit(loss_full64)

    # Create log posterior (default dtype for sampling efficiency)
    logpost_grad, grad_minibatch = make_logpost_and_score(
        loss_full, loss_minibatch, theta_star, cfg.n_data, beta, gamma
    )

    # Run SGLD
    key, k_sgld = random.split(key)
    init_sgld = theta_star + 0.01 * random.normal(k_sgld, (cfg.chains, dim))

    _, _, _, _, Ln_hist_sgld = run_sgld_online(
        k_sgld,
        init_sgld,
        grad_minibatch,
        X,
        Y,
        cfg.n_data,
        cfg.sgld_step_size,
        cfg.sgld_steps,
        cfg.sgld_warmup,
        cfg.sgld_batch_size,
        cfg.sgld_eval_every,
        cfg.sgld_thin,
        Ln_full64,  # Use f64 for consistent LLC evaluation
    )

    llc_sgld, _ = llc_ci_from_histories(Ln_hist_sgld, cfg.n_data, beta, L0)

    # Run HMC
    key, k_hmc = random.split(key)
    init_hmc = theta_star64 + 0.01 * random.normal(k_hmc, (cfg.chains, dim))

    _, _, _, _, Ln_hist_hmc = run_hmc_online_with_adaptation(
        k_hmc,
        init_hmc,
        lambda th: jax.value_and_grad(loss_full64)(th),
        cfg.hmc_draws,
        cfg.hmc_warmup,
        cfg.hmc_num_integration_steps,
        cfg.hmc_eval_every,
        cfg.hmc_thin,
        Ln_full64,
    )

    llc_hmc, _ = llc_ci_from_histories(Ln_hist_hmc, cfg.n_data, beta, L0)

    if verbose:
        print(f"LLC: SGLD={llc_sgld:.4f}, HMC={llc_hmc:.4f}")

    return llc_sgld, llc_hmc


def run_sweep(sweep_config, n_seeds=3):
    """Run experiment sweep"""

    base = sweep_config["base"]
    results = []

    for sweep in sweep_config["sweeps"]:
        name = sweep["name"]
        param = sweep["param"]
        values = sweep["values"]

        print(f"\n=== Sweeping {name} ===")
        for val in values:
            print(f"\n{param} = {val}")

            llc_sgld_seeds = []
            llc_hmc_seeds = []

            for seed in range(n_seeds):
                # Create config with swept parameter
                cfg = replace(base, **{param: val, "seed": seed})

                # Run experiment
                try:
                    llc_sgld, llc_hmc = run_experiment(cfg, verbose=False)
                    llc_sgld_seeds.append(llc_sgld)
                    llc_hmc_seeds.append(llc_hmc)
                except Exception as e:
                    print(f"  Seed {seed} failed: {e}")
                    continue

            if llc_sgld_seeds:
                result = {
                    "sweep": name,
                    "param": param,
                    "value": val,
                    "llc_sgld_mean": np.mean(llc_sgld_seeds),
                    "llc_sgld_std": np.std(llc_sgld_seeds),
                    "llc_hmc_mean": np.mean(llc_hmc_seeds),
                    "llc_hmc_std": np.std(llc_hmc_seeds),
                    "n_seeds": len(llc_sgld_seeds),
                }
                results.append(result)
                print(
                    f"  LLC: SGLD={result['llc_sgld_mean']:.3f}±{result['llc_sgld_std']:.3f}, "
                    f"HMC={result['llc_hmc_mean']:.3f}±{result['llc_hmc_std']:.3f}"
                )

    return pd.DataFrame(results)
