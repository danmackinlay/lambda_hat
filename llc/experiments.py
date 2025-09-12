# llc/experiments.py
"""Experiment sweep utilities and training functions"""

from dataclasses import replace

from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
import optax
import numpy as np
import pandas as pd

from .config import Config
from .losses import make_loss_fns


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
    """
    Thin wrapper around pipeline that returns only SGLD and HMC LLCs.
    Maintained for backwards compatibility with existing code.
    """
    from .pipeline import run_one

    # Run the full pipeline but don't save artifacts
    outputs = run_one(cfg, save_artifacts=False, skip_if_exists=False)

    # Extract SGLD and HMC results
    llc_sgld = outputs.metrics.get("sgld_llc_mean", 0.0)
    llc_hmc = outputs.metrics.get("hmc_llc_mean", 0.0)

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
