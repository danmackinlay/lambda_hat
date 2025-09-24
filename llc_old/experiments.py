# llc/experiments.py
"""Experiment sweep utilities and training functions"""

from dataclasses import replace

from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
import optax

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
            # Architecture size (parameter count) sweep
            {
                "name": "dim",
                "param": "target_params",
                "values": [500, 1_000, 2_000, 5_000, 10_000],
            },
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
