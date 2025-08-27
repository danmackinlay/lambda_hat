# llc_sgld_hmc_blackjax.py
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")  # HMC benefits from float64

import time
from dataclasses import dataclass
from functools import partial

import arviz as az
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad, grad
from jax.flatten_util import ravel_pytree

import blackjax
import numpy as np
import optax


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # Network (roughly ~10k params)
    in_dim: int = 32
    hidden: int = 300
    out_dim: int = 1

    # Data / posterior temperature
    n_data: int = 20_000
    noise_std: float = 0.10
    beta0: float = 1.0  # beta = beta0 / log(n)
    gamma: float = 1.0  # prior precision; log phi = -(gamma/2)||w-w0||^2

    # SGLD
    sgld_chains: int = 4
    sgld_steps: int = 4_000
    sgld_warmup: int = 1_000
    sgld_batch_size: int = 256
    sgld_step_size: float = 1e-6  # tune if needed; small for stability in 10k-dim

    # HMC
    hmc_chains: int = 4
    hmc_draws: int = 1_000
    hmc_warmup: int = 1_000
    hmc_num_integration_steps: int = 10  # L

    seed: int = 42


CFG = Config()


# ----------------------------
# Small MLP (~10k params)
# ----------------------------
def init_mlp_params(key, in_dim, hidden, out_dim):
    k1, k2 = random.split(key)
    W1 = random.normal(k1, (hidden, in_dim)) / jnp.sqrt(in_dim)
    b1 = jnp.zeros((hidden,))
    W2 = random.normal(k2, (out_dim, hidden)) / jnp.sqrt(hidden)
    b2 = jnp.zeros((out_dim,))
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return params


def mlp_forward(params, x):
    h = jnp.maximum(0.0, x @ params["W1"].T + params["b1"])  # ReLU
    y = h @ params["W2"].T + params["b2"]
    return y


def count_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


# ----------------------------
# Dataset (teacher-student)
# ----------------------------
def make_teacher_and_data(key, n, in_dim, hidden, out_dim, noise_std):
    k_init, k_x, k_eps = random.split(key, 3)
    w0_pytree = init_mlp_params(k_init, in_dim, hidden, out_dim)
    X = random.normal(k_x, (n, in_dim))
    y_clean = mlp_forward(w0_pytree, X)
    eps = noise_std * random.normal(k_eps, y_clean.shape)
    Y = y_clean + eps
    return w0_pytree, X, Y


# Empirical MSE (Eq. (3.7) style; mean over data)
def mse_loss_full(theta, unravel_fn, X, Y):
    params = unravel_fn(theta)
    pred = mlp_forward(params, X)
    return jnp.mean((pred - Y) ** 2)


def mse_loss_minibatch(theta, unravel_fn, Xb, Yb):
    params = unravel_fn(theta)
    pred = mlp_forward(params, Xb)
    return jnp.mean((pred - Yb) ** 2)


# ----------------------------
# Log posterior & score (tempered + localized)
# log pi(w) = - n * beta * L_n(w) - (gamma/2)||w - w0||^2
# ----------------------------
def make_logpost_and_score(X, Y, theta0, unravel, n, beta, gamma):
    # Full-batch log posterior and grad (for HMC)
    def logpost(theta):
        Ln = mse_loss_full(theta, unravel, X, Y)
        lp = -0.5 * gamma * jnp.sum((theta - theta0) ** 2)
        return lp - n * beta * Ln

    logpost_and_grad = value_and_grad(logpost)

    # SGLD gradient estimator (mini-batch)
    @jit
    def grad_logpost_minibatch(theta, batch):
        # 'batch' carries indices
        idx = batch
        Xb, Yb = X[idx], Y[idx]
        # grad of mean loss on batch, then scale by n to estimate grad of n * L_n
        g_Lb = grad(lambda th: mse_loss_minibatch(th, unravel, Xb, Yb))(theta)
        score = -gamma * (theta - theta0) - beta * n * g_Lb
        return score

    return logpost_and_grad, grad_logpost_minibatch


# ----------------------------
# SGLD chains (BlackJAX)
# API: sgld = blackjax.sgld(grad_logdensity); sgld.step(rng, position, minibatch, step_size)
# (See Sampling Book SGLD examples). :contentReference[oaicite:3]{index=3}
# ----------------------------
def run_sgld(
    key,
    init_thetas,
    grad_logpost_minibatch,
    n,
    step_size,
    num_steps,
    warmup,
    batch_size,
):
    chains = init_thetas.shape[0]
    dim = init_thetas.shape[1]
    sgld = blackjax.sgld(grad_logpost_minibatch)

    @jit
    def one_chain(key, theta0):
        # pre-generate mini-batch indices
        keys = random.split(key, num_steps)
        idxs = random.randint(keys[0], (num_steps, batch_size), 0, n)

        def body(theta, t):
            k = keys[t]
            theta = sgld.step(k, theta, idxs[t], step_size)
            return theta, theta

        _, thetas = jax.lax.scan(body, theta0, jnp.arange(num_steps))
        return thetas[warmup:]  # discard warmup

    keys = random.split(key, chains)
    samples = jax.vmap(one_chain)(keys, init_thetas)  # (chains, draws, dim)
    return samples


# ----------------------------
# HMC chains (BlackJAX)
# We use window adaptation to tune step size & mass matrix. :contentReference[oaicite:4]{index=4}
# ----------------------------
def run_hmc_with_adaptation(
    key, init_thetas, logpost_and_grad, num_integration_steps, warmup, draws
):
    chains, dim = init_thetas.shape

    def logdensity(theta):
        val, _ = logpost_and_grad(theta)
        return val

    # Warm-up uses Stan-style window adaptation in BlackJAX (here with HMC).
    def warm_one_chain(key, theta_init):
        wa = blackjax.window_adaptation(
            blackjax.hmc, logdensity, num_integration_steps=num_integration_steps
        )
        (state, params), _ = wa.run(key, theta_init, num_steps=warmup)
        kernel = blackjax.hmc(
            logdensity, **params, num_integration_steps=num_integration_steps
        ).step

        @jit
        def draw_samples(rng_key, init_state, n_draws):
            def step_fn(state, k):
                state, info = kernel(k, state)
                return state, (state.position, info.acceptance_probability)

            keys = random.split(rng_key, n_draws)
            final_state, (positions, acc_probs) = jax.lax.scan(step_fn, state, keys)
            return positions, acc_probs

        # Run post-warmup draws
        subk1, subk2 = random.split(key)
        positions, acc = draw_samples(subk2, state, draws)
        return positions, acc  # (draws, dim), (draws,)

    keys = random.split(key, chains)
    out = jax.vmap(warm_one_chain)(keys, init_thetas)
    samples, acc = out[0], out[1]  # (chains, draws, dim), (chains, draws)
    return samples, acc


# ----------------------------
# LLC estimator & diagnostics
# ----------------------------
def compute_llc(samples, theta_unravel, X, Y, n, beta, L0):
    # samples: (chains, draws, dim)
    chains, draws, dim = samples.shape

    # compute L_n(w) on the full dataset for each sample (vectorized)
    def Ln(theta):
        return mse_loss_full(theta, theta_unravel, X, Y)

    Ln_vm = jit(jax.vmap(jax.vmap(Ln)))  # over chains, draws
    Ln_vals = Ln_vm(samples)
    E_L = Ln_vals.mean()
    llc = n * beta * (E_L - L0)
    return float(llc), Ln_vals  # also return per-sample Ln for uncertainty


def arviz_from_samples(samples, name="theta"):
    """Build a light InferenceData to run ESS/Rhat, without printing 10k parameters."""
    samples_np = np.asarray(samples)  # (chains, draws, dim)
    coords = {"theta_dim": np.arange(samples_np.shape[-1])}
    dims = {name: ["theta_dim"]}
    idata = az.from_dict(posterior={name: samples_np}, coords=coords, dims=dims)
    return idata


# ----------------------------
# Main
# ----------------------------
def main(cfg: Config = CFG):
    t0 = time.time()
    print("=== Building teacher and data ===")
    key = random.PRNGKey(cfg.seed)
    w0_pytree, X, Y = make_teacher_and_data(
        key, cfg.n_data, cfg.in_dim, cfg.hidden, cfg.out_dim, cfg.noise_std
    )
    theta0, unravel = ravel_pytree(w0_pytree)
    dim = theta0.size
    print(f"Parameter dimension: {dim:,d}")

    beta = cfg.beta0 / jnp.log(cfg.n_data)
    print(f"beta = {float(beta):.6g}  (with n={cfg.n_data})")

    # log posterior & gradient factories
    logpost_and_grad, grad_logpost_minibatch = make_logpost_and_score(
        X, Y, theta0, unravel, cfg.n_data, beta, cfg.gamma
    )

    # reference L_n(w0) on the full data
    L0 = float(mse_loss_full(theta0, unravel, X, Y))

    # ===== SGLD =====
    print("\n=== SGLD (BlackJAX) ===")
    k_sgld = random.split(key, 1)[0]
    # simple overdispersed inits around w0
    init_thetas_sgld = theta0 + 0.01 * random.normal(k_sgld, (cfg.sgld_chains, dim))
    samples_sgld = run_sgld(
        k_sgld,
        init_thetas_sgld,
        grad_logpost_minibatch,
        cfg.n_data,
        cfg.sgld_step_size,
        cfg.sgld_steps,
        cfg.sgld_warmup,
        cfg.sgld_batch_size,
    )
    idata_sgld = arviz_from_samples(samples_sgld, name="theta")
    summ_sgld = az.summary(
        idata_sgld, var_names=["theta"], kind="stats", stat_focus="median"
    )
    ess_sgld = az.ess(idata_sgld)
    rhat_sgld = az.rhat(idata_sgld)
    llc_sgld, Ln_vals_sgld = compute_llc(
        samples_sgld, unravel, X, Y, cfg.n_data, beta, L0
    )
    print(f"SGLD LLC estimate: {llc_sgld:.4f}")
    print("SGLD diagnostics (aggregates over dims):")
    print(
        f"  ESS bulk (median over dims): {np.nanmedian(ess_sgld.ess_bulk.values):.1f}"
    )
    print(
        f"  R-hat (max over dims):        {np.nanmax(rhat_sgld.to_array().values):.3f}"
    )

    # ===== HMC =====
    print("\n=== HMC (BlackJAX, window adaptation) ===")
    k_hmc = random.fold_in(key, 123)
    init_thetas_hmc = theta0 + 0.01 * random.normal(k_hmc, (cfg.hmc_chains, dim))
    samples_hmc, acc_hmc = run_hmc_with_adaptation(
        k_hmc,
        init_thetas_hmc,
        logpost_and_grad,
        cfg.hmc_num_integration_steps,
        cfg.hmc_warmup,
        cfg.hmc_draws,
    )
    idata_hmc = arviz_from_samples(samples_hmc, name="theta")
    ess_hmc = az.ess(idata_hmc)
    rhat_hmc = az.rhat(idata_hmc)
    llc_hmc, Ln_vals_hmc = compute_llc(samples_hmc, unravel, X, Y, cfg.n_data, beta, L0)
    mean_acc = float(np.asarray(acc_hmc).mean())
    print(f"HMC LLC estimate: {llc_hmc:.4f}")
    print(f"HMC acceptance rate (mean over chains/draws): {mean_acc:.3f}")
    print("HMC diagnostics (aggregates over dims):")
    print(f"  ESS bulk (median over dims): {np.nanmedian(ess_hmc.ess_bulk.values):.1f}")
    print(
        f"  R-hat (max over dims):        {np.nanmax(rhat_hmc.to_array().values):.3f}"
    )

    # Uncertainty on LLC via sampling variability of E[L_n(w)]
    def llc_ci(Ln_vals, n, beta, L0):
        # simple approx from draws: transform mean(Ln)-L0 by n*beta
        means = np.asarray(Ln_vals).mean(axis=(0, 1))  # scalar
        std = (
            np.asarray(Ln_vals).mean(axis=(0, 1), keepdims=True) - means
        )  # not used; quick placeholder
        return n * beta * (means - L0)

    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
