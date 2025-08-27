# main.py
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
    # Model sizing
    in_dim: int = 32
    out_dim: int = 1
    target_params: int | None = 10_000   # if set, overrides 'hidden'
    hidden: int = 300                    # used only if target_params=None

    # Data / posterior temperature
    n_data: int = 20_000
    noise_std: float = 0.10
    beta0: float = 1.0
    prior_radius: float = 1.0  # interpretable prior: typical ||w-w0|| ≈ r
    gamma: float | None = None  # computed from prior_radius if None

    # SGLD
    sgld_chains: int = 4
    sgld_steps: int = 4_000
    sgld_warmup: int = 1_000
    sgld_batch_size: int = 256
    sgld_step_size: float = 1e-6
    sgld_thin: int = 20              # store every k-th draw for diagnostics only
    sgld_eval_every: int = 10        # compute full-data L_n(w) every k steps (for LLC mean)
    sgld_dtype: str = "float32"      # reduce memory

    # HMC
    hmc_chains: int = 4
    hmc_draws: int = 1_000
    hmc_warmup: int = 1_000
    hmc_num_integration_steps: int = 10
    hmc_thin: int = 5                # store every k-th draw for diagnostics
    hmc_eval_every: int = 1          # compute L_n(w) every k draws (usually 1 for HMC)
    hmc_dtype: str = "float64"

    # Misc
    seed: int = 42


# Small test config for quick verification
TEST_CFG = Config(
    # Small model
    in_dim=4,
    out_dim=1,
    target_params=50,  # ~50 params only

    # Small data
    n_data=100,

    # Minimal SGLD
    sgld_chains=2,
    sgld_steps=100,
    sgld_warmup=20,
    sgld_eval_every=5,
    sgld_thin=10,

    # Minimal HMC
    hmc_chains=2,
    hmc_draws=50,
    hmc_warmup=20,
    hmc_thin=5,
)

CFG = Config()  # Default full config


# ----------------------------
# Helper: Infer hidden size from target params
# ----------------------------
def infer_hidden(in_dim, out_dim, target_params):
    # params ≈ (in_dim + out_dim) * hidden + hidden + out_dim
    #        = (in_dim + out_dim + 1) * hidden + out_dim
    if target_params is None: return None
    denom = (in_dim + out_dim + 1)
    hidden = max(1, int((target_params - out_dim) // denom))
    return hidden


# ----------------------------
# Dtype helper
# ----------------------------
def as_dtype(x, dtype_str):  # 'float32' or 'float64'
    return x.astype(jnp.float32 if dtype_str == "float32" else jnp.float64)


# ----------------------------
# ERM training to find empirical minimizer
# ----------------------------
def train_to_erm(w_init, X, Y, steps=2000, lr=1e-2):
    theta, unravel = ravel_pytree(w_init)
    opt = optax.adam(lr); opt_state = opt.init(theta)
    @jit
    def step(theta, opt_state):
        loss, g = value_and_grad(lambda th: mse_loss_full(th, unravel, X, Y))(theta)
        updates, opt_state = opt.update(g, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss
    for _ in range(steps):
        theta, opt_state, _ = step(theta, opt_state)
    return theta, unravel  # θ⋆ and unravel that matches θ⋆


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
# Online mean/variance tracking (Welford)
# ----------------------------
class RunningMeanVar:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
    def value(self):
        var = self.M2 / (self.n - 1) if self.n > 1 else jnp.nan
        return self.mean, var, self.n


def llc_from_running_mean(E_L, L0, n, beta):
    return float(n * beta * (E_L - L0))


def stack_thinned(kept_list):  # list of (draws, dim)
    """Stack thinned samples, truncating to common length to avoid NaN padding"""
    m = min(k.shape[0] for k in kept_list)
    if m == 0:
        return np.empty((len(kept_list), 0, kept_list[0].shape[1]))
    return np.stack([k[:m] for k in kept_list], axis=0)


def llc_ci_from_histories(Ln_histories, n, beta, L0, alpha=0.05):
    """Compute LLC with proper CI using ESS from Ln evaluation history"""
    # pack ragged histories to a rectangular array by truncating to min length
    m = min(len(h) for h in Ln_histories if len(h) > 0)
    if m == 0:
        return 0.0, (0.0, 0.0)
    H = np.stack([np.asarray(h[:m]) for h in Ln_histories], axis=0)  # (chains, m)
    idata = az.from_dict(posterior={"L": H})
    ess = float(np.nanmedian(az.ess(idata, var_names=["L"], method="bulk").L.values))
    L_mean = float(np.nanmean(H))
    L_std = float(np.nanstd(H, ddof=1))
    # variance of mean adjusted by ESS
    se = L_std / np.sqrt(max(1.0, ess))
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    llc_mean = n * float(beta) * (L_mean - L0)
    return llc_mean, (llc_mean - z* n*float(beta)*se, llc_mean + z* n*float(beta)*se)


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
# SGLD chains (BlackJAX) - Online memory-efficient version
# ----------------------------
def run_sgld_online(key, init_thetas, grad_logpost_minibatch, n, step_size,
                    num_steps, warmup, batch_size, eval_every, thin,
                    Ln_full64):
    chains = init_thetas.shape[0]
    sgld = blackjax.sgld(grad_logpost_minibatch)
    Ln_running = [RunningMeanVar() for _ in range(chains)]
    Ln_history = [[] for _ in range(chains)]  # track Ln evaluations
    stored = []  # thinned positions for diagnostics (kept small)

    @jit
    def one_step(theta, k, idx):
        return sgld.step(k, theta, idx, step_size)

    keys = random.split(key, chains)
    for c in range(chains):
        k = keys[c]
        subkeys = random.split(k, num_steps)
        theta = init_thetas[c]
        # stream steps
        for t in range(num_steps):
            # Fix RNG hygiene: split for noise and minibatch sampling
            k_noise, k_batch = random.split(subkeys[t])
            idx = random.randint(k_batch, (batch_size,), 0, n)
            theta = one_step(theta, k_noise, idx)
            if t >= warmup and ((t - warmup) % eval_every == 0):
                # Always evaluate in float64 for consistency
                Ln = float(Ln_full64(theta.astype(jnp.float64)))
                Ln_running[c].update(Ln)
                Ln_history[c].append(Ln)
            if t >= warmup and ((t - warmup) % thin == 0):
                stored.append((c, np.array(theta)))  # small footprint

    # Pack thinned samples - avoid NaN padding
    kept_list = [[] for _ in range(chains)]
    for c, th in stored:
        kept_list[c].append(th)

    samples_thin = stack_thinned([np.stack(kl, axis=0) if kl else np.empty((0, init_thetas.shape[1])) for kl in kept_list])

    # Aggregate LLC stats
    means = [rm.value()[0] for rm in Ln_running]
    vars_ = [rm.value()[1] for rm in Ln_running]
    ns = [rm.value()[2] for rm in Ln_running]

    return samples_thin, np.array(means), np.array(vars_), np.array(ns), Ln_history


# ----------------------------
# HMC chains (BlackJAX) - Online memory-efficient version
# ----------------------------
def run_hmc_online_with_adaptation(key, init_thetas, logpost_and_grad,
                                   num_integration_steps, warmup, draws, thin, eval_every,
                                   Ln_full64):
    chains, dim = init_thetas.shape

    def logdensity(theta):
        val, _ = logpost_and_grad(theta)
        return val

    def warm_and_sample(key, theta_init):
        wa = blackjax.window_adaptation(blackjax.hmc, logdensity,
                                        num_integration_steps=num_integration_steps)
        (state, params), _ = wa.run(key, theta_init, num_steps=warmup)

        # ensure diagonal inverse mass matrix (convert dense to diag if needed)
        invM = params.get("inverse_mass_matrix", jnp.ones(dim))
        if invM.ndim == 2:  # dense -> take diagonal
            invM = jnp.diag(invM)
        tuned = dict(step_size=params["step_size"], inverse_mass_matrix=invM)
        step = blackjax.hmc(logdensity, **tuned, num_integration_steps=num_integration_steps).step

        rm = RunningMeanVar()
        kept = []
        accs = []
        Ln_hist = []  # track Ln evaluations

        keys = random.split(key, draws)
        @jit
        def one(state, k):
            return step(k, state)

        for t in range(draws):
            state, info = one(state, keys[t])
            if (t % eval_every) == 0:
                # Always evaluate in float64 for consistency
                Ln = float(Ln_full64(state.position.astype(jnp.float64)))
                rm.update(Ln)
                Ln_hist.append(Ln)
            if (t % thin) == 0:
                kept.append(np.array(state.position))
            accs.append(float(info.acceptance_rate))

        kept = np.stack(kept, axis=0) if kept else np.empty((0, dim))
        return kept, rm.value()[0], rm.value()[1], rm.value()[2], np.array(accs), np.asarray(Ln_hist)

    keys = random.split(key, chains)
    kept_list, means, vars_, ns, accs, Ln_hist_list = [], [], [], [], [], []
    for c in range(chains):
        kept, m, v, n, a, hist = warm_and_sample(keys[c], init_thetas[c])
        kept_list.append(kept)
        means.append(m)
        vars_.append(v)
        ns.append(n)
        accs.append(a)
        Ln_hist_list.append(hist)

    # pack thinned samples - avoid NaN padding
    samples_thin = stack_thinned(kept_list)

    return samples_thin, np.array(means), np.array(vars_), np.array(ns), accs, Ln_hist_list



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

    # Infer hidden size from target params if specified
    hid_from_target = infer_hidden(cfg.in_dim, cfg.out_dim, cfg.target_params)
    hidden = hid_from_target if hid_from_target is not None else cfg.hidden

    w0_pytree, X, Y = make_teacher_and_data(
        key, cfg.n_data, cfg.in_dim, hidden, cfg.out_dim, cfg.noise_std
    )

    # Train to empirical minimizer (ERM) - center the local prior there
    print("Training to empirical minimizer...")
    theta_star_f64, unravel_star_f64 = train_to_erm(w0_pytree, X.astype(jnp.float64), Y.astype(jnp.float64))
    theta_star_f32 = theta_star_f64.astype(jnp.float32)

    # Center the local prior at θ⋆, not at the teacher
    theta0_f64, unravel_f64 = theta_star_f64, unravel_star_f64
    theta0_f32, unravel_f32 = theta_star_f32, unravel_star_f64

    # Create dtype-specific data versions
    X_f32, Y_f32 = as_dtype(X, cfg.sgld_dtype), as_dtype(Y, cfg.sgld_dtype)
    X_f64, Y_f64 = as_dtype(X, cfg.hmc_dtype), as_dtype(Y, cfg.hmc_dtype)

    dim = theta0_f32.size
    print(f"Parameter dimension: {dim:,d}")

    beta = cfg.beta0 / jnp.log(cfg.n_data)
    print(f"beta = {float(beta):.6g}  (with n={cfg.n_data})")

    # Compute interpretable gamma from prior radius
    if cfg.gamma is None:
        # For typical ||w-w0|| ≈ r in d dims, set τ = r/√d ⇒ γ = d/r²
        gamma = dim / (cfg.prior_radius ** 2)
        print(f"gamma = {gamma:.6g} (from prior_radius={cfg.prior_radius}, dim={dim})")
    else:
        gamma = cfg.gamma
        print(f"gamma = {gamma:.6g} (explicit)")

    # log posterior & gradient factories for each dtype
    logpost_and_grad_f32, grad_logpost_minibatch_f32 = make_logpost_and_score(
        X_f32, Y_f32, theta0_f32, unravel_f32, cfg.n_data, beta, gamma
    )
    logpost_and_grad_f64, grad_logpost_minibatch_f64 = make_logpost_and_score(
        X_f64, Y_f64, theta0_f64, unravel_f64, cfg.n_data, beta, gamma
    )

    # Recompute L0 at empirical minimizer (do this in float64 for both samplers)
    L0 = float(mse_loss_full(theta0_f64, unravel_f64, X.astype(jnp.float64), Y.astype(jnp.float64)))
    print(f"L0 at empirical minimizer: {L0:.6f}")

    # Create unified float64 loss evaluator for LLC computation
    Ln_full64 = jit(lambda th64: mse_loss_full(th64, unravel_f64, X_f64, Y_f64))

    # ===== SGLD (Online) =====
    print("\n=== SGLD (BlackJAX, online) ===")
    k_sgld = random.split(key, 1)[0]
    # simple overdispersed inits around w0
    init_thetas_sgld = theta0_f32 + 0.01 * random.normal(k_sgld, (cfg.sgld_chains, dim)).astype(jnp.float32)

    sgld_samples_thin, sgld_Es, sgld_Vars, sgld_Ns, Ln_histories_sgld = run_sgld_online(
        k_sgld,
        init_thetas_sgld,
        grad_logpost_minibatch_f32,
        cfg.n_data,
        cfg.sgld_step_size,
        cfg.sgld_steps,
        cfg.sgld_warmup,
        cfg.sgld_batch_size,
        cfg.sgld_eval_every,
        cfg.sgld_thin,
        Ln_full64
    )

    # Compute LLC with proper CI using ESS
    llc_sgld, ci_sgld = llc_ci_from_histories(Ln_histories_sgld, cfg.n_data, beta, L0)
    print(f"SGLD LLC: {llc_sgld:.4f}  95% CI: [{ci_sgld[0]:.4f}, {ci_sgld[1]:.4f}]")

    # Diagnostics on thinned samples
    if sgld_samples_thin.shape[1] > 1:
        idata_sgld = arviz_from_samples(sgld_samples_thin, name="theta")
        # ESS returns a Dataset with variables, each variable has dimensions
        # For bulk ESS, use method='bulk' (default)
        ess_sgld = az.ess(idata_sgld, var_names=["theta"], method="bulk")
        # rhat returns a Dataset with variables
        rhat_sgld = az.rhat(idata_sgld, var_names=["theta"])
        print("SGLD diagnostics (thinned samples):")
        # Access the theta variable from the Dataset, then get its values
        print(
            f"  ESS bulk (median over dims): {np.nanmedian(ess_sgld.theta.values):.1f}"
        )
        print(
            f"  R-hat (max over dims):        {np.nanmax(rhat_sgld.theta.values):.3f}"
        )

    # ===== HMC (Online) =====
    print("\n=== HMC (BlackJAX, online) ===")
    k_hmc = random.fold_in(key, 123)
    init_thetas_hmc = theta0_f64 + 0.01 * random.normal(k_hmc, (cfg.hmc_chains, dim))

    hmc_samples_thin, hmc_Es, hmc_Vars, hmc_Ns, accs_hmc, Ln_histories_hmc = run_hmc_online_with_adaptation(
        k_hmc,
        init_thetas_hmc,
        logpost_and_grad_f64,
        cfg.hmc_num_integration_steps,
        cfg.hmc_warmup,
        cfg.hmc_draws,
        cfg.hmc_thin,
        cfg.hmc_eval_every,
        Ln_full64
    )

    # Compute LLC with proper CI using ESS
    llc_hmc, ci_hmc = llc_ci_from_histories(Ln_histories_hmc, cfg.n_data, beta, L0)
    mean_acc = float(np.mean([a.mean() for a in accs_hmc]))

    print(f"HMC LLC: {llc_hmc:.4f}  95% CI: [{ci_hmc[0]:.4f}, {ci_hmc[1]:.4f}]")
    print(f"HMC acceptance rate (mean over chains/draws): {mean_acc:.3f}")

    # Diagnostics on thinned samples
    if hmc_samples_thin.shape[1] > 1:
        idata_hmc = arviz_from_samples(hmc_samples_thin, name="theta")
        # ESS returns a Dataset with variables, each variable has dimensions
        # For bulk ESS, use method='bulk' (default)
        ess_hmc = az.ess(idata_hmc, var_names=["theta"], method="bulk")
        # rhat returns a Dataset with variables
        rhat_hmc = az.rhat(idata_hmc, var_names=["theta"])
        print("HMC diagnostics (thinned samples):")
        # Access the theta variable from the Dataset, then get its values
        print(f"  ESS bulk (median over dims): {np.nanmedian(ess_hmc.theta.values):.1f}")
        print(
            f"  R-hat (max over dims):        {np.nanmax(rhat_hmc.theta.values):.3f}"
        )

    # LLC confidence interval from running statistics
    def llc_ci_from_running(L_means, L_vars, L_ns, n, beta, L0, alpha=0.05):
        import scipy.stats as st
        # combine chain means via simple average; use within-chain SE pooled / n_chains
        llc_chain = n * beta * (L_means - L0)
        se_chain = n * beta * np.sqrt(L_vars / np.maximum(1, (L_ns - 1)))
        # conservative: combine via mean of variances / sqrt(C)
        se = float(np.sqrt(np.nanmean(se_chain**2) / max(1, len(se_chain))))
        z = st.norm.ppf(1 - alpha/2)
        return float(np.nanmean(llc_chain)), (llc_chain.mean() - z*se, llc_chain.mean() + z*se)

    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    # Use CFG for full run, TEST_CFG for quick testing
    main(CFG)  # Change to TEST_CFG for quick tests
