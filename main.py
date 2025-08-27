# main.py
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")  # HMC benefits from float64

import time
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

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
    # Model architecture
    in_dim: int = 32
    out_dim: int = 1
    depth: int = 1                    # number of hidden layers
    widths: Optional[list[int]] = None  # per-layer widths; if None, auto-infer
    activation: Literal["relu","tanh","gelu","identity"] = "relu"
    bias: bool = True
    skip_connections: bool = False
    residual_period: int = 2          # every k layers add skip if enabled
    layernorm: bool = False           # (default False; can destabilize HMC)
    init: Literal["he","xavier","lecun","orthogonal"] = "he"

    # Size control
    target_params: Optional[int] = 10_000  # if provided, fix total d and infer widths
    # keep old 'hidden' for backward compatibility
    hidden: int = 300                    # used only if target_params=None and widths=None

    # Data
    n_data: int = 20_000
    x_dist: Literal["gauss_iso","gauss_aniso","mixture","lowdim_manifold","heavy_tail"] = "gauss_iso"
    cov_decay: float = 0.95           # for anisotropy: eigvals ~ cov_decay**i
    mixture_k: int = 4
    mixture_spread: float = 2.0
    x_dim_latent: int = 2             # for low-dim manifold
    noise_model: Literal["gauss","hetero","student_t","outliers"] = "gauss"
    noise_scale: float = 0.1
    hetero_scale: float = 0.1
    student_df: float = 4.0
    outlier_frac: float = 0.05
    outlier_scale: float = 2.0

    # Teacher (can differ from student)
    teacher_depth: Optional[int] = None
    teacher_widths: Optional[list[int]] = None
    teacher_activation: Optional[str] = None
    teacher_dropout_rate: float = 0.0 # stochastic teacher if >0 (only during data gen)

    # Loss / likelihood
    loss: Literal["mse","t_regression"] = "mse"

    # Local posterior (tempering + prior)
    beta_mode: Literal["1_over_log_n","fixed"] = "1_over_log_n"
    beta0: float = 1.0
    prior_radius: Optional[float] = None  # if set, gamma = d / prior_radius**2
    gamma: float = 1.0                    # used only if prior_radius None

    # Sampling
    sampler: Literal["sgld","hmc","nuts"] = "sgld"
    chains: int = 4

    # SGLD
    sgld_steps: int = 4_000
    sgld_warmup: int = 1_000
    sgld_batch_size: int = 256
    sgld_step_size: float = 1e-6
    sgld_thin: int = 20              # store every k-th draw for diagnostics only
    sgld_eval_every: int = 10        # compute full-data L_n(w) every k steps (for LLC mean)
    sgld_dtype: str = "float32"      # reduce memory

    # HMC
    hmc_draws: int = 1_000
    hmc_warmup: int = 1_000
    hmc_num_integration_steps: int = 10
    hmc_thin: int = 5                # store every k-th draw for diagnostics
    hmc_eval_every: int = 1          # compute L_n(w) every k draws (usually 1 for HMC)
    hmc_dtype: str = "float64"

    # NUTS
    nuts_draws: int = 1_000
    nuts_warmup: int = 1_000
    nuts_thin: int = 5
    nuts_eval_every: int = 1
    nuts_dtype: str = "float64"

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

    # Minimal sampling
    chains=2,
    sgld_steps=100,
    sgld_warmup=20,
    sgld_eval_every=5,
    sgld_thin=10,

    # Minimal HMC
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


def infer_widths(in_dim: int, out_dim: int, depth: int, target_params: Optional[int], fallback_width: int = 128) -> list[int]:
    """Infer widths to hit target_params, or use fallback"""
    if target_params is None:
        return [fallback_width] * depth
    # For simplicity, use constant width h and solve approximately:
    # P(h) = (in_dim+1)h + (L-1)(h+1)h + (h+1)out_dim ≈ target_params
    L = depth
    a = (L - 1)  # coefficient of h^2
    b = (in_dim + 1) + (L - 1) + out_dim + 1  # coefficient of h
    if a == 0:
        h = max(1, (target_params - out_dim) // (in_dim + 1))
    else:
        disc = b*b + 4*a*target_params
        h = int((-b + jnp.sqrt(disc)) / (2*a))
        h = int(max(1, h))
    return [h] * L


def compute_beta_gamma(cfg: Config, d: int) -> tuple[float, float]:
    """Compute beta and gamma from config and dimension"""
    beta = cfg.beta0 / jnp.log(cfg.n_data) if cfg.beta_mode == "1_over_log_n" else cfg.beta0
    gamma = (d / (cfg.prior_radius**2)) if (cfg.prior_radius is not None) else cfg.gamma
    return float(beta), float(gamma)


# ----------------------------
# Dtype helper
# ----------------------------
def as_dtype(x, dtype_str):  # 'float32' or 'float64'
    return x.astype(jnp.float32 if dtype_str == "float32" else jnp.float64)


# ----------------------------
# ERM training to find empirical minimizer
# ----------------------------
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


# Legacy ERM training for backward compatibility
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
# Flexible MLP with arbitrary depth and activations
# ----------------------------
def act_fn(name: str):
    """Activation function factory"""
    activations = {
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "gelu": jax.nn.gelu,
        "identity": (lambda x: x)
    }
    return activations[name]


def fan_in_init(key, shape, scheme: str, fan_in: int):
    """Weight initialization schemes"""
    if scheme == "he":
        scale = jnp.sqrt(2.0 / fan_in)
    elif scheme == "xavier":
        scale = jnp.sqrt(1.0 / fan_in)
    elif scheme == "lecun":
        scale = jnp.sqrt(1.0 / fan_in)
    elif scheme == "orthogonal":
        # Simple orthogonal init fallback
        q, _ = jnp.linalg.qr(random.normal(key, shape))
        return q[:shape[0], :shape[1]]
    else:
        scale = 1.0
    return random.normal(key, shape) * scale


def init_mlp_params(key, in_dim: int, widths: list[int], out_dim: int, activation: str, bias: bool, init: str):
    """Initialize MLP with arbitrary depth"""
    keys = random.split(key, len(widths) + 1)
    layers = []
    prev = in_dim

    for i, h in enumerate(widths):
        W = fan_in_init(keys[i], (h, prev), init, prev)
        b = jnp.zeros((h,)) if bias else None
        layers.append({"W": W, "b": b})
        prev = h

    # Output layer (linear)
    W = fan_in_init(keys[-1], (out_dim, prev), "xavier", prev)
    b = jnp.zeros((out_dim,)) if bias else None
    out_layer = {"W": W, "b": b}

    return {
        "layers": layers,
        "out": out_layer
    }


def mlp_forward(params, x, activation: str = "relu", skip: bool = False, residual_period: int = 2, layernorm: bool = False):
    """Forward pass with optional residuals and layer norm"""
    act = act_fn(activation)
    bias = True  # assume bias exists in params

    h = x
    for i, lyr in enumerate(params["layers"]):
        z = h @ lyr["W"].T + (lyr["b"] if bias and lyr["b"] is not None else 0.0)

        if layernorm:
            mu = jnp.mean(z, axis=-1, keepdims=True)
            sig = jnp.std(z, axis=-1, keepdims=True) + 1e-6
            z = (z - mu) / sig

        h_new = act(z)

        if skip and (i % residual_period == residual_period - 1):
            # Project if dimensions differ
            if h.shape[-1] != h_new.shape[-1]:
                P = jnp.eye(h_new.shape[-1], h.shape[-1])[:h_new.shape[-1], :h.shape[-1]]
                h = h @ P.T
            h = h + h_new
        else:
            h = h_new

    # Output layer
    y = h @ params["out"]["W"].T + (params["out"]["b"] if bias and params["out"]["b"] is not None else 0.0)
    return y


def count_params(params):
    """Count total parameters in MLP"""
    leaves = []
    for lyr in params["layers"]:
        leaves.append(lyr["W"])
        if params["bias"] and lyr["b"] is not None:
            leaves.append(lyr["b"])
    leaves.append(params["out"]["W"])
    if params["bias"] and params["out"]["b"] is not None:
        leaves.append(params["out"]["b"])
    return sum(p.size for p in leaves)


# ----------------------------
# Flexible data generation
# ----------------------------
def sample_X(key, cfg: Config, n: int, in_dim: int):
    """Sample inputs according to various distributions"""
    if cfg.x_dist == "gauss_iso":
        return random.normal(key, (n, in_dim))
    elif cfg.x_dist == "gauss_aniso":
        vals = jnp.array([cfg.cov_decay**i for i in range(in_dim)])
        A = jnp.diag(jnp.sqrt(vals))
        Z = random.normal(key, (n, in_dim))
        return Z @ A.T
    elif cfg.x_dist == "mixture":
        keys = random.split(key, 2)
        centers = random.normal(keys[0], (cfg.mixture_k, in_dim))
        centers = cfg.mixture_spread * centers / (1e-6 + jnp.linalg.norm(centers, axis=1, keepdims=True))
        comp = random.randint(keys[1], (n,), 0, cfg.mixture_k)
        eps = random.normal(key, (n, in_dim))
        return centers[comp] + eps
    elif cfg.x_dist == "lowdim_manifold":
        kz, ka = random.split(key)
        Z = random.normal(kz, (n, cfg.x_dim_latent))
        A = random.normal(ka, (cfg.x_dim_latent, in_dim))
        X_low = jnp.tanh(Z @ A)  # non-linear embedding
        return X_low
    elif cfg.x_dist == "heavy_tail":
        # Student-t via scaled normal / sqrt(gamma)
        k1, k2 = random.split(key)
        g = random.gamma(k1, cfg.student_df/2, (n,1)) / (cfg.student_df/2)
        Z = random.normal(k2, (n, in_dim)) / jnp.sqrt(g)
        return Z
    else:
        raise ValueError(f"Unknown x_dist: {cfg.x_dist}")


def build_teacher(key, cfg: Config):
    """Build teacher network (can differ from student)"""
    t_depth = cfg.teacher_depth or cfg.depth
    t_widths = cfg.teacher_widths
    if t_widths is None:
        t_widths = infer_widths(cfg.in_dim, cfg.out_dim, t_depth, cfg.target_params, cfg.hidden)
    t_act = cfg.teacher_activation or cfg.activation

    params = init_mlp_params(key, cfg.in_dim, t_widths, cfg.out_dim, t_act, bias=True, init=cfg.init)

    def forward(X):
        Y = mlp_forward(params, X, t_act, skip=False, residual_period=cfg.residual_period, layernorm=False)
        return Y

    return params, forward


def add_noise(key, y_clean, cfg: Config, X):
    """Add noise according to various models"""
    if cfg.noise_model == "gauss":
        return y_clean + cfg.noise_scale * random.normal(key, y_clean.shape)
    elif cfg.noise_model == "hetero":
        scale = cfg.noise_scale * (1.0 + cfg.hetero_scale * jnp.linalg.norm(X, axis=1, keepdims=True))
        return y_clean + scale * random.normal(key, y_clean.shape)
    elif cfg.noise_model == "student_t":
        # Draw t noise via normal / sqrt(gamma)
        k1, k2 = random.split(key)
        g = random.gamma(k1, cfg.student_df/2, y_clean.shape) / (cfg.student_df/2)
        return y_clean + cfg.noise_scale * random.normal(k2, y_clean.shape) / jnp.sqrt(g)
    elif cfg.noise_model == "outliers":
        k1, k2, k3 = random.split(key, 3)
        base = cfg.noise_scale * random.normal(k1, y_clean.shape)
        mask = random.uniform(k2, y_clean.shape) < cfg.outlier_frac
        outl = cfg.outlier_scale * random.normal(k3, y_clean.shape)
        return y_clean + jnp.where(mask, outl, base)
    else:
        raise ValueError(f"Unknown noise_model: {cfg.noise_model}")


def make_dataset(key, cfg: Config):
    """End-to-end data generation"""
    kx, kt, kn = random.split(key, 3)
    X = sample_X(kx, cfg, cfg.n_data, cfg.in_dim)
    teacher_params, teacher_forward = build_teacher(kt, cfg)
    y_clean = teacher_forward(X)

    if cfg.teacher_dropout_rate > 0.0:
        kd = random.split(kt, 1)[0]
        mask = (random.uniform(kd, y_clean.shape) > cfg.teacher_dropout_rate).astype(y_clean.dtype)
        y_clean = y_clean * mask

    Y = add_noise(kn, y_clean, cfg, X)
    return X, Y, teacher_params, teacher_forward


# ----------------------------
# Loss / likelihood factory
# ----------------------------
def make_loss_fns(unravel, cfg: Config, X, Y):
    """Create loss functions for both full data and minibatch"""
    if cfg.loss == "mse":
        def full(theta):
            params = unravel(theta)
            pred = mlp_forward(params, X, cfg.activation, cfg.skip_connections, cfg.residual_period, cfg.layernorm)
            return jnp.mean((pred - Y)**2)

        def minibatch(theta, Xb, Yb):
            params = unravel(theta)
            pred = mlp_forward(params, Xb, cfg.activation, cfg.skip_connections, cfg.residual_period, cfg.layernorm)
            return jnp.mean((pred - Yb)**2)

    elif cfg.loss == "t_regression":
        s2 = cfg.noise_scale**2
        nu = cfg.student_df

        def neglogt(resid):
            return 0.5*(nu+1)*jnp.log1p((resid**2)/(nu*s2))

        def full(theta):
            params = unravel(theta)
            pred = mlp_forward(params, X, cfg.activation, cfg.skip_connections, cfg.residual_period, cfg.layernorm)
            return jnp.mean(neglogt(pred - Y))

        def minibatch(theta, Xb, Yb):
            params = unravel(theta)
            pred = mlp_forward(params, Xb, cfg.activation, cfg.skip_connections, cfg.residual_period, cfg.layernorm)
            return jnp.mean(neglogt(pred - Yb))
    else:
        raise ValueError(f"Unknown loss: {cfg.loss}")

    return full, minibatch


# ----------------------------
# Sampler factory
# ----------------------------
def make_sampler(cfg: Config, logpost, grad_logpost, theta_init):
    """Create sampler based on config"""
    if cfg.sampler == "sgld":
        def sample_sgld(key, data):
            X, Y = data
            return run_sgld_online(key, theta_init, grad_logpost, X, Y,
                                   cfg.n_data, cfg.sgld_step_size, cfg.sgld_steps, cfg.sgld_warmup,
                                   cfg.sgld_batch_size, cfg.sgld_eval_every, cfg.sgld_thin, 
                                   lambda th: 0.0)  # placeholder for Ln_full64
        return sample_sgld

    elif cfg.sampler == "hmc":
        def sample_hmc(key, data):
            X, Y = data
            return run_hmc_online_with_adaptation(key, theta_init, logpost,
                                                   cfg.hmc_num_integration_steps, cfg.hmc_warmup, cfg.hmc_draws,
                                                   cfg.hmc_thin, cfg.hmc_eval_every,
                                                   lambda th: 0.0)  # placeholder for Ln_full64
        return sample_hmc

    elif cfg.sampler == "nuts":
        # For NUTS we'd use blackjax.nuts, similar to HMC but with automatic step size
        def sample_nuts(key, data):
            X, Y = data
            return run_nuts_online(key, theta_init, logpost,
                                   cfg.hmc_warmup, cfg.hmc_draws, cfg.hmc_thin, cfg.hmc_eval_every,
                                   lambda th: 0.0)  # placeholder for Ln_full64
        return sample_nuts

    else:
        raise ValueError(f"Unknown sampler: {cfg.sampler}")




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
# ----------------------------
# Sampler factory
# ----------------------------
def build_sampler(cfg: Config, logdensity):
    """Factory for different samplers"""
    if cfg.sampler == "hmc":
        return blackjax.hmc, dict(num_integration_steps=cfg.hmc_num_integration_steps)
    elif cfg.sampler == "nuts":
        return blackjax.nuts, {}
    elif cfg.sampler == "sgld":
        return blackjax.sgld, {}
    else:
        raise NotImplementedError(f"Sampler {cfg.sampler} not implemented")


# ----------------------------
# Updated log posterior & score factory
# ----------------------------
def make_logpost_and_score(loss_full, loss_minibatch, theta0, n, beta, gamma):
    """Create log posterior using flexible loss functions"""
    def logpost(theta):
        Ln = loss_full(theta)
        lp = -0.5 * gamma * jnp.sum((theta - theta0) ** 2)
        return lp - n * beta * Ln

    logpost_and_grad = value_and_grad(logpost)

    @jit
    def grad_logpost_minibatch(theta, idx, X, Y):
        Xb, Yb = X[idx], Y[idx]
        g_Lb = grad(lambda th: loss_minibatch(th, Xb, Yb))(theta)
        score = -gamma * (theta - theta0) - beta * n * g_Lb
        return score

    return logpost_and_grad, grad_logpost_minibatch



# ----------------------------
# SGLD chains (BlackJAX) - Online memory-efficient version
# ----------------------------
def run_sgld_online(key, init_thetas, grad_logpost_minibatch, X, Y, n, step_size,
                    num_steps, warmup, batch_size, eval_every, thin,
                    Ln_full64):
    chains = init_thetas.shape[0]
    sgld = blackjax.sgld(grad_logpost_minibatch)
    Ln_running = [RunningMeanVar() for _ in range(chains)]
    Ln_history = [[] for _ in range(chains)]  # track Ln evaluations
    stored = []  # thinned positions for diagnostics (kept small)

    @jit
    def one_step(theta, k, idx):
        return sgld.step(k, theta, idx, X, Y, step_size)

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


# ----------------------------
# NUTS chains (BlackJAX) - Online memory-efficient version
# ----------------------------
def run_nuts_online(key, init_thetas, logpost_and_grad, warmup, draws, thin, eval_every, Ln_full64):
    """NUTS with automatic step size tuning - INCOMPLETE IMPLEMENTATION"""
    # TODO: Fix BlackJAX API usage - current implementation has issues with:
    # 1. wa.run() parameter count (expecting 2 vs 3 arguments) 
    # 2. Return value unpacking from window adaptation
    # 3. Proper NUTS state handling and iteration
    raise NotImplementedError(
        "NUTS sampler implementation needs fixing. "
        "BlackJAX API usage is incorrect. Use 'sgld' or 'hmc' samplers for now."
    )
    
    # Incomplete reference implementation below:
    # chains, dim = init_thetas.shape
    # def logdensity(theta):
    #     val, _ = logpost_and_grad(theta)
    #     return val
    # 
    # def warm_and_sample(key, theta_init):
    #     wa = blackjax.window_adaptation(blackjax.nuts, logdensity)
    #     (state, params), _ = wa.run(key, theta_init)  # API issue here
    #     nuts = blackjax.nuts(logdensity, **params)
    #     # ... rest of sampling logic
    #     return states



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

    X, Y, teacher_params, teacher_forward = make_dataset(key, cfg)

    # Initialize student network parameters
    key, subkey = random.split(key)
    widths = cfg.widths if cfg.widths else [cfg.hidden] * cfg.depth
    w0_pytree = init_mlp_params(subkey, cfg.in_dim, widths, cfg.out_dim, cfg.activation, cfg.bias, cfg.init)

    # Train to empirical minimizer (ERM) - center the local prior there
    print("Training to empirical minimizer...")
    theta_star_f64, unravel_star_f64 = train_erm(w0_pytree, cfg, X.astype(jnp.float64), Y.astype(jnp.float64))
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

    # ===== SGLD (Online) =====
    print("\n=== SGLD (BlackJAX, online) ===")
    k_sgld = random.split(key, 1)[0]
    # simple overdispersed inits around w0
    init_thetas_sgld = theta0_f32 + 0.01 * random.normal(k_sgld, (cfg.chains, dim)).astype(jnp.float32)

    sgld_samples_thin, sgld_Es, sgld_Vars, sgld_Ns, Ln_histories_sgld = run_sgld_online(
        k_sgld,
        init_thetas_sgld,
        grad_logpost_minibatch_f32,
        X_f32, Y_f32,
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
        # Access the theta variable from the xarray.Dataset
        print(
            f"  ESS bulk (median over dims): {np.nanmedian(ess_sgld.data_vars['theta'].values):.1f}"
        )
        print(
            f"  R-hat (max over dims):        {np.nanmax(rhat_sgld.data_vars['theta'].values):.3f}"
        )

    # ===== HMC (Online) =====
    print("\n=== HMC (BlackJAX, online) ===")
    k_hmc = random.fold_in(key, 123)
    init_thetas_hmc = theta0_f64 + 0.01 * random.normal(k_hmc, (cfg.chains, dim))

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
        # Access the theta variable from the xarray.Dataset
        print(f"  ESS bulk (median over dims): {np.nanmedian(ess_hmc.data_vars['theta'].values):.1f}")
        print(
            f"  R-hat (max over dims):        {np.nanmax(rhat_hmc.data_vars['theta'].values):.3f}"
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


# ----------------------------
# Experiment runner for parameter sweeps
# ----------------------------
def sweep_space():
    """Define experiment sweep space"""
    return {
        'base': Config(
            in_dim=32, out_dim=1, n_data=5000,
            sgld_steps=2000, sgld_warmup=500,
            hmc_draws=500, hmc_warmup=200
        ),
        'sweeps': [
            # Architecture sweeps
            {'name': 'depth', 'param': 'depth', 'values': [1, 2, 3, 4]},
            {'name': 'width', 'param': 'hidden', 'values': [50, 100, 200, 400]},
            {'name': 'activation', 'param': 'activation', 'values': ['relu', 'tanh', 'gelu']},
            
            # Data sweeps  
            {'name': 'x_dist', 'param': 'x_dist', 'values': ['gauss_iso', 'gauss_aniso', 'mixture', 'lowdim_manifold']},
            {'name': 'noise', 'param': 'noise_model', 'values': ['gauss', 'hetero', 'student_t']},
            
            # Loss sweeps
            {'name': 'loss', 'param': 'loss', 'values': ['mse', 't_regression']},
        ]
    }


def run_sweep(sweep_config, n_seeds=3):
    """Run experiment sweep"""
    import pandas as pd
    from dataclasses import replace
    
    base = sweep_config['base']
    results = []
    
    for sweep in sweep_config['sweeps']:
        name = sweep['name']
        param = sweep['param']
        values = sweep['values']
        
        print(f"\n=== Sweeping {name} ===")
        for val in values:
            print(f"\n{param} = {val}")
            
            llc_sgld_seeds = []
            llc_hmc_seeds = []
            
            for seed in range(n_seeds):
                # Create config with swept parameter
                cfg = replace(base, **{param: val, 'seed': seed})
                
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
                    'sweep': name,
                    'param': param,
                    'value': val,
                    'llc_sgld_mean': np.mean(llc_sgld_seeds),
                    'llc_sgld_std': np.std(llc_sgld_seeds),
                    'llc_hmc_mean': np.mean(llc_hmc_seeds),
                    'llc_hmc_std': np.std(llc_hmc_seeds),
                    'n_seeds': len(llc_sgld_seeds)
                }
                results.append(result)
                print(f"  LLC: SGLD={result['llc_sgld_mean']:.3f}±{result['llc_sgld_std']:.3f}, "
                      f"HMC={result['llc_hmc_mean']:.3f}±{result['llc_hmc_std']:.3f}")
    
    return pd.DataFrame(results)


def run_experiment(cfg: Config, verbose=True):
    """Run single experiment and return LLCs"""
    key = random.PRNGKey(cfg.seed)
    
    # Generate data
    X, Y, _, _ = make_dataset(key, cfg)
    
    # Initialize network
    key, subkey = random.split(key)
    widths = cfg.widths if cfg.widths else [cfg.hidden] * cfg.depth
    w0 = init_mlp_params(subkey, cfg.in_dim, widths, cfg.out_dim, cfg.activation, cfg.bias, cfg.init)
    
    # Train ERM
    theta_star, unravel = train_erm(w0, cfg, X, Y)
    
    # Setup sampling
    beta = cfg.beta0 / jnp.log(cfg.n_data)
    dim = theta_star.size
    gamma = dim / (cfg.prior_radius ** 2) if cfg.prior_radius else cfg.gamma
    
    # Create loss functions
    loss_full, loss_minibatch = make_loss_fns(unravel, cfg, X, Y)
    L0 = float(loss_full(theta_star))
    
    # Create log posterior
    logpost_grad, grad_minibatch = make_logpost_and_score(
        loss_full, loss_minibatch, theta_star, cfg.n_data, beta, gamma
    )
    
    # Run SGLD
    key, k_sgld = random.split(key)
    init_sgld = theta_star + 0.01 * random.normal(k_sgld, (cfg.chains, dim))
    
    _, _, _, _, Ln_hist_sgld = run_sgld_online(
        k_sgld, init_sgld, grad_minibatch, X, Y,
        cfg.n_data, cfg.sgld_step_size, cfg.sgld_steps, cfg.sgld_warmup,
        cfg.sgld_batch_size, cfg.sgld_eval_every, cfg.sgld_thin,
        jit(loss_full)
    )
    
    llc_sgld, _ = llc_ci_from_histories(Ln_hist_sgld, cfg.n_data, beta, L0)
    
    # Run HMC  
    key, k_hmc = random.split(key)
    init_hmc = theta_star + 0.01 * random.normal(k_hmc, (cfg.chains, dim))
    
    _, _, _, _, _, Ln_hist_hmc = run_hmc_online_with_adaptation(
        k_hmc, init_hmc, logpost_grad,
        cfg.hmc_num_integration_steps, cfg.hmc_warmup, cfg.hmc_draws,
        cfg.hmc_thin, cfg.hmc_eval_every,
        jit(loss_full)
    )
    
    llc_hmc, _ = llc_ci_from_histories(Ln_hist_hmc, cfg.n_data, beta, L0)
    
    if verbose:
        print(f"LLC: SGLD={llc_sgld:.4f}, HMC={llc_hmc:.4f}")
    
    return llc_sgld, llc_hmc


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        print("Running parameter sweep...")
        results_df = run_sweep(sweep_space(), n_seeds=2)
        print("\n=== Sweep Results ===")
        print(results_df.to_string())
        results_df.to_csv("llc_sweep_results.csv", index=False)
        print("\nResults saved to llc_sweep_results.csv")
    else:
        # Use CFG for full run, TEST_CFG for quick testing
        main(CFG)  # Change to TEST_CFG for quick tests
