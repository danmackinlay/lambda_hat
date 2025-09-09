# main.py
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")  # HMC benefits from float64
os.environ.setdefault("MPLBACKEND", "Agg")  # Headless rendering - no GUI windows

import time
from dataclasses import dataclass, replace
from typing import Literal, Optional
from datetime import datetime

import arviz as az
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad, grad
from jax.flatten_util import ravel_pytree
from scipy.stats import norm
import scipy.stats as st

import blackjax
import numpy as np
import optax
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # Ensure headless backend even if pyplot was already imported


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # Model architecture
    in_dim: int = 32
    out_dim: int = 1
    depth: int = 1  # number of hidden layers
    widths: Optional[list[int]] = None  # per-layer widths; if None, auto-infer
    activation: Literal["relu", "tanh", "gelu", "identity"] = "relu"
    bias: bool = True
    skip_connections: bool = False
    residual_period: int = 2  # every k layers add skip if enabled
    layernorm: bool = False  # (default False; can destabilize HMC)
    init: Literal["he", "xavier", "lecun", "orthogonal"] = "he"

    # Size control
    target_params: Optional[int] = 10_000  # if provided, fix total d and infer widths
    # keep old 'hidden' for backward compatibility
    hidden: int = 300  # used only if target_params=None and widths=None

    # Data
    n_data: int = 20_000
    x_dist: Literal[
        "gauss_iso", "gauss_aniso", "mixture", "lowdim_manifold", "heavy_tail"
    ] = "gauss_iso"
    cov_decay: float = 0.95  # for anisotropy: eigvals ~ cov_decay**i
    mixture_k: int = 4
    mixture_spread: float = 2.0
    x_dim_latent: int = 2  # for low-dim manifold
    noise_model: Literal["gauss", "hetero", "student_t", "outliers"] = "gauss"
    noise_scale: float = 0.1
    hetero_scale: float = 0.1
    student_df: float = 4.0
    outlier_frac: float = 0.05
    outlier_scale: float = 2.0

    # Teacher (can differ from student)
    teacher_depth: Optional[int] = None
    teacher_widths: Optional[list[int]] = None
    teacher_activation: Optional[str] = None
    teacher_dropout_rate: float = 0.0  # stochastic teacher if >0 (only during data gen)

    # Loss / likelihood
    loss: Literal["mse", "t_regression"] = "mse"

    # Local posterior (tempering + prior)
    beta_mode: Literal["1_over_log_n", "fixed"] = "1_over_log_n"
    beta0: float = 1.0
    prior_radius: Optional[float] = None  # if set, gamma = d / prior_radius**2
    gamma: float = 1.0  # used only if prior_radius None

    # Sampling
    sampler: Literal["sgld", "hmc", "mclmc", "nuts"] = "sgld"
    chains: int = 4

    # SGLD
    sgld_steps: int = 4_000
    sgld_warmup: int = 1_000
    sgld_batch_size: int = 256
    sgld_step_size: float = 1e-6
    sgld_thin: int = 20  # store every k-th draw for diagnostics only
    sgld_eval_every: int = 10  # compute full-data L_n(w) every k steps (for LLC mean)
    sgld_dtype: str = "float32"  # reduce memory

    # HMC
    hmc_draws: int = 1_000
    hmc_warmup: int = 1_000
    hmc_num_integration_steps: int = 10
    hmc_thin: int = 5  # store every k-th draw for diagnostics
    hmc_eval_every: int = 1  # compute L_n(w) every k draws (usually 1 for HMC)
    hmc_dtype: str = "float64"

    # MCLMC (unadjusted)
    mclmc_draws: int = 2_000  # post-tuning steps (MCLMC yields 1 sample per step)
    mclmc_eval_every: int = 1
    mclmc_thin: int = 10
    mclmc_dtype: str = "float64"  # keep f64 for stability (like HMC)

    # MCLMC tuning
    mclmc_tune_steps: int = 2_000  # steps used by the automatic tuner
    mclmc_diagonal_preconditioning: bool = False
    mclmc_desired_energy_var: float = 5e-4  # target EEV (per Sampling Book)
    mclmc_integrator: Literal[
        "isokinetic_mclachlan",
        "isokinetic_velocity_verlet",
        "isokinetic_yoshida",
        "isokinetic_omelyan",
    ] = "isokinetic_mclachlan"

    # (optional) adjusted MCLMC
    mclmc_adjusted: bool = False
    mclmc_adjusted_target_accept: float = 0.90  # per docs' guidance
    mclmc_grad_per_step_override: Optional[float] = None  # work accounting calibration

    # NUTS
    nuts_draws: int = 1_000
    nuts_warmup: int = 1_000
    nuts_thin: int = 5
    nuts_eval_every: int = 1
    nuts_dtype: str = "float64"

    # Misc
    seed: int = 42
    use_tqdm: bool = True
    progress_update_every: int = 50  # step/draw interval for bar postfix refresh
    profile_adaptation: bool = True  # time warmup/adaptation separately

    # Diagnostics and plotting
    diag_mode: Literal["none", "subset", "proj"] = (
        "proj"  # default: tiny random projections
    )
    diag_k: int = 16  # number of dimensions/projections to track
    diag_seed: int = 1234  # seed for dimension selection/projections
    max_theta_plot_dims: int = 8  # cap for plotting even if k is larger
    save_plots_prefix: Optional[str] = None  # e.g., "diag" to save PNGs

    # Artifacts and visualization saving
    artifacts_dir: str = "artifacts"  # base directory for saving artifacts
    save_plots: bool = False  # whether to save all diagnostic plots
    show_plots: bool = False  # whether to display plots (default: headless)
    auto_create_run_dir: bool = True  # create timestamped run directories
    save_manifest: bool = True  # save run configuration to manifest.txt
    save_readme_snippet: bool = True  # generate README_snippet.md
    auto_update_readme: bool = False  # auto-update README with markers (optional)


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
    # Minimal MCLMC
    mclmc_draws=80,
    mclmc_tune_steps=100,
    mclmc_thin=8,
    # Enable headless plot saving for testing
    save_plots=True,
    show_plots=False,
    save_manifest=True,
    save_readme_snippet=True,
)

CFG = Config()  # Default full config


# ----------------------------
# Timing and work counters
# ----------------------------
@dataclass
class RunStats:
    # wall-clock
    t_build: float = 0.0
    t_train: float = 0.0
    t_sgld_warmup: float = 0.0
    t_sgld_sampling: float = 0.0
    t_hmc_warmup: float = 0.0
    t_hmc_sampling: float = 0.0
    t_mclmc_warmup: float = 0.0
    t_mclmc_sampling: float = 0.0

    # work counters (proxy for computational work)
    # Count "gradient-equivalent" evaluations to compare samplers.
    # - SGLD: ~1 minibatch gradient per step -> +1
    # - HMC: ~num_integration_steps gradients per draw (leapfrog) -> +L
    # - Add full-data loss evals and log-prob grads as separate counters for transparency.
    n_sgld_minibatch_grads: int = 0
    n_sgld_full_loss: int = 0
    n_hmc_leapfrog_grads: int = 0
    n_hmc_full_loss: int = 0
    n_hmc_warmup_leapfrog_grads: int = 0  # estimated during adaptation
    n_mclmc_steps: int = 0
    n_mclmc_full_loss: int = 0


def tic():
    return time.perf_counter()


def toc(t0):
    return time.perf_counter() - t0


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


# ----------------------------
# Work-Normalized Variance utilities
# ----------------------------
def llc_mean_and_se_from_histories(Ln_histories, n, beta, L0):
    """Compute LLC mean and SE using ESS from ArviZ"""
    valid = [h for h in Ln_histories if len(h) > 1]
    if not valid:
        return np.nan, np.nan, np.nan
    m = min(len(h) for h in valid)
    H = np.stack([np.asarray(h[:m]) for h in valid], axis=0)  # (chains, m)
    idata = az.from_dict(posterior={"L": H})
    ess = float(np.nanmedian(az.ess(idata, var_names=["L"]).data_vars["L"].values))
    L_mean = float(np.nanmean(H))
    L_std = float(np.nanstd(H, ddof=1))
    lam_hat = n * float(beta) * (L_mean - L0)
    se = n * float(beta) * (L_std / np.sqrt(max(1.0, ess)))
    return lam_hat, se, ess


def work_normalized_variance(se, time_seconds: float, grad_work: int):
    """Compute WNV in both time and gradient units"""
    return dict(
        WNV_seconds=float(se**2 * max(1e-12, time_seconds)),
        WNV_grads=float(se**2 * max(1.0, grad_work)),
    )


# ----------------------------
# Helper: Infer hidden size from target params
# ----------------------------
def infer_widths(
    in_dim: int,
    out_dim: int,
    depth: int,
    target_params: Optional[int],
    fallback_width: int = 128,
) -> list[int]:
    """Infer widths to hit target_params, or use fallback"""
    if target_params is None:
        return [fallback_width] * depth
    # For simplicity, use constant width h and solve approximately:
    # P(h) = (in_dim+1)h + (L-1)(h+1)h + (h+1)out_dim ≈ target_params
    L = depth
    a = L - 1  # coefficient of h^2
    b = (in_dim + 1) + (L - 1) + out_dim + 1  # coefficient of h
    if a == 0:
        h = max(1, (target_params - out_dim) // (in_dim + 1))
    else:
        disc = b * b + 4 * a * target_params
        h = int((-b + jnp.sqrt(disc)) / (2 * a))
        h = int(max(1, h))
    return [h] * L


def compute_beta_gamma(cfg: Config, d: int) -> tuple[float, float]:
    """Compute beta and gamma from config and dimension"""
    beta = (
        cfg.beta0 / jnp.log(cfg.n_data)
        if cfg.beta_mode == "1_over_log_n"
        else cfg.beta0
    )
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
    opt = optax.adam(lr)
    opt_state = opt.init(theta)

    @jit
    def step(theta, opt_state):
        loss, g = value_and_grad(
            lambda th: jnp.mean((mlp_forward(unravel(th), X) - Y) ** 2)
        )(theta)
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
        "identity": (lambda x: x),
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
        # Use He scaling instead of true orthogonal for robustness
        scale = jnp.sqrt(2.0 / fan_in)
        return random.normal(key, shape) * scale
    else:
        scale = 1.0
    return random.normal(key, shape) * scale


def init_mlp_params(
    key,
    in_dim: int,
    widths: list[int],
    out_dim: int,
    activation: str,
    bias: bool,
    init: str,
):
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

    return {"layers": layers, "out": out_layer}


def mlp_forward(
    params,
    x,
    activation: str = "relu",
    skip: bool = False,
    residual_period: int = 2,
    layernorm: bool = False,
):
    """Forward pass with optional residuals and layer norm"""
    act = act_fn(activation)

    h = x
    for i, lyr in enumerate(params["layers"]):
        z = h @ lyr["W"].T + (lyr["b"] if lyr["b"] is not None else 0.0)

        if layernorm:
            mu = jnp.mean(z, axis=-1, keepdims=True)
            sig = jnp.std(z, axis=-1, keepdims=True) + 1e-6
            z = (z - mu) / sig

        h_new = act(z)

        if skip and (i % residual_period == residual_period - 1):
            # Project if dimensions differ
            if h.shape[-1] != h_new.shape[-1]:
                P = jnp.eye(h_new.shape[-1], h.shape[-1])[
                    : h_new.shape[-1], : h.shape[-1]
                ]
                h = h @ P.T
            h = h + h_new
        else:
            h = h_new

    # Output layer
    y = h @ params["out"]["W"].T + (
        params["out"]["b"] if params["out"]["b"] is not None else 0.0
    )
    return y


def count_params(params):
    """Count total parameters in MLP"""
    leaves = []
    for lyr in params["layers"]:
        leaves.append(lyr["W"])
        if lyr["b"] is not None:
            leaves.append(lyr["b"])
    leaves.append(params["out"]["W"])
    if params["out"]["b"] is not None:
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
        centers = (
            cfg.mixture_spread
            * centers
            / (1e-6 + jnp.linalg.norm(centers, axis=1, keepdims=True))
        )
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
        g = random.gamma(k1, cfg.student_df / 2, (n, 1)) / (cfg.student_df / 2)
        Z = random.normal(k2, (n, in_dim)) / jnp.sqrt(g)
        return Z
    else:
        raise ValueError(f"Unknown x_dist: {cfg.x_dist}")


def build_teacher(key, cfg: Config):
    """Build teacher network (can differ from student)"""
    t_depth = cfg.teacher_depth or cfg.depth
    t_widths = cfg.teacher_widths
    if t_widths is None:
        t_widths = infer_widths(
            cfg.in_dim, cfg.out_dim, t_depth, cfg.target_params, cfg.hidden
        )
    t_act = cfg.teacher_activation or cfg.activation

    params = init_mlp_params(
        key, cfg.in_dim, t_widths, cfg.out_dim, t_act, bias=True, init=cfg.init
    )

    def forward(X):
        Y = mlp_forward(
            params,
            X,
            t_act,
            skip=False,
            residual_period=cfg.residual_period,
            layernorm=False,
        )
        return Y

    return params, forward


def add_noise(key, y_clean, cfg: Config, X):
    """Add noise according to various models"""
    if cfg.noise_model == "gauss":
        return y_clean + cfg.noise_scale * random.normal(key, y_clean.shape)
    elif cfg.noise_model == "hetero":
        scale = cfg.noise_scale * (
            1.0 + cfg.hetero_scale * jnp.linalg.norm(X, axis=1, keepdims=True)
        )
        return y_clean + scale * random.normal(key, y_clean.shape)
    elif cfg.noise_model == "student_t":
        # Draw t noise via normal / sqrt(gamma)
        k1, k2 = random.split(key)
        g = random.gamma(k1, cfg.student_df / 2, y_clean.shape) / (cfg.student_df / 2)
        return y_clean + cfg.noise_scale * random.normal(k2, y_clean.shape) / jnp.sqrt(
            g
        )
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
        mask = (random.uniform(kd, y_clean.shape) > cfg.teacher_dropout_rate).astype(
            y_clean.dtype
        )
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
            pred = mlp_forward(
                params,
                X,
                cfg.activation,
                cfg.skip_connections,
                cfg.residual_period,
                cfg.layernorm,
            )
            return jnp.mean((pred - Y) ** 2)

        def minibatch(theta, Xb, Yb):
            params = unravel(theta)
            pred = mlp_forward(
                params,
                Xb,
                cfg.activation,
                cfg.skip_connections,
                cfg.residual_period,
                cfg.layernorm,
            )
            return jnp.mean((pred - Yb) ** 2)

    elif cfg.loss == "t_regression":
        s2 = cfg.noise_scale**2
        nu = cfg.student_df

        def neglogt(resid):
            return 0.5 * (nu + 1) * jnp.log1p((resid**2) / (nu * s2))

        def full(theta):
            params = unravel(theta)
            pred = mlp_forward(
                params,
                X,
                cfg.activation,
                cfg.skip_connections,
                cfg.residual_period,
                cfg.layernorm,
            )
            return jnp.mean(neglogt(pred - Y))

        def minibatch(theta, Xb, Yb):
            params = unravel(theta)
            pred = mlp_forward(
                params,
                Xb,
                cfg.activation,
                cfg.skip_connections,
                cfg.residual_period,
                cfg.layernorm,
            )
            return jnp.mean(neglogt(pred - Yb))
    else:
        raise ValueError(f"Unknown loss: {cfg.loss}")

    return full, minibatch


# ----------------------------
# Sampler factory
# ----------------------------
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
    z = norm.ppf(1 - alpha / 2)
    llc_mean = n * float(beta) * (L_mean - L0)
    return llc_mean, (
        llc_mean - z * n * float(beta) * se,
        llc_mean + z * n * float(beta) * se,
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


def select_diag_dims(dim, k, seed):
    """Select k random dimensions from d for subset diagnostics"""
    k = min(k, dim)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dim, size=k, replace=False)).astype(int)


def make_projection_matrix(dim, k, seed):
    """Create k random unit vectors for projection diagnostics"""
    k = min(k, dim)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((k, dim)).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-8
    return R  # (k, d)


def prepare_diag_targets(dim, cfg):
    """Prepare diagnostic targets based on config"""
    if cfg.diag_mode == "subset":
        return dict(diag_dims=select_diag_dims(dim, cfg.diag_k, cfg.diag_seed))
    elif cfg.diag_mode == "proj":
        return dict(Rproj=make_projection_matrix(dim, cfg.diag_k, cfg.diag_seed))
    return {}  # none


# ----------------------------
# Log posterior & score (tempered + localized)
# log pi(w) = - n * beta * L_n(w) - (gamma/2)||w - w0||^2
# ----------------------------
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
    def grad_logpost_minibatch(theta, minibatch):  # <- accepts (Xb, Yb)
        Xb, Yb = minibatch
        g_Lb = grad(lambda th: loss_minibatch(th, Xb, Yb))(theta)
        return -gamma * (theta - theta0) - beta * n * g_Lb

    return logpost_and_grad, grad_logpost_minibatch


def make_logdensity_for_mclmc(loss_full64, theta0_f64, n, beta, gamma):
    """Create log-density function for MCLMC sampler (f64 precision)

    MCLMC expects a pure log-density function, not a gradient closure.
    This extracts the log posterior: log π(θ) = -nβ L_n(θ) - γ/2 ||θ-θ₀||²
    """

    @jit
    def logdensity(theta64):
        Ln = loss_full64(theta64)  # already f64 X,Y inside loss_full64
        lp = -0.5 * gamma * jnp.sum((theta64 - theta0_f64) ** 2)
        return lp - n * beta * Ln

    return logdensity


# ----------------------------
# SGLD chains (BlackJAX) - Online memory-efficient version
# ----------------------------
def run_sgld_online(
    key,
    init_thetas,
    grad_logpost_minibatch,
    X,
    Y,
    n,
    step_size,
    num_steps,
    warmup,
    batch_size,
    eval_every,
    thin,
    Ln_full64,
    use_tqdm=True,
    progress_update_every=50,
    stats: RunStats | None = None,
    diag_dims=None,
    Rproj=None,
):
    chains = init_thetas.shape[0]

    # SGLD (BlackJAX 1.2.5): Use top-level factory: sgld = blackjax.sgld(grad_estimator)
    # .step returns *only* the new position (ArrayTree), not (position, info).
    # Source (1.2.5): blackjax/sgmcmc/sgld.py -> as_top_level_api().step_fn returns kernel()
    # which returns new_position (single value).
    # API: https://blackjax-devs.github.io/blackjax/autoapi/blackjax/sgmcmc/sgld/index.html
    # Some docs/examples show (new_position, info) unpacking - that's a doc bug for 1.2.5.
    sgld = blackjax.sgld(
        grad_logpost_minibatch
    )  # Top-level factory (no need for submodule)
    sgld_step = jax.jit(sgld.step)

    Ln_running = [RunningMeanVar() for _ in range(chains)]
    Ln_history = [[] for _ in range(chains)]  # track Ln evaluations
    stored = []  # thinned positions for diagnostics (kept small)

    def one_step(theta, k, idx):
        # Create minibatch from indices
        Xb, Yb = X[idx], Y[idx]
        # SGLD step in 1.2.5 returns single position, not tuple
        new_theta = sgld_step(k, theta, (Xb, Yb), step_size)
        return new_theta

    # Use fold_in for deterministic per-chain RNG streams
    keys = [random.fold_in(key, c) for c in range(chains)]

    # Per-chain timing accumulators
    if stats:
        stats_chain_warm = 0.0
        stats_chain_samp = 0.0
        eval_time_accumulator = 0.0

    for c in range(chains):
        k = keys[c]
        subkeys = random.split(k, num_steps)
        theta = init_thetas[c]

        rng_bar = range(num_steps)
        if use_tqdm:
            rng_bar = tqdm(rng_bar, desc=f"SGLD(c{c})", leave=False, total=num_steps)

        # Chain-level timing
        t_chain = tic() if stats else None

        for t in rng_bar:
            # Fix RNG hygiene: split for noise and minibatch sampling
            k_noise, k_batch = random.split(subkeys[t])
            idx = random.randint(k_batch, (batch_size,), 0, n)
            theta = one_step(theta, k_noise, idx)
            if stats:
                stats.n_sgld_minibatch_grads += 1  # one mb-grad per step

            # Mark transition from warmup to sampling timing
            if t == warmup and stats:
                stats_chain_warm += toc(t_chain)
                t_chain = tic()

            if t >= warmup:
                if (t - warmup) % eval_every == 0:
                    # Time LLC evaluation separately from sampling
                    eval_t0 = tic() if stats else None
                    # Always evaluate in float64 for consistency
                    Ln_val = Ln_full64(theta.astype(jnp.float64))
                    Ln = float(jax.device_get(Ln_val))
                    if stats:
                        eval_time_accumulator += toc(eval_t0)
                    Ln_running[c].update(Ln)
                    Ln_history[c].append(Ln)
                    if stats:
                        stats.n_sgld_full_loss += 1

                if (t - warmup) % thin == 0:
                    # Store minimal theta info based on diag_mode
                    vec = np.array(theta)
                    if diag_dims is not None:  # subset
                        to_store = vec[diag_dims]
                    elif Rproj is not None:  # proj
                        to_store = Rproj @ vec
                    else:  # none
                        to_store = None
                    if to_store is not None:
                        stored.append((c, to_store))

            # progress bar updates
            if use_tqdm and (t % progress_update_every == 0 or t == num_steps - 1):
                # Optional: display current E[L] for this chain
                mean_L = (
                    float(Ln_running[c].value()[0])
                    if Ln_running[c].n > 0
                    else float("nan")
                )
                rng_bar.set_postfix_str(f"L̄≈{mean_L:.4f}")

        # End of chain timing
        if stats:
            # Block on actual computed value for accurate timing
            jax.block_until_ready(theta)
            stats_chain_samp += toc(t_chain)
        # end for t
    # end for c

    # Accumulate per-chain timings to global stats
    if stats:
        stats.t_sgld_warmup += stats_chain_warm
        # Subtract LLC evaluation time from sampling time for pure sampling WNV
        stats.t_sgld_sampling += stats_chain_samp - eval_time_accumulator

    # Pack thinned samples - avoid NaN padding
    kept_list = [[] for _ in range(chains)]
    for c, th in stored:
        kept_list[c].append(th)

    # Determine shape for empty arrays based on diagnostic mode
    if diag_dims is not None:
        empty_shape = (0, len(diag_dims))
    elif Rproj is not None:
        empty_shape = (0, Rproj.shape[0])
    else:
        empty_shape = (0, 0)  # No theta storage

    samples_thin = stack_thinned(
        [np.stack(kl, axis=0) if kl else np.empty(empty_shape) for kl in kept_list]
    )

    # Aggregate LLC stats
    means = [rm.value()[0] for rm in Ln_running]
    vars_ = [rm.value()[1] for rm in Ln_running]
    ns = [rm.value()[2] for rm in Ln_running]

    return samples_thin, np.array(means), np.array(vars_), np.array(ns), Ln_history


# ----------------------------
# HMC chains (BlackJAX) - Online memory-efficient version
# ----------------------------
def run_hmc_online_with_adaptation(
    key,
    init_thetas,
    logpost_and_grad,
    num_integration_steps,
    warmup,
    draws,
    thin,
    eval_every,
    Ln_full64,
    use_tqdm=True,
    progress_update_every=50,
    stats: RunStats | None = None,
    diag_dims=None,
    Rproj=None,
):
    chains, dim = init_thetas.shape

    def logdensity(theta):
        val, _ = logpost_and_grad(theta)
        return val

    kept_list, means, vars_, ns, accs, Ln_hist_list = [], [], [], [], [], []
    # Use fold_in for deterministic per-chain RNG streams
    chain_keys = [random.fold_in(key, c) for c in range(chains)]

    for c in range(chains):
        ck = chain_keys[c]
        pbar = (
            tqdm(total=warmup + draws, desc=f"HMC(c{c})", leave=False)
            if use_tqdm
            else None
        )

        t0 = tic()
        # Stan-style window adaptation (BlackJAX 1.2.5):
        # (state, parameters), adapt_info = blackjax.window_adaptation(blackjax.hmc, logdensity,
        #                                       is_mass_matrix_diagonal=True,
        #                                       num_integration_steps=L).run(rng, init_pos, num_steps=warmup)
        # Then build tuned kernel: blackjax.hmc(logdensity, **parameters, num_integration_steps=L).step
        # Quickstart: https://blackjax-devs.github.io/blackjax/examples/quickstart.html
        # API: https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html
        wa = blackjax.window_adaptation(
            blackjax.hmc,
            logdensity,
            is_mass_matrix_diagonal=True,
            num_integration_steps=num_integration_steps,
        )
        (state, params), _ = wa.run(ck, init_thetas[c], num_steps=warmup)
        if stats:
            stats.t_hmc_warmup += toc(t0)
            # Approximation: window adaptation uses ~L+1 gradients per step
            stats.n_hmc_warmup_leapfrog_grads += warmup * (num_integration_steps + 1)
        if pbar:
            pbar.update(warmup)

        invM = params.get("inverse_mass_matrix", jnp.ones(dim))
        if invM.ndim == 2:
            invM = jnp.diag(invM)  # keep diagonal mass
        kernel = blackjax.hmc(
            logdensity,
            **dict(step_size=params["step_size"], inverse_mass_matrix=invM),
            num_integration_steps=num_integration_steps,
        ).step

        # Precompile the kernel to avoid XLA compilation during sampling
        compiled_step = jax.jit(kernel)
        pre_t = tic()
        ck, ck_draw = random.split(ck)  # Decorrelate adaptation and sampling
        draw_keys = random.split(ck_draw, draws)  # Use separate key stream
        state, info = compiled_step(draw_keys[0], state)  # Trigger compilation
        jax.block_until_ready(state.position)
        if stats:
            stats.t_hmc_warmup += toc(pre_t)

        rm, kept, acc_chain, Lhist = RunningMeanVar(), [], [], []
        hmc_eval_time = 0.0
        t1 = tic()

        # Process the first sample (already computed during precompilation)
        if (0 % eval_every) == 0:
            eval_t0 = tic() if stats else None
            Ln_val = Ln_full64(state.position.astype(jnp.float64))
            Ln = float(jax.device_get(Ln_val))
            if stats:
                hmc_eval_time += toc(eval_t0)
            rm.update(Ln)
            Lhist.append(Ln)
            if stats:
                stats.n_hmc_full_loss += 1
        if (0 % thin) == 0:
            vec = np.array(state.position)
            if diag_dims is not None:
                to_store = vec[diag_dims]
            elif Rproj is not None:
                to_store = Rproj @ vec
            else:
                to_store = None
            if to_store is not None:
                kept.append(to_store)
        acc_chain.append(get_accept(info))
        if stats:
            steps = getattr(info, "num_integration_steps", num_integration_steps)
            stats.n_hmc_leapfrog_grads += int(steps + 1)
        if pbar:
            pbar.set_postfix_str(
                f"acc≈{np.mean(acc_chain):.2f}, L̄≈{(rm.value()[0] if rm.n > 0 else float('nan')):.4f}"
            )
            pbar.update(1)

        # Continue with remaining samples
        for t in range(1, draws):
            state, info = compiled_step(draw_keys[t], state)
            if (t % eval_every) == 0:
                # Time LLC evaluation separately from sampling
                eval_t0 = tic() if stats else None
                Ln_val = Ln_full64(state.position.astype(jnp.float64))
                Ln = float(jax.device_get(Ln_val))
                if stats:
                    hmc_eval_time += toc(eval_t0)
                rm.update(Ln)
                Lhist.append(Ln)
                if stats:
                    stats.n_hmc_full_loss += 1
            if (t % thin) == 0:
                # Store minimal theta info based on diag_mode
                vec = np.array(state.position)
                if diag_dims is not None:  # subset
                    to_store = vec[diag_dims]
                elif Rproj is not None:  # proj
                    to_store = Rproj @ vec
                else:  # none
                    to_store = None
                if to_store is not None:
                    kept.append(to_store)
            # HMC (BlackJAX 1.2.5): info is HMCInfo with flat acceptance_rate field
            # Source: https://blackjax-devs.github.io/blackjax/_modules/blackjax/mcmc/hmc.html
            acc_chain.append(get_accept(info))
            if stats:
                # More accurate: velocity-Verlet uses ~L+1 gradient evals
                steps = getattr(info, "num_integration_steps", num_integration_steps)
                stats.n_hmc_leapfrog_grads += int(steps + 1)
            if pbar and (t % progress_update_every == 0 or t == draws - 1):
                pbar.set_postfix_str(
                    f"acc≈{np.mean(acc_chain):.2f}, L̄≈{(rm.value()[0] if rm.n > 0 else float('nan')):.4f}"
                )
                pbar.update(1)

        if stats:
            # Block on actual computed value for accurate timing
            jax.block_until_ready(state.position)
            # Subtract LLC evaluation time from sampling time for pure sampling WNV
            stats.t_hmc_sampling += toc(t1) - hmc_eval_time
        if pbar:
            pbar.close()

        # Determine shape for empty arrays based on diagnostic mode
        if diag_dims is not None:
            empty_shape = (0, len(diag_dims))
        elif Rproj is not None:
            empty_shape = (0, Rproj.shape[0])
        else:
            empty_shape = (0, 0)  # No theta storage
        kept_list.append(np.stack(kept, 0) if kept else np.empty(empty_shape))
        m, v, n = rm.value()
        means.append(m)
        vars_.append(v)
        ns.append(n)
        accs.append(np.array(acc_chain))
        Ln_hist_list.append(np.asarray(Lhist))

    samples_thin = stack_thinned(kept_list)
    return (
        samples_thin,
        np.array(means),
        np.array(vars_),
        np.array(ns),
        accs,
        Ln_hist_list,
    )


# ----------------------------
# MCLMC chains (BlackJAX) - Online memory-efficient version
# ----------------------------
def run_mclmc_online(
    key,
    init_theta,  # (chains, dim) f64
    logdensity_fn,  # jitted f64 fn from make_logdensity_for_mclmc
    draws: int,
    eval_every: int,
    thin: int,
    Ln_full64,  # jitted f64 loss for LLC evaluation
    diag_dims=None,
    Rproj=None,  # tiny θ diagnostics (subset/proj/none)
    tuner_steps: int = 2000,
    diagonal_preconditioning: bool = False,
    desired_energy_var: float = 5e-4,
    integrator_name: str = "isokinetic_mclachlan",
    use_tqdm: bool = True,
    progress_update_every: int = 50,
    stats: RunStats | None = None,
):
    """
    Unadjusted MCLMC: tune (L, step_size) using blackjax.mclmc_find_L_and_step_size,
    then run draws steps, computing L_n online and storing thinned tiny diagnostics.
    API grounded on Sampling Book's MCLMC page (1.2.x):
      - mclmc_find_L_and_step_size(...)
      - blackjax.mclmc(logdensity_fn, L=..., step_size=...).step
      - blackjax.mcmc.integrators.isokinetic_mclachlan
    See: https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html
    """
    # MCLMC (BlackJAX 1.2.x) quick facts:
    # - Two hyperparameters L (momentum decoherence length) and step_size ε.
    # - Use BlackJAX's automatic tuner to find (L, ε) before running.
    # - Unadjusted MCLMC *does not* have accept/reject; adjusted variant exists and
    #   typically targets ~0.9 MH acceptance; prefer unadjusted unless unbiasedness is
    #   a hard requirement (per docs).
    # - Sampling Book "How to run MCLMC" shows the exact sequence used here:
    #   init -> build_kernel/integrator -> mclmc_find_L_and_step_size -> blackjax.mclmc(...)
    #   https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html
    # - Integrators list includes isokinetic_* variants (we default to isokinetic_mclachlan):
    #   https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/integrators/index.html

    chains, dim = init_theta.shape

    # 1) initial state (per chain)
    def init_state(ck, pos):
        return blackjax.mcmc.mclmc.init(
            position=pos, logdensity_fn=logdensity_fn, rng_key=ck
        )

    # 2) kernel factory for tuner (per docs example)
    integrator = getattr(blackjax.mcmc.integrators, integrator_name)

    def build_kernel(inverse_mass_matrix):
        return blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=integrator,
            inverse_mass_matrix=inverse_mass_matrix,
        )

    # outputs
    kept_list, means, vars_, ns, Ln_hist_list, energy_delta_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # per-chain run
    keys = [random.fold_in(key, c) for c in range(chains)]
    for c in range(chains):
        ck = keys[c]
        init_state_c = init_state(ck, init_theta[c])

        # 3) tune L and step_size (per docs)
        tune_k1, tune_k2, run_k = random.split(ck, 3)
        t0 = tic()
        tuned_state, tuned_params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=build_kernel,
            num_steps=tuner_steps,
            state=init_state_c,
            rng_key=tune_k1,
            diagonal_preconditioning=diagonal_preconditioning,
            desired_energy_var=desired_energy_var,
        )
        if stats:
            # tuner cost is part of "warmup" wall time for MCLMC
            stats.t_mclmc_warmup += toc(t0)

        # tuned_params has fields .L and .step_size (per docs)
        alg = blackjax.mclmc(
            logdensity_fn, L=tuned_params.L, step_size=tuned_params.step_size
        )
        step = jax.jit(alg.step)
        state = tuned_state
        kept, Lhist, Ehist = [], [], []
        rm = RunningMeanVar()

        pbar = tqdm(total=draws, desc=f"MCLMC(c{c})", leave=False) if use_tqdm else None
        keys_draw = random.split(run_k, draws)
        t1 = tic()
        for t in range(draws):
            state, info = step(keys_draw[t], state)  # unadjusted: no accept/reject
            if (t % eval_every) == 0:
                Ln = float(Ln_full64(state.position.astype(jnp.float64)))
                rm.update(Ln)
                Lhist.append(Ln)
                if hasattr(
                    info, "energy_change"
                ):  # MCLMCInfo has energy_change in index
                    Ehist.append(float(info.energy_change))
                if stats:
                    stats.n_mclmc_full_loss += 1  # count LLC evals
            if (t % thin) == 0:
                vec = np.array(state.position)
                if diag_dims is not None:
                    to_store = vec[diag_dims]
                elif Rproj is not None:
                    to_store = Rproj @ vec
                else:
                    to_store = None
                if to_store is not None:
                    kept.append(to_store)
            if pbar and (t % progress_update_every == 0 or t == draws - 1):
                pbar.set_postfix_str(f"L̄≈{np.mean(Lhist):.4f}" if Lhist else "")
                pbar.update(1)

        # end chain
        if stats:
            jax.block_until_ready(state.position)
            stats.t_mclmc_sampling += toc(t1)
            stats.n_mclmc_steps += draws

        if pbar:
            pbar.close()

        # Determine shape for empty arrays based on diagnostic mode
        if diag_dims is not None:
            empty_shape = (0, len(diag_dims))
        elif Rproj is not None:
            empty_shape = (0, Rproj.shape[0])
        else:
            empty_shape = (0, 0)  # No theta storage
        kept_list.append(np.stack(kept, 0) if kept else np.empty(empty_shape))
        m, v, n = rm.value()
        means.append(m)
        vars_.append(v)
        ns.append(n)
        energy_delta_list.append(np.asarray(Ehist))
        Ln_hist_list.append(np.asarray(Lhist))

    samples_thin = stack_thinned(kept_list)
    return (
        samples_thin,
        np.array(means),
        np.array(vars_),
        np.array(ns),
        energy_delta_list,
        Ln_hist_list,
    )


# ----------------------------
# Uniform Sampler Registry/Dispatcher
# ----------------------------
def run_sampler(
    cfg,
    name: str,
    *,  # "sgld" | "hmc" | "mclmc"
    init_theta_f32,
    init_theta_f64,
    logpost_and_grad_f32,
    logpost_and_grad_f64,
    grad_minibatch_f32,
    Ln_full64,
    X_f32,
    Y_f32,
    theta0_f64,
    beta,
    gamma,
    stats=None,
):
    """Uniform interface for running any sampler with consistent inputs/outputs"""
    dim = init_theta_f64.shape[1]
    diag_targets = prepare_diag_targets(dim, cfg)

    if name == "sgld":
        return run_sgld_online(
            random.PRNGKey(cfg.seed + 10),
            init_theta_f32,
            grad_minibatch_f32,
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
    elif name == "hmc":
        return run_hmc_online_with_adaptation(
            random.PRNGKey(cfg.seed + 20),
            init_theta_f64,
            logpost_and_grad_f64,
            cfg.hmc_num_integration_steps,
            cfg.hmc_warmup,
            cfg.hmc_draws,
            cfg.hmc_thin,
            cfg.hmc_eval_every,
            Ln_full64,
            use_tqdm=cfg.use_tqdm,
            progress_update_every=cfg.progress_update_every,
            stats=stats,
            **diag_targets,
        )
    elif name == "mclmc":
        # Create logdensity for MCLMC
        logdensity = make_logdensity_for_mclmc(
            Ln_full64, theta0_f64, cfg.n_data, beta, gamma
        )
        return run_mclmc_online(
            random.PRNGKey(cfg.seed + 30),
            init_theta_f64,
            logdensity,
            cfg.mclmc_draws,
            cfg.mclmc_eval_every,
            cfg.mclmc_thin,
            Ln_full64,
            use_tqdm=cfg.use_tqdm,
            progress_update_every=cfg.progress_update_every,
            tuner_steps=cfg.mclmc_tune_steps,
            diagonal_preconditioning=cfg.mclmc_diagonal_preconditioning,
            desired_energy_var=cfg.mclmc_desired_energy_var,
            integrator_name=cfg.mclmc_integrator,
            stats=stats,
            **diag_targets,
        )
    else:
        raise ValueError(f"Unknown sampler: {name}")


# ----------------------------
# NUTS chains (BlackJAX) - Online memory-efficient version
# ----------------------------
def run_nuts_online(
    key, init_thetas, logpost_and_grad, warmup, draws, thin, eval_every, Ln_full64
):
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


# ----------------------------
# Plotting and diagnostics
# ----------------------------


def _idata_from_L(Ln_histories):
    """Create ArviZ InferenceData from L_n histories"""
    H = _stack_histories(Ln_histories)
    return (
        (az.from_dict(posterior={"L": H}), H.shape[1]) if H is not None else (None, 0)
    )


def _idata_from_theta(samples_thin, max_dims=8):
    """Create ArviZ InferenceData from theta samples"""
    S = np.asarray(samples_thin)
    if S.size == 0 or S.shape[1] < 2:
        return None, []
    k = S.shape[-1]
    idx = list(range(min(k, max_dims)))
    idata = az.from_dict(
        posterior={"theta": S},
        coords={"theta_dim": np.arange(k)},
        dims={"theta": ["theta_dim"]},
    )
    return idata, idx


def _running_llc(Ln_histories, n, beta, L0):
    """Compute running LLC estimates"""
    H = _stack_histories(Ln_histories)
    if H is None:
        return None, None
    cmean = np.cumsum(H, 1) / np.arange(1, H.shape[1] + 1)[None, :]
    lam = n * float(beta) * (cmean - L0)
    pooled = (
        n * float(beta) * (np.cumsum(H.mean(0)) / np.arange(1, H.shape[1] + 1) - L0)
    )
    return lam, pooled


# ----------------------------
# Plotting helpers
# ----------------------------
def _finalize_figure(
    cfg: Config, save_prefix: str | None, save_dir: str | None, name: str
):
    """Save and/or show figure, then close it to prevent blocking"""
    saved_paths = []

    # Save via legacy save_prefix (backward compatibility)
    if save_prefix:
        stem = f"{save_prefix}_{name}"
        path = f"{stem}.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        saved_paths.append(path)

    # Save via new save_dir system
    if save_dir:
        saved_paths.extend(save_plot(save_dir, name))

    # Show if requested (usually False for headless)
    if cfg.show_plots:
        plt.show()
    else:
        plt.close()  # Release memory, no GUI

    return saved_paths


def save_plot(save_dir: str, name: str, fmt: str = "png") -> list[str]:
    """Save current matplotlib figure to save_dir with consistent naming"""
    from pathlib import Path

    paths = []
    stem = Path(save_dir) / name

    if fmt in ("png", "both"):
        path = f"{stem}.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        paths.append(path)

    if fmt in ("svg", "both"):
        path = f"{stem}.svg"
        plt.savefig(path, bbox_inches="tight")
        paths.append(path)

    return paths


def write_html_gallery(
    run_dir: str, images: list[str], title: str = "LLC Sampler Diagnostics"
):
    """Generate HTML gallery for viewing all saved plots"""
    from pathlib import Path

    rd = Path(run_dir)
    rels = [os.path.relpath(p, start=rd) for p in images if os.path.exists(p)]

    items = []
    for r in rels:
        items.append(
            f'<div><img src="{r}" style="max-width: 720px; height: auto;"><br><small>{r}</small></div><hr/>'
        )

    html = f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        img {{ border: 1px solid #ddd; margin-bottom: 10px; }}
        small {{ color: #666; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p><strong>Run directory:</strong> {rd}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>Total images:</strong> {len(rels)}</p>
    <hr/>
    {"".join(items)}
</body>
</html>"""

    gallery_path = rd / "index.html"
    gallery_path.write_text(html)
    return str(gallery_path)


def plot_diagnostics(
    sgld_samples_thin,
    Ln_histories_sgld,
    hmc_samples_thin,
    Ln_histories_hmc,
    accs_hmc,
    n,
    beta,
    L0,
    cfg: Config,
    mclmc_samples_thin=None,
    Ln_histories_mclmc=None,
    energy_deltas_mclmc=None,
    max_theta_dims=8,
    save_prefix=None,
    save_dir=None,
):
    """Plot comprehensive convergence diagnostics"""
    saved_files = []

    def save_plot(filename):
        """Helper to save plot with consistent naming and location"""
        if save_prefix or save_dir:
            save_name = filename
            if save_prefix:
                save_name = f"{save_prefix}_{save_name}"
            if save_dir:
                save_name = os.path.join(save_dir, save_name)
            plt.savefig(save_name, dpi=160, bbox_inches="tight")

    # Running LLC plots
    samplers = [("SGLD", Ln_histories_sgld), ("HMC", Ln_histories_hmc)]
    if Ln_histories_mclmc is not None:
        samplers.append(("MCLMC", Ln_histories_mclmc))

    for name, H in samplers:
        lam, pooled = _running_llc(H, n, beta, L0)
        if lam is None:
            continue
        T = lam.shape[1]
        plt.figure(figsize=(7, 4))
        for c in range(lam.shape[0]):
            plt.plot(np.arange(1, T + 1), lam[c], alpha=0.4, lw=1)
        plt.plot(np.arange(1, T + 1), pooled, lw=2, label="pooled")
        plt.xlabel("L_n evaluations")
        plt.ylabel(r"$\hat\lambda_t$")
        plt.title(f"{name}: running LLC")
        plt.legend()
        plt.tight_layout()
        saved_files.extend(
            _finalize_figure(cfg, save_prefix, save_dir, f"{name.lower()}_llc_running")
        )

    # L_n trace, ACF, ESS, Rhat
    samplers = [("SGLD", Ln_histories_sgld), ("HMC", Ln_histories_hmc)]
    if Ln_histories_mclmc is not None:
        samplers.append(("MCLMC", Ln_histories_mclmc))

    for name, H in samplers:
        idata_L, T = _idata_from_L(H)
        if idata_L is None:
            continue

        # Trace plot
        az.plot_trace(idata_L, var_names=["L"])
        plt.suptitle(f"{name}: L_n trace", y=1.02)
        plt.tight_layout()
        saved_files.extend(
            _finalize_figure(cfg, save_prefix, save_dir, f"{name.lower()}_L_trace")
        )

        # Autocorrelation
        az.plot_autocorr(idata_L, var_names=["L"])
        plt.suptitle(f"{name}: L_n ACF", y=1.02)
        plt.tight_layout()
        saved_files.extend(
            _finalize_figure(cfg, save_prefix, save_dir, f"{name.lower()}_L_acf")
        )

        # ESS
        try:
            az.plot_ess(idata_L, var_names=["L"])
            plt.suptitle(f"{name}: ESS(L_n)", y=1.02)
            plt.tight_layout()
            saved_files.extend(
                _finalize_figure(cfg, save_prefix, save_dir, f"{name.lower()}_L_ess")
            )
        except:
            pass

        # R-hat
        try:
            az.plot_forest(idata_L, var_names=["L"], r_hat=True)
            plt.suptitle(f"{name}: R̂(L_n)", y=1.02)
            plt.tight_layout()
            saved_files.extend(
                _finalize_figure(cfg, save_prefix, save_dir, f"{name.lower()}_L_rhat")
            )
        except:
            pass

    # Tiny theta diagnostics (only if we stored subset/proj)
    samplers = [("SGLD", sgld_samples_thin), ("HMC", hmc_samples_thin)]
    if mclmc_samples_thin is not None:
        samplers.append(("MCLMC", mclmc_samples_thin))

    for name, S in samplers:
        idata_th, idx = _idata_from_theta(S, max_dims=max_theta_dims)
        if idata_th is None:
            continue

        coords = {"theta_dim": idx}

        # Theta trace
        try:
            az.plot_trace(idata_th, var_names=["theta"], coords=coords)
            plt.suptitle(f"{name}: θ trace (k={len(idx)})", y=1.02)
            plt.tight_layout()
            saved_files.extend(
                _finalize_figure(
                    cfg, save_prefix, save_dir, f"{name.lower()}_theta_trace"
                )
            )
        except:
            pass

        # Rank plot
        try:
            az.plot_rank(idata_th, var_names=["theta"], coords=coords)
            plt.suptitle(f"{name}: θ rank (k={len(idx)})", y=1.02)
            plt.tight_layout()
            saved_files.extend(
                _finalize_figure(
                    cfg, save_prefix, save_dir, f"{name.lower()}_theta_rank"
                )
            )
        except:
            pass

    # HMC acceptance histogram
    if accs_hmc:
        acc = np.concatenate([np.asarray(a).ravel() for a in accs_hmc if len(a)])
        if acc.size:
            plt.figure(figsize=(6, 4))
            plt.hist(acc, bins=20, density=True)
            plt.xlabel("acceptance_rate")
            plt.ylabel("density")
            plt.title("HMC acceptance")
            plt.tight_layout()
            saved_files.extend(
                _finalize_figure(cfg, save_prefix, save_dir, "hmc_acceptance")
            )

    # MCLMC energy change histogram
    if energy_deltas_mclmc is not None:
        try:
            e = np.concatenate(
                [np.asarray(eh) for eh in energy_deltas_mclmc if len(eh)]
            )
            if e.size:
                plt.figure(figsize=(6, 4))
                plt.hist(e, bins=20, density=True)
                plt.xlabel("ΔH")
                plt.ylabel("density")
                plt.title("MCLMC energy change histogram")
                plt.tight_layout()
                saved_files.extend(
                    _finalize_figure(cfg, save_prefix, save_dir, "mclmc_energy_hist")
                )
        except Exception:
            pass

    return saved_files


# ----------------------------
# Data Artifact Saving (Analysis-Ready Format)
# ----------------------------
def save_idata_L(run_dir: str, name: str, Ln_histories) -> str | None:
    """Save L_n histories as ArviZ InferenceData NetCDF for dynamic plotting"""
    if not Ln_histories:
        return None

    from pathlib import Path

    H = _stack_histories(Ln_histories)
    if H is None:
        return None

    # Create ArviZ InferenceData from L_n histories
    idata = az.from_dict(posterior={"L": H})
    path = Path(run_dir) / f"{name}_L.nc"
    idata.to_netcdf(str(path))
    return str(path)


def save_theta_thin(run_dir: str, name: str, samples_thin) -> str | None:
    """Save thinned theta samples as compressed npz"""
    from pathlib import Path

    # samples_thin: list of arrays, convert to (chains, draws, k)
    S = np.array(samples_thin) if samples_thin is not None else np.array([])
    if S.size == 0:
        return None

    path = Path(run_dir) / f"{name}_theta_thin.npz"
    np.savez_compressed(str(path), samples=S)
    return str(path)


def save_metrics(run_dir: str, name: str, metrics: dict) -> str:
    """Save sampler metrics as JSON"""
    from pathlib import Path
    import json

    path = Path(run_dir) / f"{name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=str))
    return str(path)


def save_config(run_dir: str, cfg: Config) -> str:
    """Save configuration as JSON"""
    from pathlib import Path
    import json
    from dataclasses import asdict

    path = Path(run_dir) / "config.json"
    path.write_text(json.dumps(asdict(cfg), indent=2, default=str))
    return str(path)


def save_L0(run_dir: str, L0: float) -> str:
    """Save L0 value for running LLC reconstruction"""
    from pathlib import Path

    path = Path(run_dir) / "L0.txt"
    path.write_text(f"{L0:.10f}")
    return str(path)


def _stack_histories(Ln_histories) -> np.ndarray | None:
    """Convert list of L_n histories to (chains, draws) array for ArviZ"""
    if not Ln_histories:
        return None

    # Filter out empty histories
    valid_histories = [h for h in Ln_histories if len(h) > 0]
    if not valid_histories:
        return None

    # Stack into (chains, draws) format
    try:
        H = np.array(valid_histories)  # (chains, draws)
        return H
    except ValueError:
        # Handle different length histories by padding to max length
        max_len = max(len(h) for h in valid_histories)
        H = np.full((len(valid_histories), max_len), np.nan)
        for i, h in enumerate(valid_histories):
            H[i, : len(h)] = h
        return H


# ----------------------------
# Artifact Management
# ----------------------------
def create_run_directory(cfg: Config) -> str:
    """Create timestamped run directory for artifacts"""
    if not cfg.save_plots and not cfg.save_manifest and not cfg.save_readme_snippet:
        return ""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg.artifacts_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def save_run_manifest(
    run_dir: str, cfg: Config, stats: Optional[object] = None
) -> None:
    """Save run configuration and statistics to manifest.txt"""
    if not run_dir or not cfg.save_manifest:
        return

    manifest_path = os.path.join(run_dir, "manifest.txt")

    # Convert config to dict for serialization
    config_dict = {
        field.name: getattr(cfg, field.name)
        for field in cfg.__dataclass_fields__.values()
    }

    with open(manifest_path, "w") as f:
        f.write("# LLC Run Configuration and Results\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")

        f.write("## Configuration\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")

        if stats:
            f.write("\n## Runtime Statistics\n")
            if hasattr(stats, "__dict__"):
                for key, value in stats.__dict__.items():
                    f.write(f"{key}: {value}\n")


def save_readme_snippet(run_dir: str, cfg: Config, run_name: str = "") -> None:
    """Generate README_snippet.md for easy documentation"""
    if not run_dir or not cfg.save_readme_snippet:
        return

    snippet_path = os.path.join(run_dir, "README_snippet.md")
    timestamp = os.path.basename(run_dir)

    with open(snippet_path, "w") as f:
        f.write(f"### Run {run_name or timestamp}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("**Key Configuration:**\n")
        f.write(f"- Model: {cfg.depth}-layer MLP, {cfg.activation} activation\n")
        f.write(f"- Parameters: ~{cfg.target_params:,} target\n")
        f.write(f"- Data: n={cfg.n_data:,}, {cfg.x_dist} distribution\n")
        f.write(f"- Samplers: {cfg.sampler}\n")
        f.write(f"- Chains: {cfg.chains}\n\n")

        f.write("**Diagnostic Plots:**\n")
        # List plot files that should be generated
        plot_types = [
            "llc_running",
            "L_trace",
            "L_acf",
            "L_ess",
            "L_rhat",
            "theta_trace",
            "theta_rank",
            "acceptance",
            "energy_hist",
        ]
        samplers = ["sgld", "hmc", "mclmc"]

        for sampler in samplers:
            for plot_type in plot_types:
                if plot_type in ["acceptance"] and sampler != "hmc":
                    continue
                if plot_type in ["energy_hist"] and sampler != "mclmc":
                    continue
                f.write(f"- `{sampler}_{plot_type}.png`\n")

        f.write(f"\n**Artifacts Location:** `{run_dir}/`\n\n")


def update_readme_with_run(run_dir: str, cfg: Config) -> None:
    """Optionally update main README with run information using markers"""
    if not cfg.auto_update_readme:
        return

    readme_path = "README.md"
    if not os.path.exists(readme_path):
        return

    # Implementation for auto-updating README would go here
    # This is optional and more complex, so leaving as placeholder
    pass


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
            cfg.hmc_num_integration_steps,
            cfg.hmc_warmup,
            cfg.hmc_draws,
            cfg.hmc_thin,
            cfg.hmc_eval_every,
            Ln_full64,
            use_tqdm=cfg.use_tqdm,
            progress_update_every=cfg.progress_update_every,
            stats=stats,
            **diag_targets,
        )
    )

    # Compute LLC with proper CI using ESS
    llc_hmc, ci_hmc = llc_ci_from_histories(Ln_histories_hmc, cfg.n_data, beta, L0)
    mean_acc = float(np.mean([a.mean() for a in accs_hmc]))

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
    jax.block_until_ready(hmc_samples_thin)  # Sync before total runtime measurement
    print(f"Total Runtime:    {time.time() - t0:.2f}")

    print("\n=== Work Summary ===")
    print(f"SGLD - Minibatch grads: {stats.n_sgld_minibatch_grads}")
    print(f"SGLD - Full loss evals: {stats.n_sgld_full_loss}")
    print(f"HMC - Leapfrog grads:   {stats.n_hmc_leapfrog_grads}")
    print(f"HMC - Full loss evals:  {stats.n_hmc_full_loss}")
    print(f"HMC - Warmup grads:     {stats.n_hmc_warmup_leapfrog_grads}")
    print(f"MCLMC - Steps:          {stats.n_mclmc_steps}")
    print(f"MCLMC - Full loss evals: {stats.n_mclmc_full_loss}")

    # Plot diagnostics if enabled
    saved_files = []
    if cfg.diag_mode != "none":
        print("\n=== Generating Diagnostic Plots ===")
        saved_files = plot_diagnostics(
            sgld_samples_thin,
            Ln_histories_sgld,
            hmc_samples_thin,
            Ln_histories_hmc,
            accs_hmc,
            cfg.n_data,
            beta,
            L0,
            cfg,
            mclmc_samples_thin,
            Ln_histories_mclmc,
            energy_deltas_mclmc,
            max_theta_dims=cfg.max_theta_plot_dims,
            save_prefix=cfg.save_plots_prefix,
            save_dir=run_dir if cfg.save_plots else None,
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

        # Save thinned theta samples
        save_theta_thin(run_dir, "sgld", sgld_samples_thin)
        save_theta_thin(run_dir, "hmc", hmc_samples_thin)
        save_theta_thin(run_dir, "mclmc", mclmc_samples_thin)

        # Save metrics for each sampler
        save_metrics(
            run_dir,
            "sgld",
            {
                "llc_mean": float(llc_sgld_mean),
                "llc_se": float(se_sgld),
                "ess": float(ess_sgld),
                "timing_warmup": float(stats.t_sgld_warmup),
                "timing_sampling": float(stats.t_sgld_sampling),
                "n_steps": int(stats.n_sgld_minibatch_grads),
                "n_full_loss": int(stats.n_sgld_full_loss),
            },
        )

        save_metrics(
            run_dir,
            "hmc",
            {
                "llc_mean": float(llc_hmc_mean),
                "llc_se": float(se_hmc),
                "ess": float(ess_hmc),
                "timing_warmup": float(stats.t_hmc_warmup),
                "timing_sampling": float(stats.t_hmc_sampling),
                "n_leapfrog_grads": int(stats.n_hmc_leapfrog_grads),
                "n_full_loss": int(stats.n_hmc_full_loss),
                "mean_acceptance": float(
                    np.mean([np.mean(a) for a in accs_hmc if len(a)])
                )
                if accs_hmc
                else 0.0,
            },
        )

        save_metrics(
            run_dir,
            "mclmc",
            {
                "llc_mean": float(llc_mclmc_mean),
                "llc_se": float(se_mclmc),
                "ess": float(ess_mclmc),
                "timing_warmup": float(stats.t_mclmc_warmup),
                "timing_sampling": float(stats.t_mclmc_sampling),
                "n_steps": int(stats.n_mclmc_steps),
                "n_full_loss": int(stats.n_mclmc_full_loss),
            },
        )

        # Save configuration
        save_config(run_dir, cfg)

        save_run_manifest(run_dir, cfg, stats)
        save_readme_snippet(run_dir, cfg)
        update_readme_with_run(run_dir, cfg)  # Optional auto-update

        # Generate HTML gallery if we have saved files
        if saved_files:
            gallery_path = write_html_gallery(
                run_dir, saved_files, title="LLC Sampler Diagnostics"
            )
            print(f"HTML gallery: {gallery_path}")

        print(f"Artifacts saved to: {run_dir}")

    jax.block_until_ready(hmc_samples_thin)  # Final sync before runtime report
    print(f"\nDone in {time.time() - t0:.1f}s.")


# ----------------------------
# Experiment runner for parameter sweeps
# ----------------------------
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
    init_hmc = theta_star + 0.01 * random.normal(k_hmc, (cfg.chains, dim))

    _, _, _, _, _, Ln_hist_hmc = run_hmc_online_with_adaptation(
        k_hmc,
        init_hmc,
        logpost_grad,
        cfg.hmc_num_integration_steps,
        cfg.hmc_warmup,
        cfg.hmc_draws,
        cfg.hmc_thin,
        cfg.hmc_eval_every,
        Ln_full64,  # Use f64 for consistent LLC evaluation
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
