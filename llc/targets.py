from dataclasses import dataclass
from typing import Callable, List, Optional
import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
import optax

from .config import Config

@dataclass
class TargetBundle:
    d: int
    theta0_f32: jnp.ndarray
    theta0_f64: jnp.ndarray
    loss_full_f32: Callable[[jnp.ndarray], jnp.ndarray]
    loss_minibatch_f32: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    loss_full_f64: Callable[[jnp.ndarray], jnp.ndarray]
    loss_minibatch_f64: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    X_f32: jnp.ndarray
    Y_f32: jnp.ndarray
    X_f64: jnp.ndarray
    Y_f64: jnp.ndarray
    L0: float

# Model utilities
def infer_widths(in_dim: int, out_dim: int, depth: int,
                 target_params: Optional[int], fallback_width: int = 128) -> List[int]:
    """Infer widths to hit target_params, or use fallback"""
    if target_params is None:
        return [fallback_width] * depth
    L = depth
    a = L - 1  # coefficient of h^2
    b = (in_dim + 1) + (L - 1) + out_dim  # coefficient of h
    if a == 0:
        h = max(1, (target_params - out_dim) // (in_dim + 1))
    else:
        disc = b * b + 4 * a * target_params
        h = int((-b + jnp.sqrt(disc)) / (2 * a))
        h = int(max(1, h))
    return [h] * L

def act_fn(name: str):
    """Activation function factory"""
    return {
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "gelu": jax.nn.gelu,
        "identity": (lambda x: x),
    }[name]

def init_mlp_params(key, in_dim: int, widths: List[int], out_dim: int, activation: str):
    """Initialize MLP with arbitrary depth"""
    keys = random.split(key, len(widths) + 1)
    layers = []
    prev = in_dim

    for i, h in enumerate(widths):
        # He initialization for ReLU
        W = random.normal(keys[i], (h, prev)) * jnp.sqrt(2.0 / prev)
        b = jnp.zeros((h,))
        layers.append({"W": W, "b": b})
        prev = h

    # Output layer (Xavier init)
    W = random.normal(keys[-1], (out_dim, prev)) * jnp.sqrt(1.0 / prev)
    b = jnp.zeros((out_dim,))
    out_layer = {"W": W, "b": b}

    return {"layers": layers, "out": out_layer}

def mlp_forward(params, x, activation: str = "relu"):
    """Forward pass through MLP"""
    act = act_fn(activation)

    h = x
    for lyr in params["layers"]:
        z = h @ lyr["W"].T + lyr["b"]
        h = act(z)

    # Output layer (linear)
    y = h @ params["out"]["W"].T + params["out"]["b"]
    return y

# Data generation
def sample_X(key, cfg: Config, n: int, in_dim: int):
    """Sample inputs according to distribution"""
    if cfg.x_dist == "gauss_iso":
        return random.normal(key, (n, in_dim))
    elif cfg.x_dist == "mixture":
        # Simple 4-component mixture
        k1, k2, k3 = random.split(key, 3)
        k = 4  # number of components
        centers = random.normal(k1, (k, in_dim))
        centers = 2.0 * centers / (1e-6 + jnp.linalg.norm(centers, axis=1, keepdims=True))
        comp = random.randint(k2, (n,), 0, k)
        eps = random.normal(k3, (n, in_dim))
        return centers[comp] + eps
    else:
        raise ValueError(f"Unknown x_dist: {cfg.x_dist}")

def add_noise(key, Y_clean, cfg: Config):
    """Add noise to outputs"""
    if cfg.noise_model == "gauss":
        noise = random.normal(key, Y_clean.shape) * cfg.noise_scale
        return Y_clean + noise
    elif cfg.noise_model == "student_t":
        # Student-t noise
        k1, k2 = random.split(key)
        nu = cfg.student_df
        g = random.gamma(k1, nu/2, Y_clean.shape) / (nu/2)
        z = random.normal(k2, Y_clean.shape) / jnp.sqrt(g)
        return Y_clean + cfg.noise_scale * z
    else:
        raise ValueError(f"Unknown noise_model: {cfg.noise_model}")

# Loss functions
def make_loss_fns(unravel, cfg: Config, X, Y):
    """Create loss functions for both full data and minibatch"""
    if cfg.loss == "mse":
        def full(theta):
            params = unravel(theta)
            pred = mlp_forward(params, X, cfg.activation)
            return jnp.mean((pred - Y) ** 2)

        def minibatch(theta, Xb, Yb):
            params = unravel(theta)
            pred = mlp_forward(params, Xb, cfg.activation)
            return jnp.mean((pred - Yb) ** 2)

    elif cfg.loss == "t_regression":
        s2 = cfg.noise_scale**2
        nu = cfg.student_df

        def neglogt(resid):
            return 0.5 * (nu + 1) * jnp.log1p((resid**2) / (nu * s2))

        def full(theta):
            params = unravel(theta)
            pred = mlp_forward(params, X, cfg.activation)
            return jnp.mean(neglogt(pred - Y))

        def minibatch(theta, Xb, Yb):
            params = unravel(theta)
            pred = mlp_forward(params, Xb, cfg.activation)
            return jnp.mean(neglogt(pred - Yb))
    else:
        raise ValueError(f"Unknown loss: {cfg.loss}")

    return full, minibatch

# ERM training
def train_erm(w_init_pytree, cfg: Config, X, Y, steps=2000, lr=1e-2):
    """Train to empirical risk minimizer"""
    theta, unravel = ravel_pytree(w_init_pytree)
    loss_full, _ = make_loss_fns(unravel, cfg, X, Y)
    opt = optax.adam(lr)
    opt_state = opt.init(theta)

    @jax.jit
    def step(theta, opt_state):
        loss, g = jax.value_and_grad(loss_full)(theta)
        updates, opt_state = opt.update(g, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss

    for _ in range(steps):
        theta, opt_state, _ = step(theta, opt_state)

    return theta, unravel

def as_dtype(x, dtype_str):
    """Convert array to specified dtype"""
    return x.astype(jnp.float32 if dtype_str == "float32" else jnp.float64)

def build_target(key, cfg: Config) -> TargetBundle:
    """Teacher–student MLP or quadratic baseline; returns closures + θ*; cast to f32/f64."""
    if cfg.target == "mlp":
        # Generate data
        key, k1, k2, k3 = random.split(key, 4)
        n = cfg.n_data
        X = sample_X(k1, cfg, n, cfg.in_dim)

        # Build and evaluate teacher
        widths = cfg.widths or infer_widths(cfg.in_dim, cfg.out_dim, cfg.depth,
                                           cfg.target_params, fallback_width=300)
        teacher_params = init_mlp_params(k2, cfg.in_dim, widths, cfg.out_dim, cfg.activation)
        Y_clean = mlp_forward(teacher_params, X, cfg.activation)
        Y = add_noise(k3, Y_clean, cfg)

        # Init student params & train to ERM (θ*)
        key, subkey = random.split(key)
        w0_pytree = init_mlp_params(subkey, cfg.in_dim, widths, cfg.out_dim, cfg.activation)
        theta_star_f64, unravel_star_f64 = train_erm(
            w0_pytree, cfg, X.astype(jnp.float64), Y.astype(jnp.float64)
        )

        # Convert to f32
        params_star_f64 = unravel_star_f64(theta_star_f64)
        params_star_f32 = jax.tree_util.tree_map(
            lambda a: a.astype(jnp.float32), params_star_f64
        )
        theta0_f32, unravel_star_f32 = ravel_pytree(params_star_f32)

        # Cast data to both dtypes
        X_f32, Y_f32 = as_dtype(X, cfg.sgld_dtype), as_dtype(Y, cfg.sgld_dtype)
        X_f64, Y_f64 = as_dtype(X, cfg.hmc_dtype), as_dtype(Y, cfg.hmc_dtype)

        # Loss fns for each dtype
        loss_full_f32, loss_minibatch_f32 = make_loss_fns(
            unravel_star_f32, cfg, X_f32, Y_f32
        )
        loss_full_f64, loss_minibatch_f64 = make_loss_fns(
            unravel_star_f64, cfg, X_f64, Y_f64
        )

        L0 = float(loss_full_f64(theta_star_f64))
        d = int(theta0_f32.size)

        return TargetBundle(
            d=d,
            theta0_f32=theta0_f32,
            theta0_f64=theta_star_f64,
            loss_full_f32=loss_full_f32,
            loss_minibatch_f32=loss_minibatch_f32,
            loss_full_f64=loss_full_f64,
            loss_minibatch_f64=loss_minibatch_f64,
            X_f32=X_f32,
            Y_f32=Y_f32,
            X_f64=X_f64,
            Y_f64=Y_f64,
            L0=L0,
        )

    elif cfg.target == "quadratic":
        # Analytic diagnostic: L_n(θ) = 0.5 ||θ||^2
        d = int(cfg.quad_dim or cfg.target_params or cfg.in_dim)
        theta0_f64 = jnp.zeros((d,), dtype=jnp.float64)
        theta0_f32 = theta0_f64.astype(jnp.float32)

        # loss_full(θ) = 0.5 ||θ||^2 ; minibatch ignores Xb,Yb but keeps signature
        def _lf(theta):
            return 0.5 * jnp.sum(theta * theta)

        def _lb(theta, Xb, Yb):
            return _lf(theta)

        # Provide trivial data so SGLD minibatching works without special cases
        n = int(cfg.n_data)
        X_f32 = jnp.zeros((n, 1), dtype=jnp.float32)
        Y_f32 = jnp.zeros((n, 1), dtype=jnp.float32)
        X_f64 = jnp.zeros((n, 1), dtype=jnp.float64)
        Y_f64 = jnp.zeros((n, 1), dtype=jnp.float64)

        L0 = 0.0  # L_n at θ0=0
        return TargetBundle(
            d=d,
            theta0_f32=theta0_f32,
            theta0_f64=theta0_f64,
            loss_full_f32=lambda th: _lf(th.astype(jnp.float32)).astype(jnp.float32),
            loss_minibatch_f32=lambda th, Xb, Yb: _lb(
                th.astype(jnp.float32), Xb, Yb
            ).astype(jnp.float32),
            loss_full_f64=lambda th: _lf(th.astype(jnp.float64)).astype(jnp.float64),
            loss_minibatch_f64=lambda th, Xb, Yb: _lb(
                th.astype(jnp.float64), Xb, Yb
            ).astype(jnp.float64),
            X_f32=X_f32,
            Y_f32=Y_f32,
            X_f64=X_f64,
            Y_f64=Y_f64,
            L0=L0,
        )
    else:
        raise ValueError(f"Unknown target: {cfg.target}")