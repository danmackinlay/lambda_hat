# llc/models.py
"""Model utilities for flexible MLP architectures"""

from typing import Optional, List
import jax
import jax.numpy as jnp
from jax import random


def infer_widths(
    in_dim: int,
    out_dim: int,
    depth: int,
    target_params: Optional[int],
    fallback_width: int = 128,
) -> List[int]:
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
    widths: List[int],
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