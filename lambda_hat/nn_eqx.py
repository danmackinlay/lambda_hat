"""Equinox neural network modules for Lambda-Hat.

This module provides MLP implementations using Equinox, replacing the legacy Haiku modules.
"""

from typing import Callable, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp


def _activation(name: str) -> Callable:
    """Get activation function by name."""
    activations = {
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "gelu": jax.nn.gelu,
        "identity": lambda x: x,
        "none": lambda x: x,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name]


def infer_widths(
    in_dim: int,
    out_dim: int,
    depth: int,
    target_params: Optional[int],
    fallback_width: int = 128,
) -> List[int]:
    """Infer widths to hit target_params, or use fallback.

    Computes MLP hidden layer widths that approximately achieve a target parameter count.
    For constant width h and L hidden layers:
        P(h) = (in_dim+1)h + (L-1)(h+1)h + (h+1)out_dim ≈ target_params

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        depth: Number of hidden layers
        target_params: Target parameter count (None = use fallback_width)
        fallback_width: Default width when target_params is None

    Returns:
        List of hidden layer widths (constant width across layers)
    """
    if target_params is None:
        return [fallback_width] * depth
    # For simplicity, use constant width h and solve approximately:
    # P(h) = (in_dim+1)h + (L-1)(h+1)h + (h+1)out_dim ≈ target_params
    L = depth
    a = L - 1  # coefficient of h^2
    b = (in_dim + 1) + (L - 1) + out_dim  # coefficient of h
    if a == 0:
        h = max(1, (target_params - out_dim) // (in_dim + 1))
    else:
        import math

        disc = float(b) * float(b) + 4.0 * float(a) * float(target_params)
        h = int((-float(b) + math.sqrt(disc)) / (2.0 * float(a)))
        h = int(max(1, h))
    return [h] * L


class EqxMLP(eqx.Module):
    """Multi-layer perceptron with optional layer normalization and skip connections.

    This is a simplified Equinox implementation replacing the legacy Haiku MLP.
    It provides the same core functionality (widths, activation, layernorm, skip connections)
    but with Equinox's cleaner API.

    Attributes:
        layers: List of linear layers (and optional LayerNorms)
        out_layer: Final output linear layer
        activation: Activation function
        use_layernorm: Whether to use LayerNorm after each hidden layer
        use_skip: Whether to use skip/residual connections
        residual_period: How often to add residuals (every N layers)
    """

    layers: List[eqx.Module]
    out_layer: eqx.nn.Linear
    activation: Callable
    use_layernorm: bool
    use_skip: bool
    residual_period: int

    def __init__(
        self,
        in_dim: int,
        widths: List[int],
        out_dim: int,
        activation: str = "relu",
        bias: bool = True,
        layernorm: bool = False,
        skip: bool = False,
        residual_period: int = 2,
        *,
        key: jax.Array,
    ):
        """Initialize MLP.

        Args:
            in_dim: Input dimension
            widths: Hidden layer widths (depth = len(widths))
            out_dim: Output dimension
            activation: Activation function name ("relu", "tanh", "gelu", "identity")
            bias: Whether to use bias in linear layers
            layernorm: Whether to use LayerNorm after each hidden layer
            skip: Whether to use skip/residual connections
            residual_period: Add residuals every N layers (when skip=True)
            key: JAX random key for initialization
        """
        keys = jax.random.split(key, len(widths) + 1)
        self.activation = _activation(activation)
        self.use_layernorm = layernorm
        self.use_skip = skip
        self.residual_period = residual_period

        # Build hidden layers
        layers = []
        current_dim = in_dim
        for i, width in enumerate(widths):
            layers.append(eqx.nn.Linear(current_dim, width, use_bias=bias, key=keys[i]))
            if layernorm:
                layers.append(eqx.nn.LayerNorm(width))
            current_dim = width

        self.layers = layers
        self.out_layer = eqx.nn.Linear(current_dim, out_dim, use_bias=bias, key=keys[-1])

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Handles both single samples (..., in_dim) and batches (batch, ..., in_dim).
        For batched inputs, automatically vmaps over the batch dimension.

        Args:
            x: Input array of shape (..., in_dim) or (batch, ..., in_dim)

        Returns:
            Output array of shape (..., out_dim) or (batch, ..., out_dim)
        """

        def _forward_single(x_single):
            """Forward pass for a single sample."""
            h = x_single
            layer_idx = 0  # Track actual layer index for residuals

            for layer in self.layers:
                if isinstance(layer, eqx.nn.LayerNorm):
                    # LayerNorm doesn't count as a layer for residual purposes
                    h = layer(h)
                else:
                    # Linear layer
                    h_prev = h
                    h = self.activation(layer(h))

                    # Add skip connection if enabled and at residual period
                    if self.use_skip and layer_idx > 0 and (layer_idx % self.residual_period == 0):
                        # Only add residual if dimensions match
                        if h_prev.shape == h.shape:
                            h = h + h_prev

                    layer_idx += 1

            return self.out_layer(h)

        # Check if input is batched (has more than 1 dimension)
        if x.ndim > 1:
            # Vmap over the first (batch) dimension
            return jax.vmap(_forward_single)(x)
        else:
            return _forward_single(x)


def build_mlp(
    in_dim: int,
    widths: List[int],
    out_dim: int,
    activation: str = "relu",
    bias: bool = True,
    layernorm: bool = False,
    skip: bool = False,
    residual_period: int = 2,
    *,
    key: jax.Array,
) -> EqxMLP:
    """Factory function to build an MLP.

    This replaces the Haiku build_mlp_forward_fn factory. Returns an Equinox module
    that can be called directly (no separate init/apply).

    Args:
        in_dim: Input dimension
        widths: Hidden layer widths
        out_dim: Output dimension
        activation: Activation function name
        bias: Whether to use bias
        layernorm: Whether to use LayerNorm
        skip: Whether to use skip connections
        residual_period: Residual period for skip connections
        key: JAX random key for initialization

    Returns:
        Initialized EqxMLP module (ready to call)
    """
    return EqxMLP(
        in_dim=in_dim,
        widths=widths,
        out_dim=out_dim,
        activation=activation,
        bias=bias,
        layernorm=layernorm,
        skip=skip,
        residual_period=residual_period,
        key=key,
    )


def count_params(model: eqx.Module) -> int:
    """Count trainable parameters in an Equinox module.

    Args:
        model: Equinox module

    Returns:
        Total number of trainable parameters
    """
    params, _ = eqx.partition(model, eqx.is_array)
    return sum(x.size for x in jax.tree.leaves(params))
