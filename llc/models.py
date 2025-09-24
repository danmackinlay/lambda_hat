# llc/models.py
"""Modernized model definitions using Haiku"""

from typing import Optional, List
import haiku as hk
import jax
import jax.numpy as jnp


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
    b = (in_dim + 1) + (L - 1) + out_dim  # coefficient of h
    if a == 0:
        h = max(1, (target_params - out_dim) // (in_dim + 1))
    else:
        disc = b * b + 4 * a * target_params
        h = int((-b + jnp.sqrt(disc)) / (2 * a))
        h = int(max(1, h))
    return [h] * L


class MLP(hk.Module):
    """Flexible MLP architecture using Haiku"""

    def __init__(
        self,
        widths: List[int],
        out_dim: int,
        activation: str = "relu",
        bias: bool = True,
        init: str = "he",
        skip: bool = False,
        residual_period: int = 2,
        layernorm: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.widths = widths
        self.out_dim = out_dim
        self.activation = activation
        self.bias = bias
        self.init = init
        self.skip = skip
        self.residual_period = residual_period
        self.layernorm = layernorm

    def _get_activation(self):
        """Get activation function"""
        activations = {
            "relu": jax.nn.relu,
            "tanh": jnp.tanh,
            "gelu": jax.nn.gelu,
            "identity": lambda x: x,
        }
        return activations[self.activation]

    def _get_initializer(self):
        """Get weight initializer"""
        if self.init == "he":
            return hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")
        elif self.init == "xavier":
            return hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")
        elif self.init == "lecun":
            return hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")
        elif self.init == "orthogonal":
            # Use He scaling for consistency with original implementation
            return hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")
        else:
            return hk.initializers.TruncatedNormal(stddev=1.0)

    def __call__(self, x):
        """Forward pass with optional residuals and layer norm"""
        act = self._get_activation()
        w_init = self._get_initializer()

        h = x

        # Hidden layers
        for i, width in enumerate(self.widths):
            linear = hk.Linear(
                width,
                with_bias=self.bias,
                w_init=w_init,
                name=f"layer_{i}"
            )
            z = linear(h)

            if self.layernorm:
                z = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(z)

            h_new = act(z)

            # Skip connections
            if self.skip and (i % self.residual_period == self.residual_period - 1):
                # Project if dimensions differ
                if h.shape[-1] != h_new.shape[-1]:
                    d_in, d_out = h.shape[-1], h_new.shape[-1]
                    h = h[:, :min(d_in, d_out)]
                    if d_out > h.shape[-1]:
                        pad = jnp.zeros((h.shape[0], d_out - h.shape[-1]), dtype=h.dtype)
                        h = jnp.concatenate([h, pad], axis=1)
                h = h + h_new
            else:
                h = h_new

        # Output layer (always Xavier initialization for stability)
        output_init = hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")
        y = hk.Linear(
            self.out_dim,
            with_bias=self.bias,
            w_init=output_init,
            name="output"
        )(h)

        return y


def build_mlp_forward_fn(
    in_dim: int,
    widths: List[int],
    out_dim: int,
    activation: str = "relu",
    bias: bool = True,
    init: str = "he",
    skip: bool = False,
    residual_period: int = 2,
    layernorm: bool = False,
):
    """Build a Haiku-transformed MLP forward function"""

    def forward_fn(x):
        mlp = MLP(
            widths=widths,
            out_dim=out_dim,
            activation=activation,
            bias=bias,
            init=init,
            skip=skip,
            residual_period=residual_period,
            layernorm=layernorm,
        )
        return mlp(x)

    # Transform to pure JAX function
    model = hk.transform(forward_fn)
    return model


def count_params(params):
    """Count total parameters in a Haiku parameter tree"""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))