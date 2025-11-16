# lambda_hat/equinox_adapter.py
"""Equinox adapter: single safe interface for parameter manipulation.

This module provides the ONLY way samplers should interact with model parameters.
No sampler should ever call ravel_pytree or tree_leaves on a full model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

Array = jnp.ndarray
PRNGKey = Any  # jax.random.PRNGKey type


@dataclass(frozen=True)
class VectorisedModel:
    """A thin wrapper that exposes a flat-parameter view for samplers."""

    unravel_arrays: Callable[[Array], Any]  # flat -> arrays subtree
    static_tree: Any  # static subtree (non-arrays)
    size: int  # dimension of flat vector
    dtype: jnp.dtype

    def to_model(self, flat: Array) -> Any:
        """Convert flat vector back to full model (arrays + static)."""
        arrays = self.unravel_arrays(flat)
        return eqx.combine(arrays, self.static_tree)


def _cast_arrays_dtype(arrays, dtype):
    """Cast floating-point arrays to specified dtype."""
    return jax.tree.map(
        lambda x: (
            x.astype(dtype)
            if (hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating))
            else x
        ),
        arrays,
    )


def ensure_dtype(module: Any, dtype: jnp.dtype) -> Any:
    """Cast all floating-point arrays in module to specified dtype.

    Args:
        module: Equinox module
        dtype: Target dtype (jnp.float32 or jnp.float64)

    Returns:
        Module with arrays cast to dtype
    """
    arrays, static = eqx.partition(module, eqx.is_array)
    arrays = _cast_arrays_dtype(arrays, dtype)
    return eqx.combine(arrays, static)


def vectorise_model(module: Any, *, dtype: jnp.dtype) -> Tuple[VectorisedModel, Array]:
    """Return a VectorisedModel and its initial flat vector (cast to dtype).

    This is the ONLY safe way to flatten an Equinox model for samplers.

    Args:
        module: Equinox module
        dtype: Target dtype for parameters

    Returns:
        Tuple of (VectorisedModel, flat_params)
    """
    # Ensure correct dtype
    module = ensure_dtype(module, dtype)

    # Partition into arrays and static (activation functions, etc.)
    arrays, static = eqx.partition(module, eqx.is_array)

    # Flatten ONLY the arrays (SAFE: no activation functions)
    flat, unravel = ravel_pytree(arrays)
    flat = flat.astype(dtype)

    vm = VectorisedModel(unravel_arrays=unravel, static_tree=static, size=flat.size, dtype=dtype)
    return vm, flat


def filter_predict_fn(model_apply: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a model call with Equinox filtering for JIT-safety."""
    return eqx.filter_jit(model_apply)


def value_and_grad_filtered(fn: Callable[..., Array]) -> Callable[..., Tuple[Array, Array]]:
    """Compute value and gradient, correctly ignoring static leaves.

    Args:
        fn: Function to differentiate

    Returns:
        Function that returns (value, gradient)
    """
    return eqx.filter_value_and_grad(fn)
