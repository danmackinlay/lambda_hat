# lambda_hat/types.py
"""Shared type definitions to avoid circular imports."""

from typing import Dict, NamedTuple

import jax.numpy as jnp


class SamplerRunResult(NamedTuple):
    traces: Dict[str, jnp.ndarray]
    timings: Dict[str, float]  # {'adaptation': 0.0, 'sampling': 0.0, 'total': 0.0}
    work: Dict[str, float]  # {'n_full_loss': int, 'n_minibatch_grads': int}
