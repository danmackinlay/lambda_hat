# lambda_hat/vi/__init__.py
"""Variational inference algorithms for Local Learning Coefficient estimation."""

# Import MFA eagerly (always available)
from lambda_hat.vi import mfa  # noqa: F401
from lambda_hat.vi.base import VIAlgorithm

# Re-export key utilities for convenience
from lambda_hat.vi.mfa import fit_vi_and_estimate_lambda, make_whitener
from lambda_hat.vi.registry import get, register

# Flow is imported lazily via registry to avoid requiring flowjax
# It will be loaded on first use via get("flow")

__all__ = [
    "VIAlgorithm",
    "get",
    "register",
    "fit_vi_and_estimate_lambda",
    "make_whitener",
]
