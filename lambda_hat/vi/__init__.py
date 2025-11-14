# lambda_hat/vi/__init__.py
"""Variational inference algorithms for Local Learning Coefficient estimation."""

# Explicitly import algorithm modules to trigger registration
# This ensures algorithms are available when importing the vi package
from lambda_hat.vi import flow, mfa  # noqa: F401
from lambda_hat.vi.base import VIAlgorithm

# Re-export key utilities for convenience
from lambda_hat.vi.mfa import fit_vi_and_estimate_lambda, make_whitener
from lambda_hat.vi.registry import get, register

__all__ = [
    "VIAlgorithm",
    "get",
    "register",
    "fit_vi_and_estimate_lambda",
    "make_whitener",
]
