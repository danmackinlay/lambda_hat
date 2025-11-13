"""Pytest configuration for lambda_hat tests.

CRITICAL: Enable JAX x64 precision BEFORE any imports of JAX or lambda_hat.
This must be the first thing that happens in the test environment.
"""

import os

# Set environment variable before JAX is imported anywhere
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Now import JAX and configure it
import jax

jax.config.update("jax_enable_x64", True)
