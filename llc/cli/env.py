"""Runtime environment configuration for CLI."""

import os


def configure_runtime():
    """Configure JAX and matplotlib environment for CLI usage."""
    # Set environment for plotting/headless at import time (match argparse CLI behavior)
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("MPLBACKEND", "Agg")  # headless backend for server environments