# lambda_hat/vi/registry.py
"""Registry for variational inference algorithms."""

from typing import Callable, Dict

from lambda_hat.vi.base import VIAlgorithm

# Registry will be populated by algorithm modules
_REGISTRY: Dict[str, Callable[[], VIAlgorithm]] = {}


def register(name: str, factory: Callable[[], VIAlgorithm]) -> None:
    """Register a VI algorithm factory.

    Args:
        name: Algorithm name (e.g., "mfa", "flow")
        factory: Callable that returns a VIAlgorithm instance
    """
    _REGISTRY[name] = factory


def get(name: str) -> VIAlgorithm:
    """Get a VI algorithm by name.

    Args:
        name: Algorithm name (e.g., "mfa", "flow")

    Returns:
        VIAlgorithm instance

    Raises:
        ValueError: If algorithm name is not registered
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown VI algorithm '{name}'. Available algorithms: {available}")
    return _REGISTRY[name]()


# Import algorithm modules to trigger registration
# This must come after the registry functions are defined
from lambda_hat.vi import flow, mfa  # noqa: E402, F401
