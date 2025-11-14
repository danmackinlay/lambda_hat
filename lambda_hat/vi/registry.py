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
    # Lazy-load flow on first access to avoid requiring flowjax
    if name == "flow" and name not in _REGISTRY:
        try:
            from lambda_hat.vi import flow  # noqa: F401
        except ImportError:
            # flow.py will raise a better error message when run() is called
            pass

    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown VI algorithm '{name}'. Available algorithms: {available}")
    return _REGISTRY[name]()


# Import algorithm modules to trigger registration
# This must come after the registry functions are defined
# Note: flow is imported lazily in get() to avoid requiring flowjax
from lambda_hat.vi import mfa  # noqa: E402, F401
