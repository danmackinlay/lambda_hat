"""Optuna configuration schema and loader.

Provides YAML-first configuration for hyperparameter optimization with:
- Declarative search spaces (no hardcoded ranges in Python)
- Unified budgets and concurrency controls
- Validation of search space definitions
"""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _validate_budgets(cfg: DictConfig):
    """Validate budget and concurrency settings.

    Args:
        cfg: Optuna configuration

    Raises:
        ValueError: If budgets are invalid
    """
    hmc_sec = cfg.execution.budget.hmc_sec
    trial_sec = cfg.execution.budget.trial_sec
    batch_size = cfg.optuna.concurrency.batch_size

    if hmc_sec <= 0:
        raise ValueError(f"execution.budget.hmc_sec must be > 0, got {hmc_sec}")
    if trial_sec <= 0:
        raise ValueError(f"execution.budget.trial_sec must be > 0, got {trial_sec}")
    if batch_size < 1:
        raise ValueError(f"optuna.concurrency.batch_size must be >= 1, got {batch_size}")


def _validate_search_spaces(cfg: DictConfig):
    """Validate search space definitions.

    Args:
        cfg: Optuna configuration

    Raises:
        ValueError: If search spaces are invalid
    """
    methods = cfg.get("methods", [])
    search_space = cfg.get("search_space", {})

    for method in methods:
        if method not in search_space:
            raise ValueError(f"Method '{method}' is listed but has no search_space definition")

        method_space = search_space[method]
        for param_name, spec in method_space.items():
            dist = spec.get("dist")
            if dist not in ("float", "int", "categorical"):
                raise ValueError(
                    f"Invalid dist '{dist}' for {method}.{param_name}. "
                    f"Must be float, int, or categorical."
                )

            # Validate required fields per distribution type
            if dist in ("float", "int"):
                if "low" not in spec or "high" not in spec:
                    raise ValueError(f"{method}.{param_name} requires 'low' and 'high' for {dist}")
            elif dist == "categorical":
                if "choices" not in spec:
                    raise ValueError(f"{method}.{param_name} requires 'choices' for categorical")


def load_cfg(
    config_path: Path | str,
    dotlist_overrides: list[str] | None = None,
) -> DictConfig:
    """Load and validate Optuna configuration from YAML.

    Args:
        config_path: Path to Optuna config YAML
        dotlist_overrides: Optional CLI overrides (e.g., ["optuna.max_trials_per_method=50"])

    Returns:
        Resolved and validated configuration

    Raises:
        FileNotFoundError: If config_path doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Optuna config not found: {config_path}")

    # Load base config
    cfg = OmegaConf.load(config_path)

    # Apply CLI overrides
    if dotlist_overrides:
        overrides_conf = OmegaConf.from_dotlist(dotlist_overrides)
        cfg = OmegaConf.merge(cfg, overrides_conf)

    # Set defaults for optional fields
    OmegaConf.set_struct(cfg, False)  # Allow adding missing fields

    if "optuna" not in cfg:
        cfg.optuna = {}
    if "concurrency" not in cfg.optuna:
        cfg.optuna.concurrency = {}
    if "batch_size" not in cfg.optuna.concurrency:
        cfg.optuna.concurrency.batch_size = 16

    if "execution" not in cfg:
        cfg.execution = {}
    if "budget" not in cfg.execution:
        cfg.execution.budget = {}
    if "hmc_sec" not in cfg.execution.budget:
        cfg.execution.budget.hmc_sec = 7200  # 2 hours default
    if "trial_sec" not in cfg.execution.budget:
        cfg.execution.budget.trial_sec = 600  # 10 minutes default

    # Validate budgets and search spaces
    _validate_budgets(cfg)
    _validate_search_spaces(cfg)

    return cfg
