"""Optuna configuration schema and loader.

Provides YAML-first configuration for hyperparameter optimization with:
- Declarative search spaces (no hardcoded ranges in Python)
- Executor routing (AUTO mapping or explicit per-method/role)
- Unified budgets and concurrency controls
- Validation against loaded Parsl configuration
"""

import logging
from pathlib import Path

import parsl
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _available_executor_labels() -> set[str]:
    """Read executor labels from the currently loaded Parsl config.

    Returns:
        Set of executor labels (e.g., {"htex64", "htex32"} or {"htex_slurm"})

    Raises:
        RuntimeError: If Parsl is not loaded yet
    """
    try:
        dfk = parsl.dfk()
        return {ex.label for ex in dfk.config.executors}
    except RuntimeError as e:
        raise RuntimeError(
            "Parsl must be loaded before validating Optuna config. Call parsl.load(...) first."
        ) from e


def _resolve_executor_map(cfg: DictConfig, available_labels: set[str]) -> dict[str, str]:
    """Resolve executor routing rules (AUTO or explicit).

    Args:
        cfg: Optuna configuration
        available_labels: Set of executor labels from loaded Parsl config

    Returns:
        Mapping from (role, method) keys to executor labels.
        Keys: "hmc" for HMC reference, "{method_name}" for method trials.

    AUTO rules (when cfg.execution.executor_map is omitted):
    1. If both htex64 and htex32 exist: hmc→htex64, mclmc→htex64, sgld/vi→htex32
    2. If only one executor exists: route all to that label (warn once)
    """
    explicit_map = cfg.execution.get("executor_map", {})

    # Build final routing table
    routing = {}

    # HMC reference routing
    if "hmc" in explicit_map:
        routing["hmc"] = explicit_map["hmc"]
    elif "htex64" in available_labels:
        routing["hmc"] = "htex64"
    elif len(available_labels) == 1:
        routing["hmc"] = next(iter(available_labels))
        log.warning(
            "AUTO: Only one executor (%s) available; routing HMC there.",
            routing["hmc"],
        )
    else:
        raise RuntimeError(f"Cannot route HMC: no htex64 and multiple executors {available_labels}")

    # Method routing
    methods = cfg.get("methods", [])
    explicit_methods = explicit_map.get("methods", {})

    both_htex = {"htex64", "htex32"}.issubset(available_labels)

    for method in methods:
        if method in explicit_methods:
            routing[method] = explicit_methods[method]
        elif both_htex and method == "mclmc":
            routing[method] = "htex64"
        elif both_htex:  # sgld, vi default to htex32
            routing[method] = "htex32"
        elif len(available_labels) == 1:
            routing[method] = next(iter(available_labels))
        else:
            raise RuntimeError(f"Cannot route method {method}: no AUTO rule and multiple executors")

    return routing


def _validate_executor_labels(routing: dict[str, str], available_labels: set[str]):
    """Validate that all routed executors actually exist.

    Args:
        routing: Executor routing map
        available_labels: Available executor labels from Parsl config

    Raises:
        ValueError: If any routed label is not available
    """
    for key, label in routing.items():
        if label not in available_labels:
            raise ValueError(
                f"Executor '{label}' for {key} not found in Parsl config. "
                f"Available: {available_labels}"
            )


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

    # Validate per-executor caps if specified
    per_executor = cfg.optuna.concurrency.get("per_executor", {})
    for label, cap in per_executor.items():
        if cap < 1:
            raise ValueError(f"optuna.concurrency.per_executor.{label} must be >= 1, got {cap}")


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
    validate_executors: bool = True,
) -> DictConfig:
    """Load and validate Optuna configuration from YAML.

    Args:
        config_path: Path to Optuna config YAML
        dotlist_overrides: Optional CLI overrides (e.g., ["optuna.max_trials_per_method=50"])
        validate_executors: If True, validate executor labels against loaded Parsl config

    Returns:
        Resolved and validated configuration

    Raises:
        FileNotFoundError: If config_path doesn't exist
        ValueError: If configuration is invalid
        RuntimeError: If Parsl not loaded (when validate_executors=True)

    Side effects:
        Logs detected executor labels and effective routing
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

    # Executor validation and routing
    if validate_executors:
        available = _available_executor_labels()
        log.info("Detected executor labels: %s", available)

        routing = _resolve_executor_map(cfg, available)
        _validate_executor_labels(routing, available)

        log.info("Executor routing:")
        for key, label in sorted(routing.items()):
            log.info("  %s → %s", key, label)

        # Store resolved routing in config for workflow access
        cfg._executor_routing = routing  # type: ignore

    return cfg
