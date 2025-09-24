# llc/validation.py
"""Configuration validation with fail-fast approach"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# BlackJAX 1.2.5 documented MCLMC parameters
VALID_MCLMC_PARAMS = {
    # Core parameters
    "mclmc_draws",
    "mclmc_eval_every",
    "mclmc_thin",
    "mclmc_dtype",
    # Tuning parameters (fractional API)
    "mclmc_num_steps",
    "mclmc_frac_tune1",
    "mclmc_frac_tune2",
    "mclmc_frac_tune3",
    # Adaptation knobs (documented in BlackJAX 1.2.5)
    "mclmc_diagonal_preconditioning",
    "mclmc_desired_energy_var",
    "mclmc_trust_in_estimate",
    "mclmc_num_effective_samples",
    "mclmc_integrator",
    # Optional adjusted MCLMC
    "mclmc_adjusted",
    "mclmc_adjusted_target_accept",
    "mclmc_grad_per_step_override",
}

# Deprecated parameters that should trigger clear error messages
DEPRECATED_MCLMC_PARAMS = {
    "mclmc_tune_steps": "mclmc_num_steps",
    "mclmc_num_steps_tune1": "mclmc_frac_tune1 * mclmc_num_steps",
    "mclmc_num_steps_tune2": "mclmc_frac_tune2 * mclmc_num_steps",
    "mclmc_num_steps_tune3": "mclmc_frac_tune3 * mclmc_num_steps",
}


def validate_mclmc_config(cfg_dict: Dict[str, Any]) -> None:
    """
    Strict validator for MCLMC parameters. Fails fast on invalid/deprecated keys.

    Args:
        cfg_dict: Configuration dictionary to validate

    Raises:
        ValueError: On validation failure with clear migration guidance
    """
    # Check for deprecated parameters
    deprecated_found = []
    for old_key, replacement in DEPRECATED_MCLMC_PARAMS.items():
        if old_key in cfg_dict:
            deprecated_found.append((old_key, replacement))

    if deprecated_found:
        error_lines = [
            "Found deprecated MCLMC parameters that are incompatible with BlackJAX 1.2.5:",
            ""
        ]
        for old_key, replacement in deprecated_found:
            error_lines.append(f"  ❌ {old_key} → use {replacement}")

        error_lines.extend([
            "",
            "Migration guide:",
            "  1. Replace mclmc_tune_steps with mclmc_num_steps (total adaptation steps)",
            "  2. Replace num_steps_tune1/2/3 with mclmc_frac_tune1/2/3 (fractions ∈ [0,1])",
            "  3. Ensure mclmc_frac_tune1 + mclmc_frac_tune2 + mclmc_frac_tune3 ≤ 1.0",
            "",
            "See: https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/mclmc_adaptation/"
        ])
        raise ValueError("\n".join(error_lines))

    # Check for unknown MCLMC parameters
    mclmc_keys = {k for k in cfg_dict.keys() if k.startswith("mclmc_")}
    unknown_keys = mclmc_keys - VALID_MCLMC_PARAMS

    if unknown_keys:
        error_lines = [
            f"Unknown MCLMC parameters: {sorted(unknown_keys)}",
            "",
            "Valid MCLMC parameters for BlackJAX 1.2.5:",
        ]
        for param in sorted(VALID_MCLMC_PARAMS):
            error_lines.append(f"  ✓ {param}")

        error_lines.extend([
            "",
            "Documentation: https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/mclmc_adaptation/"
        ])
        raise ValueError("\n".join(error_lines))

    # Validate fraction semantics
    fractions = []
    for frac_key in ["mclmc_frac_tune1", "mclmc_frac_tune2", "mclmc_frac_tune3"]:
        if frac_key in cfg_dict:
            frac_val = cfg_dict[frac_key]
            if not isinstance(frac_val, (int, float)) or not (0.0 <= frac_val <= 1.0):
                raise ValueError(f"{frac_key} must be a number in [0.0, 1.0], got: {frac_val}")
            fractions.append(frac_val)

    if fractions and sum(fractions) > 1.0:
        raise ValueError(
            f"MCLMC tuning fractions sum to {sum(fractions):.3f} > 1.0. "
            f"Must satisfy: frac_tune1 + frac_tune2 + frac_tune3 ≤ 1.0"
        )

    # Validate num_steps is positive
    if "mclmc_num_steps" in cfg_dict:
        num_steps = cfg_dict["mclmc_num_steps"]
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError(f"mclmc_num_steps must be a positive integer, got: {num_steps}")

        # Validate that no nonzero fraction would yield zero steps (starvation check)
        ns = num_steps
        f1 = float(cfg_dict.get("mclmc_frac_tune1", 0.0))
        f2 = float(cfg_dict.get("mclmc_frac_tune2", 0.0))
        f3 = float(cfg_dict.get("mclmc_frac_tune3", 0.0))

        # If any fraction > 0 but ns*f < 1, fail
        starving = [i for i, f in enumerate((f1, f2, f3), start=1) if f > 0 and int(ns * f) == 0]
        if starving:
            raise ValueError(f"MCLMC config invalid: num_steps={ns} too small for nonzero "
                             f"fractions in phases {starving}. Increase num_steps or zero those fractions.")

    logger.debug("MCLMC configuration validation passed")


def validate_config_before_dispatch(cfg_dict: Dict[str, Any]) -> None:
    """
    Validate configuration before job dispatch. Call this once per atomic run.

    Args:
        cfg_dict: Configuration dictionary to validate

    Raises:
        ValueError: On validation failure
    """
    # Check samplers and validate relevant parameters
    samplers = cfg_dict.get("samplers", ())
    if isinstance(samplers, str):
        samplers = (samplers,)

    if "mclmc" in samplers:
        validate_mclmc_config(cfg_dict)

    # Future: Add validators for other samplers as needed
    # if "hmc" in samplers:
    #     validate_hmc_config(cfg_dict)