# llc/validation.py

__all__ = ["validate_mclmc_config", "validate_config_before_dispatch"]

def validate_mclmc_config(cfg: dict):
    """Validate MCLMC configuration parameters."""
    bad_deprecated = [k for k in cfg if k in
        ("mclmc_tune_steps","mclmc_num_steps_tune1","mclmc_num_steps_tune2","mclmc_num_steps_tune3")]
    if bad_deprecated:
        raise ValueError("Found deprecated MCLMC parameters: "
                         f"{bad_deprecated}. Migration guide: blackjax-devs.github.io")

    num_steps = int(cfg.get("mclmc_num_steps", 0))
    if num_steps <= 0:
        raise ValueError("mclmc_num_steps must be a positive integer")

    fracs = [float(cfg.get(f"mclmc_frac_tune{i}", 0.0)) for i in (1,2,3)]
    for i, f in enumerate(fracs, 1):
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"mclmc_frac_tune{i} must be a number in [0.0, 1.0]")

    if sum(fracs) > 1.0 + 1e-12:
        s = sum(fracs)
        raise ValueError(f"frac_tune1 + frac_tune2 + frac_tune3 ≤ 1.0 required; fractions sum to {s:.3f} > 1.0")

    # Unknown keys check (restrict to known set)
    known = {
        "samplers",
        "mclmc_num_steps","mclmc_frac_tune1","mclmc_frac_tune2","mclmc_frac_tune3",
        "mclmc_desired_energy_var","mclmc_trust_in_estimate","mclmc_num_effective_samples",
        "mclmc_diagonal_preconditioning","mclmc_integrator",
    }
    unknown = [k for k in cfg if k.startswith("mclmc_") and k not in known]
    if unknown:
        raise ValueError(f"Unknown MCLMC parameters: {unknown}. "
                         "Valid MCLMC parameters include: "
                         f"{sorted([k for k in known if k.startswith('mclmc_')])}")

def validate_config_before_dispatch(cfg: dict):
    """Validate configuration before dispatching to samplers."""
    samplers = cfg.get("samplers", ())
    if isinstance(samplers, str):
        samplers = (samplers,)
    if "mclmc" in samplers:
        validate_mclmc_config(cfg)