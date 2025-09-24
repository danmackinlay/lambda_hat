from dataclasses import dataclass, replace, fields
from typing import Optional, Literal, List
import yaml
import json
import hashlib
import logging

@dataclass
class Config:
    # problem / model
    target: Literal["mlp", "quadratic"] = "mlp"
    quad_dim: Optional[int] = None
    in_dim: int = 32
    out_dim: int = 1
    depth: int = 3
    widths: Optional[List[int]] = None
    activation: Literal["relu", "tanh", "gelu", "identity"] = "relu"
    target_params: Optional[int] = 10_000
    n_data: int = 20_000
    x_dist: Literal["gauss_iso","mixture"] = "gauss_iso"
    noise_model: Literal["gauss","student_t"] = "gauss"
    noise_scale: float = 0.1
    student_df: float = 4.0
    loss: Literal["mse","t_regression"] = "mse"

    # tempered local posterior
    beta_mode: Literal["1_over_log_n","fixed"] = "1_over_log_n"
    beta0: float = 1.0
    prior_radius: Optional[float] = None
    gamma: float = 1.0

    # samplers (exactly one per run)
    samplers: tuple[str, ...] = ("sgld",)
    chains: int = 4
    use_batched_chains: bool = True

    # SGLD
    sgld_steps: int = 16000
    sgld_warmup: int = 1000
    sgld_batch_size: int = 256
    sgld_step_size: float = 1e-6
    sgld_eval_every: int = 10
    sgld_thin: int = 20
    sgld_dtype: str = "float32"
    sgld_precond: Literal["none","rmsprop","adam"] = "none"
    sgld_beta1: float = 0.9
    sgld_beta2: float = 0.999
    sgld_eps: float = 1e-8
    sgld_bias_correction: bool = True

    # HMC
    hmc_draws: int = 5000
    hmc_warmup: int = 1000
    hmc_num_integration_steps: int = 10
    hmc_eval_every: int = 1
    hmc_thin: int = 5
    hmc_dtype: str = "float64"

    # MCLMC (BlackJAX 1.2.5 fractional tuner)
    mclmc_draws: int = 8000
    mclmc_eval_every: int = 1
    mclmc_thin: int = 10
    mclmc_dtype: str = "float64"
    mclmc_num_steps: int = 2000
    mclmc_frac_tune1: float = 0.1
    mclmc_frac_tune2: float = 0.1
    mclmc_frac_tune3: float = 0.1
    mclmc_diagonal_preconditioning: bool = False
    mclmc_desired_energy_var: float = 5e-4
    mclmc_trust_in_estimate: float = 1.0
    mclmc_num_effective_samples: float = 150.0
    mclmc_integrator: Literal[
        "isokinetic_mclachlan",
        "isokinetic_velocity_verlet",
        "isokinetic_yoshida",
        "isokinetic_omelyan"
    ] = "isokinetic_mclachlan"

    # io/misc
    seed: int = 42
    runs_dir: str = "runs"
    save_plots: bool = True
    show_plots: bool = False

# tiny presets (copy your numbers as needed)
def apply_preset(cfg: Config, preset: Optional[str]) -> Config:
    if preset == "quick":
        return replace(cfg, sgld_steps=1000, sgld_warmup=200, hmc_draws=200, hmc_warmup=100,
                       mclmc_draws=400, chains=4, n_data=1000, save_plots=True)
    if preset == "full":
        return replace(cfg, sgld_steps=10000, sgld_warmup=2000, hmc_draws=2000, hmc_warmup=1000,
                       mclmc_draws=4000, chains=4, n_data=5000)
    return cfg

def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)

def coerce_types(cfg_dict: dict) -> dict:
    """Coerce string values to appropriate types based on Config field types."""
    type_map = {}
    for field in fields(Config):
        type_map[field.name] = field.type

    result = {}
    for k, v in cfg_dict.items():
        if k not in type_map:
            result[k] = v
            continue

        field_type = type_map[k]

        # Handle basic types
        if v is None:
            result[k] = None
        elif field_type in (int, float, str, bool):
            if field_type == bool and isinstance(v, str):
                result[k] = v.lower() in ('true', 'yes', '1')
            else:
                result[k] = field_type(v)
        # Handle tuples
        elif hasattr(field_type, '__origin__') and field_type.__origin__ == tuple:
            if isinstance(v, (list, tuple)):
                result[k] = tuple(v)
            elif isinstance(v, str):
                result[k] = (v,)
            else:
                result[k] = v
        else:
            result[k] = v

    return result

def override_config(cfg: Config, overrides: dict) -> Config:
    """Apply dictionary overrides to config, ignoring unknown keys."""
    allowed = {f.name for f in fields(Config)}
    unknown = set(overrides) - allowed
    if unknown:
        logging.getLogger(__name__).warning(
            "Ignoring unknown config keys: %s", sorted(unknown)
        )
    known = {k: overrides[k] for k in overrides if k in allowed}
    known = coerce_types(known)
    return replace(cfg, **known)