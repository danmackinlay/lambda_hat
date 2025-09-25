# llc/config.py
"""Structured configuration using Hydra and dataclasses"""

from dataclasses import dataclass, field
from typing import Optional, List
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    """Neural network architecture configuration"""

    in_dim: int = 32
    out_dim: int = 1
    depth: int = 3
    widths: Optional[List[int]] = None
    activation: str = "relu"
    bias: bool = True
    skip_connections: bool = False
    residual_period: int = 2
    layernorm: bool = False
    init: str = "he"
    target_params: Optional[int] = 10_000
    hidden: int = 300  # fallback if target_params and widths are None


@dataclass
class DataConfig:
    """Data generation configuration"""

    n_data: int = 20_000
    x_dist: str = "gauss_iso"
    cov_decay: float = 0.95
    mixture_k: int = 4
    mixture_spread: float = 2.0
    x_dim_latent: int = 2
    noise_model: str = "gauss"
    noise_scale: float = 0.1
    hetero_scale: float = 0.1
    student_df: float = 4.0
    outlier_frac: float = 0.05
    outlier_scale: float = 2.0


@dataclass
class TeacherConfig:
    """Teacher network configuration"""

    depth: Optional[int] = None
    widths: Optional[List[int]] = None
    activation: Optional[str] = None
    dropout_rate: float = 0.0


@dataclass
class TrainingConfig:
    """ERM training configuration"""

    optimizer: str = "adam"
    learning_rate: float = 0.001
    erm_steps: int = 5000
    early_stop_tol: Optional[float] = 1e-6


@dataclass
class PosteriorConfig:
    """Local posterior configuration"""

    loss: str = "mse"
    beta_mode: str = "1_over_log_n"
    beta0: float = 1.0
    prior_radius: Optional[float] = None
    gamma: float = 1.0
    # prior_center field removed - unused vestigial parameter


@dataclass
class SGLDConfig:
    """SGLD sampler configuration"""

    steps: int = 16_000
    warmup: int = 1_000
    batch_size: int = 256
    step_size: float = 1e-6
    dtype: str = "float32"
    precond: str = "none"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    bias_correction: bool = True
    eval_every: int = 10  # Record loss every N steps


@dataclass
class HMCConfig:
    """HMC sampler configuration"""

    draws: int = 5_000
    warmup: int = 1_000
    num_integration_steps: int = 10
    step_size: float = 0.01
    dtype: str = "float64"
    adapt_step_size: bool = True
    target_acceptance: float = 0.8


@dataclass
class MCLMCConfig:
    """MCLMC sampler configuration"""

    draws: int = 8_000
    L: float = 1.0
    step_size: float = 0.1
    dtype: str = "float64"
    diagonal_preconditioning: bool = False
    # Tuning parameters
    num_steps: int = 2_000
    frac_tune1: float = 0.1
    frac_tune2: float = 0.1
    frac_tune3: float = 0.1
    desired_energy_var: float = 5e-4
    trust_in_estimate: float = 1.0
    num_effective_samples: float = 150.0
    integrator: str = "isokinetic_mclachlan"


@dataclass
class SamplerConfig:
    """Combined sampler configuration"""

    chains: int = 4
    sgld: SGLDConfig = field(default_factory=SGLDConfig)
    hmc: HMCConfig = field(default_factory=HMCConfig)
    mclmc: MCLMCConfig = field(default_factory=MCLMCConfig)


@dataclass
class OutputConfig:
    """Output and visualization configuration (bloat removed)"""

    save_plots: bool = True
    show_plots: bool = False
    # Removed unused fields: max_theta_plot_dims, diag_mode, diag_k, diag_seed, record_Ln_every, record_theta_every


@dataclass
class Config:
    """Main configuration combining all components"""

    # Target selection
    target: str = "mlp"  # "mlp" or "quadratic"
    quad_dim: Optional[int] = None

    # Components
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Misc
    seed: int = 42
    use_tqdm: bool = True
    progress_update_every: int = 50
    profile_adaptation: bool = True


def setup_config():
    """Register configuration schemas with Hydra's ConfigStore"""
    cs = ConfigStore.instance()

    # Register main config schema. Renaming avoids collision with config.yaml.
    cs.store(name="base_config", node=Config)

    # REMOVE ALL OTHER cs.store() CALLS BELOW.
    # Hydra validates YAML presets automatically using the type hints
    # defined in the Config dataclass (e.g., Config.model is ModelConfig).
