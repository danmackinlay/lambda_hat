# lambda_hat/config.py
"""Lightweight config helpers for OmegaConf-only."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Neural network architecture configuration"""

    in_dim: int = 32
    out_dim: int = 1
    depth: int = 3
    widths: Optional[List[int]] = None
    activation: str = "relu"
    bias: bool = True
    layernorm: bool = False
    init: str = "he"
    target_params: Optional[int] = 10_000
    hidden: int = 300  # fallback if target_params and widths are None
    # Residual/skip connections
    skip_connections: bool = False
    residual_period: int = 2


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


def validate_teacher_cfg(tcfg: dict):
    """Validate teacher config contains only allowed keys."""
    allowed = {
        "depth",
        "widths",
        "activation",
        "dropout_rate",
        "target_params",
        "hidden",
    }
    extra = set(tcfg.keys()) - allowed
    if extra:
        raise ValueError(f"Disallowed teacher keys: {sorted(extra)}")

    # Optionally enforce at most one size driver
    size_drivers = [tcfg.get("target_params"), tcfg.get("hidden")]
    if sum(x is not None for x in size_drivers) > 1:
        raise ValueError("Teacher config cannot specify both target_params and hidden")


@dataclass
class TeacherConfig:
    """Teacher network configuration"""

    depth: Optional[int] = None
    widths: Optional[List[int]] = None
    dropout_rate: float = 0.0
    activation: str = "relu"
    target_params: Optional[int] = None
    hidden: Optional[int] = None


@dataclass
class TrainingConfig:
    """ERM training configuration"""

    optimizer: str = "adam"
    learning_rate: float = 0.001
    steps: int = 5000
    early_stop_tol: Optional[float] = 1e-6


@dataclass
class PosteriorConfig:
    """Local posterior configuration"""

    loss: str = "mse"
    beta_mode: str = "1_over_log_n"
    beta0: float = 1.0
    prior_radius: Optional[float] = None
    gamma: float = 1.0


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
    eval_every: int = 100


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
    integrator: str = "isokinetic_mclachlan"


@dataclass
class VIConfig:
    """Variational Inference sampler configuration"""

    M: int = 8  # number of mixture components
    r: int = 2  # rank budget per component
    steps: int = 5_000  # optimization steps
    batch_size: int = 256  # minibatch size
    lr: float = 0.01  # learning rate
    eval_every: int = 50  # how often to record metrics
    gamma: float = 0.001  # localizer strength (may be overridden by posterior config)
    eval_samples: int = 64  # MC samples for final LLC estimate
    dtype: str = "float32"  # precision: "float32" or "float64"
    use_whitening: bool = True  # enable geometry whitening


@dataclass
class SamplerConfig:
    """Combined sampler configuration"""

    chains: int = 4
    sgld: SGLDConfig = field(default_factory=SGLDConfig)
    hmc: HMCConfig = field(default_factory=HMCConfig)
    mclmc: MCLMCConfig = field(default_factory=MCLMCConfig)
    vi: VIConfig = field(default_factory=VIConfig)


@dataclass
class OutputConfig:
    """Output and visualization configuration"""

    save_plots: bool = True
    show_plots: bool = False


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
