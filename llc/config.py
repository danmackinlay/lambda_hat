# llc/config.py
"""Configuration classes and default settings"""

from dataclasses import dataclass
from typing import Optional, Literal, List


@dataclass
class Config:
    # ---- Target model selection ----
    # "mlp": current neural network target (default)
    # "quadratic": analytical diagnostic L_n(θ) = 0.5 ||θ||^2 (ignores data)
    target: Literal["mlp", "quadratic"] = "mlp"
    # For 'quadratic', you can set the parameter dimension here; if None we fall
    # back to target_params (else in_dim).
    quad_dim: Optional[int] = None

    # Model architecture
    in_dim: int = 32
    out_dim: int = 1
    depth: int = 1  # number of hidden layers
    widths: Optional[List[int]] = None  # per-layer widths; if None, auto-infer
    activation: Literal["relu", "tanh", "gelu", "identity"] = "relu"
    bias: bool = True
    skip_connections: bool = False
    residual_period: int = 2  # every k layers add skip if enabled
    layernorm: bool = False  # (default False; can destabilize HMC)
    init: Literal["he", "xavier", "lecun", "orthogonal"] = "he"

    # Size control
    target_params: Optional[int] = 10_000  # if provided, fix total d and infer widths
    # keep old 'hidden' for backward compatibility
    hidden: int = 300  # used only if target_params=None and widths=None

    # Data
    n_data: int = 20_000
    x_dist: Literal[
        "gauss_iso", "gauss_aniso", "mixture", "lowdim_manifold", "heavy_tail"
    ] = "gauss_iso"
    cov_decay: float = 0.95  # for anisotropy: eigvals ~ cov_decay**i
    mixture_k: int = 4
    mixture_spread: float = 2.0
    x_dim_latent: int = 2  # for low-dim manifold
    noise_model: Literal["gauss", "hetero", "student_t", "outliers"] = "gauss"
    noise_scale: float = 0.1
    hetero_scale: float = 0.1
    student_df: float = 4.0
    outlier_frac: float = 0.05
    outlier_scale: float = 2.0

    # Teacher (can differ from student)
    teacher_depth: Optional[int] = None
    teacher_widths: Optional[List[int]] = None
    teacher_activation: Optional[str] = None
    teacher_dropout_rate: float = 0.0  # stochastic teacher if >0 (only during data gen)

    # Loss / likelihood
    loss: Literal["mse", "t_regression"] = "mse"

    # Local posterior (tempering + prior)
    beta_mode: Literal["1_over_log_n", "fixed"] = "1_over_log_n"
    beta0: float = 1.0
    prior_radius: Optional[float] = None  # if set, gamma = d / prior_radius**2
    gamma: float = 1.0  # used only if prior_radius None
    prior_center: Optional[List[float]] = None  # prior mean (if None, use zeros)
    reference_for_L0: Optional[float] = None  # reference value for L0 baseline

    # Sampling
    # Choose which samplers to run. Order controls reporting & gallery grouping.
    samplers: list[str] = ("sgld", "hmc", "mclmc")
    chains: int = 4
    # Optimization: use batched (vmap+scan) chain execution for speed
    use_batched_chains: bool = False

    # SGLD
    sgld_steps: int = 4_000
    sgld_warmup: int = 1_000
    sgld_batch_size: int = 256
    sgld_step_size: float = 1e-6
    sgld_thin: int = 20  # store every k-th draw for diagnostics only
    sgld_eval_every: int = 10  # compute full-data L_n(w) every k steps (for LLC mean)
    sgld_dtype: str = "float32"  # reduce memory
    # SGLD preconditioning (optional)
    sgld_precond: Literal["none", "rmsprop", "adam"] = "none"
    sgld_beta1: float = 0.9  # Adam first-moment EMA
    sgld_beta2: float = 0.999  # RMSProp/Adam second-moment EMA
    sgld_eps: float = 1e-8  # numerical stabilizer in preconditioner
    sgld_bias_correction: bool = True  # Adam bias correction on/off

    # SGHMC (Stochastic Gradient Hamiltonian Monte Carlo)
    sghmc_steps: int = 4_000
    sghmc_warmup: int = 1_000
    sghmc_batch_size: int = 256
    sghmc_step_size: float = 1e-6
    sghmc_temperature: float = 1.0  # temperature parameter for SGHMC
    sghmc_thin: int = 20  # store every k-th draw for diagnostics only
    sghmc_eval_every: int = 10  # compute full-data L_n(w) every k steps
    sghmc_dtype: str = "float32"  # reduce memory

    # HMC
    hmc_draws: int = 1_000
    hmc_warmup: int = 1_000
    hmc_num_integration_steps: int = 10
    hmc_thin: int = 5  # store every k-th draw for diagnostics
    hmc_eval_every: int = 1  # compute L_n(w) every k draws (usually 1 for HMC)
    hmc_dtype: str = "float64"

    # MCLMC (unadjusted)
    mclmc_draws: int = 2_000  # post-tuning steps (MCLMC yields 1 sample per step)
    mclmc_eval_every: int = 1
    mclmc_thin: int = 10
    mclmc_dtype: str = "float64"  # keep f64 for stability (like HMC)

    # MCLMC tuning
    mclmc_tune_steps: int = 2_000  # steps used by the automatic tuner
    mclmc_diagonal_preconditioning: bool = False
    mclmc_desired_energy_var: float = 5e-4  # target EEV (per Sampling Book)
    mclmc_integrator: Literal[
        "isokinetic_mclachlan",
        "isokinetic_velocity_verlet",
        "isokinetic_yoshida",
        "isokinetic_omelyan",
    ] = "isokinetic_mclachlan"

    # (optional) adjusted MCLMC
    mclmc_adjusted: bool = False
    mclmc_adjusted_target_accept: float = 0.90  # per docs' guidance
    mclmc_grad_per_step_override: Optional[float] = None  # work accounting calibration

    # Misc
    seed: int = 42
    use_tqdm: bool = True
    progress_update_every: int = 50  # step/draw interval for bar postfix refresh
    profile_adaptation: bool = True  # time warmup/adaptation separately

    # Diagnostics and plotting
    diag_mode: Literal["none", "subset", "proj"] = (
        "proj"  # default: tiny random projections
    )
    diag_k: int = 16  # number of dimensions/projections to track
    diag_seed: int = 1234  # seed for dimension selection/projections
    max_theta_plot_dims: int = 8  # cap for plotting even if k is larger
    save_plots_prefix: Optional[str] = None  # e.g., "diag" to save PNGs

    # Artifacts and visualization saving
    artifacts_dir: str = "artifacts"  # base directory for saving artifacts
    save_plots: bool = False  # whether to save all diagnostic plots
    show_plots: bool = False  # whether to display plots (default: headless)
    auto_create_run_dir: bool = True  # create timestamped run directories
    save_manifest: bool = True  # save run configuration to manifest.txt
    save_readme_snippet: bool = True  # generate README_snippet.md
    auto_update_readme: bool = False  # auto-update README with markers (optional)


# Small test config for quick verification
TEST_CFG = Config(
    # Small model
    in_dim=4,
    out_dim=1,
    target_params=50,  # ~50 params only
    # Small data
    n_data=100,
    # Minimal sampling
    chains=2,
    sgld_steps=100,
    sgld_warmup=20,
    sgld_eval_every=5,
    sgld_thin=10,
    # Minimal SGHMC
    sghmc_steps=100,
    sghmc_warmup=20,
    sghmc_eval_every=5,
    sghmc_thin=10,
    # Minimal HMC
    hmc_draws=50,
    hmc_warmup=20,
    hmc_thin=5,
    # Minimal MCLMC
    mclmc_draws=80,
    mclmc_tune_steps=100,
    mclmc_thin=8,
    # Enable headless plot saving for testing
    save_plots=True,
    show_plots=False,
    save_manifest=True,
    save_readme_snippet=True,
)

CFG = Config()  # Default full config
