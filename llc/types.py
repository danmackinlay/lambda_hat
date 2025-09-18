# llc/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class SamplerResult:
    """Raw sampler outputs (no plotting, no ArviZ).
    All arrays are per-chain (ragged allowed) unless noted.
    """

    Ln_histories: List[np.ndarray]  # (Ti,) per chain
    theta_thin: Optional[
        List[np.ndarray] | np.ndarray
    ]  # per chain (Ti, D) or ndarray (C, T, D)
    acceptance: Optional[List[np.ndarray]] = None  # HMC only (Ti,)
    energy: Optional[List[np.ndarray]] = None  # HMC only (Ti,)
    timings: dict = field(default_factory=dict)  # {'warmup': float, 'sampling': float}
    work: dict = field(default_factory=dict)  # grad counts, steps, etc.


@dataclass
class RunStats:
    """Statistics tracking for computational work and timing"""

    # wall-clock
    t_build: float = 0.0
    t_train: float = 0.0
    t_sgld_warmup: float = 0.0
    t_sgld_sampling: float = 0.0
    t_hmc_warmup: float = 0.0
    t_hmc_sampling: float = 0.0
    t_mclmc_warmup: float = 0.0
    t_mclmc_sampling: float = 0.0
    t_sghmc_warmup: float = 0.0
    t_sghmc_sampling: float = 0.0

    # work counters (proxy for computational work)
    # Count "gradient-equivalent" evaluations to compare samplers.
    # - SGLD: ~1 minibatch gradient per step -> +1
    # - HMC: ~num_integration_steps gradients per draw (leapfrog) -> +L
    # - SGHMC: ~1 minibatch gradient per step (like SGLD) -> +1
    # - Add full-data loss evals and log-prob grads as separate counters for transparency.
    n_sgld_minibatch_grads: int = 0
    n_sgld_full_loss: int = 0
    n_hmc_leapfrog_grads: int = 0
    n_hmc_full_loss: int = 0
    n_hmc_warmup_leapfrog_grads: int = 0  # estimated during adaptation
    n_mclmc_steps: int = 0
    n_mclmc_full_loss: int = 0
    n_sghmc_minibatch_grads: int = 0
    n_sghmc_full_loss: int = 0


@dataclass
class RunOutputs:
    """Results from running one complete experiment"""

    run_dir: Optional[str]
    metrics: Dict[str, Any]
    histories: Dict[str, Any]  # {"sgld": [...], "hmc": [...], "mclmc": [...]}
    L0: float
