# llc/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class SamplerResult:
    """Raw sampler outputs (no plotting, no ArviZ).
    All arrays are per-chain (ragged allowed) unless noted.
    """
    Ln_histories: List[np.ndarray]                       # (Ti,) per chain
    theta_thin: Optional[List[np.ndarray] | np.ndarray]  # per chain (Ti, D) or ndarray (C, T, D)
    acceptance: Optional[List[np.ndarray]] = None        # HMC only (Ti,)
    energy: Optional[List[np.ndarray]] = None            # HMC only (Ti,)
    timings: dict = field(default_factory=dict)          # {'warmup': float, 'sampling': float}
    work: dict = field(default_factory=dict)             # grad counts, steps, etc.