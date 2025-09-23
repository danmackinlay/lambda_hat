from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, List, Iterable, Any, Iterator, Tuple

from llc.config import Config


@dataclass
class ProblemVariant:
    name: str
    overrides: Dict[str, Any]


@dataclass
class SamplerVariant:
    name: str  # "sgld" | "sghmc" | "hmc" | "mclmc"
    overrides: Dict[str, Any]  # e.g. {"sgld_precond": "adam", "sgld_step_size": 1e-6}


def expand_matrix(
    base_cfg: Config,
    problems: List[ProblemVariant],
    samplers: List[SamplerVariant],
    seeds: Iterable[int],
) -> Iterator[Tuple[str, str, int, Config]]:
    """
    Cross product of problem × sampler × seed.
    Yields (problem_name, sampler_name, seed, cfg) with **exactly one sampler** in cfg.
    """
    for p in problems:
        for s in samplers:
            for seed in seeds:
                cfg = replace(
                    base_cfg, seed=seed, samplers=[s.name], **p.overrides, **s.overrides
                )
                yield p.name, s.name, seed, cfg
