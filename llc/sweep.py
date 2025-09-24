from dataclasses import dataclass, replace
from typing import Dict, Any, Iterable, List, Tuple, Iterator
import json
from .config import Config

@dataclass
class ProblemVariant:
    name: str
    overrides: Dict[str, Any]

@dataclass
class SamplerVariant:
    name: str
    overrides: Dict[str, Any]

@dataclass
class Study:
    base: Config
    problems: List[ProblemVariant]
    samplers: List[SamplerVariant]
    seeds: List[int]

def expand_matrix(base_cfg: Config, problems: List[ProblemVariant],
                  samplers: List[SamplerVariant], seeds: Iterable[int]) -> Iterator[Tuple[str,str,int,Config]]:
    for p in problems:
        for s in samplers:
            for seed in seeds:
                yield p.name, s.name, seed, replace(base_cfg, seed=seed, samplers=(s.name,), **p.overrides, **s.overrides)

def run_sweep(*, base_cfg: Config, problems, samplers, seeds, executor):
    """executor(payload)->result; payloads are atomic one-sampler Configs."""
    items = list(expand_matrix(base_cfg, problems, samplers, seeds))
    cfgs = [cfg for _,_,_,cfg in items]
    return executor(cfgs)  # caller handles concurrency + GPU pinning

def load_study_yaml(path: str) -> Study:
    """Load study YAML file."""
    import yaml
    from .config import apply_preset, override_config

    with open(path) as f:
        doc = yaml.safe_load(f)

    # Build base config
    base_dict = doc.get("base", {})
    preset = base_dict.pop("preset", None)
    base = apply_preset(Config(), preset)
    if base_dict:
        base = override_config(base, base_dict)

    # Parse problems
    problems = [ProblemVariant(**p) for p in doc.get("problems", [])]

    # Parse samplers
    samplers = [SamplerVariant(**s) for s in doc.get("samplers", [])]

    # Parse seeds
    seeds = doc.get("seeds", [0])

    return Study(base=base, problems=problems, samplers=samplers, seeds=seeds)