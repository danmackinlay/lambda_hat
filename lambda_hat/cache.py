import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# map sampler -> prefixes to strip for inactive samplers
_SAMPLER_FIELDS = {"sgld":("sgld_",), "hmc":("hmc_",), "mclmc":("mclmc_",)}

def _strip_irrelevant(d: dict) -> dict:
    samplers = list(d.get("samplers") or [])
    if len(samplers) != 1: return d
    active = samplers[0]
    drop = tuple(p for s,ps in _SAMPLER_FIELDS.items() if s!=active for p in ps)
    return {k:v for k,v in d.items() if not any(k.startswith(p) for p in drop)}

def _normalize(cfg) -> dict:
    d = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    for k in ["runs_dir","save_plots","show_plots"]: d.pop(k, None)
    return _strip_irrelevant(d)

def _code_version() -> str:
    v = os.environ.get("LAMBDA_HAT_CODE_VERSION")
    if v: return str(v)
    # simple filesha of llc/*
    h = hashlib.sha1()
    for root,_,files in os.walk("lambda_hat"):
        for f in sorted(files):
            if f.endswith(".py"):
                h.update(open(os.path.join(root,f),"rb").read())
    return f"filesha-{h.hexdigest()[:12]}"

def run_id(cfg) -> str:
    payload = json.dumps({"cfg": _normalize(cfg), "code": _code_version()}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]

def run_family_id(cfg) -> str:
    d = _normalize(cfg)
    d.pop("samplers", None)
    for ps in _SAMPLER_FIELDS.values():
        for p in ps:
            for k in list(d):
                if k.startswith(p): d.pop(k)
    return hashlib.sha1(json.dumps({"cfg":d}, sort_keys=True).encode()).hexdigest()[:12]

def load_cached_outputs(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load cached outputs from a run directory."""
    metrics_path = Path(run_dir) / "metrics.json"
    l0_path = Path(run_dir) / "L0.txt"

    if not metrics_path.exists():
        return None

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        L0 = 0.0
        if l0_path.exists():
            with open(l0_path, "r") as f:
                L0 = float(f.read().strip())

        return {
            "metrics": metrics,
            "L0": L0,
            "run_dir": run_dir,
            "cached": True,
        }
    except Exception:
        return None