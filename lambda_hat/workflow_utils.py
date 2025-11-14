"""Workflow utilities for target/run ID computation and config composition.

Extracted from Snakefile to support both Snakemake and Parsl workflows.
"""

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

# Register OmegaConf resolvers
from lambda_hat import omegaconf_support  # noqa: F401


def _fingerprint_payload_build(cfg):
    """Extract semantic fingerprint from build config for target ID.

    Args:
        cfg: OmegaConf DictConfig for target building

    Returns:
        dict: Resolved config with non-semantic fields removed
    """
    c = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    # Drop non-semantic / unstable fields
    for k in ["runtime", "store"]:
        c.pop(k, None)
    return c


def target_id_for(cfg):
    """Compute content-addressed target ID from build config.

    Args:
        cfg: OmegaConf DictConfig for target building

    Returns:
        str: Target ID with format 'tgt_<12-char-hash>'
    """
    blob = json.dumps(_fingerprint_payload_build(cfg), sort_keys=True)
    return "tgt_" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def run_id_for(cfg):
    """Compute content-addressed run ID from sample config.

    Args:
        cfg: OmegaConf DictConfig for sampling run

    Returns:
        str: Run ID (8-char SHA1 hash)
    """
    blob = json.dumps(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def compose_build_cfg(t, conf_dir=None, store_root="runs", jax_enable_x64=True):
    """Compose full build config from target specification.

    Args:
        t: Target spec dict with keys {model, data, teacher?, seed, overrides?}
        conf_dir: Path to lambda_hat/conf directory (default: auto-detect)
        store_root: Root directory for artifacts (default: "runs")
        jax_enable_x64: Enable 64-bit precision (default: True)

    Returns:
        OmegaConf DictConfig: Fully composed build configuration
    """
    if conf_dir is None:
        conf_dir = Path(__file__).parent / "conf"
    else:
        conf_dir = Path(conf_dir)

    cfg = OmegaConf.load(conf_dir / "workflow.yaml")
    cfg = OmegaConf.merge(
        cfg,
        {"model": OmegaConf.load(conf_dir / "model" / f"{t['model']}.yaml")},
        {"data": OmegaConf.load(conf_dir / "data" / f"{t['data']}.yaml")},
        {"teacher": OmegaConf.load(conf_dir / "teacher" / f"{t.get('teacher', '_null')}.yaml")},
        {
            "target": {"seed": t["seed"]},
            "jax": {"enable_x64": jax_enable_x64},
            "store": {"root": store_root},
        },
    )
    if "overrides" in t:
        cfg = OmegaConf.merge(cfg, t["overrides"])
    return cfg


def compose_sample_cfg(tid, s, conf_dir=None, store_root="runs", jax_enable_x64=True):
    """Compose full sampling config from target ID and sampler specification.

    Args:
        tid: Target ID string (e.g., 'tgt_abc123')
        s: Sampler spec dict with keys {name, seed?, overrides?}
        conf_dir: Path to lambda_hat/conf directory (default: auto-detect)
        store_root: Root directory for artifacts (default: "runs")
        jax_enable_x64: Enable 64-bit precision (default: True)

    Returns:
        OmegaConf DictConfig: Fully composed sampling configuration
    """
    if conf_dir is None:
        conf_dir = Path(__file__).parent / "conf"
    else:
        conf_dir = Path(conf_dir)

    base = OmegaConf.load(conf_dir / "sample" / "base.yaml")
    smpl = OmegaConf.load(conf_dir / "sample" / "sampler" / f"{s['name']}.yaml")
    cfg = OmegaConf.merge(
        base,
        {"sampler": smpl},
        {
            "target_id": tid,
            "jax": {"enable_x64": jax_enable_x64},
            "store": {"root": store_root},
            "runtime": {"seed": s.get("seed", 12345)},
        },
    )
    if "overrides" in s:
        cfg = OmegaConf.merge(cfg, {"sampler": {s["name"]: s["overrides"]}})
    return cfg
