# llc/cache.py
"""
Deterministic run caching to avoid re-running identical experiments.
Uses hash of normalized config + code version to generate run IDs.
"""

import hashlib
import json
import os
import glob
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional
from pathlib import Path

# Registry of sampler-specific config field prefixes
_SAMPLER_FIELDS: Dict[str, tuple[str, ...]] = {
    "sgld": ("sgld_",),
    "sghmc": ("sghmc_",),
    "hmc": ("hmc_",),
    "mclmc": ("mclmc_",),
}


def _strip_irrelevant_sampler_fields(d: dict) -> dict:
    """
    Return a shallow dict copy with sampler-specific fields for *inactive* samplers removed.
    If cfg.samplers is missing/ambiguous, leave dict unchanged (fail-safe).
    """
    samplers = list(d.get("samplers") or [])
    if len(samplers) != 1:
        return d
    active = samplers[0]
    if active not in _SAMPLER_FIELDS:
        return d
    inactive_prefixes = tuple(
        p for s, prefixes in _SAMPLER_FIELDS.items() if s != active for p in prefixes
    )
    if not inactive_prefixes:
        return d
    # Do not mutate original
    out = {}
    for k, v in d.items():
        if any(k.startswith(p) for p in inactive_prefixes):
            continue
        out[k] = v
    return out


def _normalize_cfg(cfg) -> Dict[str, Any]:
    """
    Normalize config for hashing by removing volatile non-mathematical flags.
    These flags don't affect the computation results.
    """
    d = asdict(cfg) if is_dataclass(cfg) else dict(cfg)

    # Remove flags that don't change mathematical results
    # Note: cache_salt is NOT in this list - we want it to affect the cache key
    volatile_keys = [
        "use_tqdm",
        "auto_create_run_dir",
        "save_plots",
        "save_manifest",
        "save_readme_snippet",
        "runs_dir",  # Location doesn't affect results
        "diag_mode",  # Diagnostics don't affect LLC computation
        "progress_update_every",
    ]

    for k in volatile_keys:
        d.pop(k, None)

    # NEW: keep cache key scoped to the active sampler by dropping other samplers' fields
    d = _strip_irrelevant_sampler_fields(d)

    return d


def _code_version() -> str:
    """Return a code fingerprint:
    1) if LLC_CODE_VERSION env is set, use it (for CI/Modal);
    2) else hash local source files so code changes invalidate cache automatically."""
    v = os.environ.get("LLC_CODE_VERSION")
    if v:
        return str(v)

    paths = (
        glob.glob("llc/**/*.py", recursive=True)
        + glob.glob("llc/*.py")
        + ["pyproject.toml"]
    )
    h = hashlib.sha1()
    for p in sorted({p for p in paths if os.path.exists(p)}):
        with open(p, "rb") as f:
            h.update(f.read())
    return f"filesha-{h.hexdigest()[:12]}"


def run_id(cfg) -> str:
    """
    Generate deterministic run ID from config and code version.

    Args:
        cfg: Configuration object (Config dataclass or dict)

    Returns:
        12-character hex string uniquely identifying this run
    """
    payload = json.dumps(
        {"cfg": _normalize_cfg(cfg), "code": _code_version()},
        sort_keys=True,
        default=str,  # Handle any non-JSON types like numpy
    )

    # Use SHA1 hash (sufficient for cache key purposes)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def run_family_id(cfg) -> str:
    """
    Group runs by problem/data/model/seed.
    Excludes sampler choice and *all* sampler-specific knobs.
    Excludes code version to allow cross-version comparison within a family.

    Args:
        cfg: Configuration object (Config dataclass or dict)

    Returns:
        12-character hex string identifying the family of runs
    """
    d = _normalize_cfg(cfg)
    # Drop sampler selection entirely
    d.pop("samplers", None)
    # Drop all sampler-specific fields regardless of active sampler
    for prefixes in _SAMPLER_FIELDS.values():
        for p in prefixes:
            for k in list(d.keys()):
                if k.startswith(p):
                    d.pop(k, None)
    payload = {"cfg": d}
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def load_cached_outputs(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load cached outputs from a run directory.

    Args:
        run_dir: Path to the run directory

    Returns:
        Dict with metrics and metadata if found, None otherwise
    """
    metrics_path = Path(run_dir) / "metrics.json"
    Path(run_dir) / "config.json"
    l0_path = Path(run_dir) / "L0.txt"

    if not metrics_path.exists():
        return None

    try:
        # Load metrics
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Load L0 if available
        L0 = 0.0
        if l0_path.exists():
            with open(l0_path, "r") as f:
                L0 = float(f.read().strip())

        # Return minimal structure matching RunOutputs
        return {
            "metrics": metrics,
            "L0": L0,
            "run_dir": run_dir,
            "cached": True,  # Flag to indicate this was loaded from cache
        }
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load cache from {run_dir}: {e}")
        return None


def should_skip(cfg, runs_dir: str = "runs") -> tuple[bool, str, Optional[Dict]]:
    """
    Check if a run should be skipped based on existing cache.

    Args:
        cfg: Configuration object
        runs_dir: Base directory for run outputs

    Returns:
        Tuple of (should_skip, run_dir, cached_outputs)
    """
    rid = run_id(cfg)
    run_dir = os.path.join(runs_dir, rid)

    # Check if results already exist
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        cached = load_cached_outputs(run_dir)
        if cached:
            return True, run_dir, cached

    return False, run_dir, None
