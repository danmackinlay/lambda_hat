# llc/cache.py
"""
Deterministic run caching to avoid re-running identical experiments.
Uses hash of normalized config + code version to generate run IDs.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional
from pathlib import Path


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
        "artifacts_dir",  # Location doesn't affect results
        "diag_mode",  # Diagnostics don't affect LLC computation
        "progress_update_every",
    ]

    for k in volatile_keys:
        d.pop(k, None)

    return d


def _code_version() -> str:
    """
    Code version fingerprint.
    Prefer git (rev + dirty). If git is unavailable (e.g. Modal runtime),
    fall back to hashing the contents of the local source files.
    Can be overridden by LLC_CODE_VERSION env var.
    """
    # 0) explicit override
    env_override = os.environ.get("LLC_CODE_VERSION")
    if env_override:
        return env_override

    # 1) try git
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return f"{rev[:12]}{'+dirty' if dirty else ''}"
    except Exception:
        pass

    # 2) fallback: hash source files (stable across machines)
    try:
        import glob
        paths = (
            glob.glob("llc/**/*.py", recursive=True)
            + glob.glob("llc/*.py")
            + ["pyproject.toml"]
        )
        h = hashlib.sha1()
        for p in sorted(set(filter(os.path.exists, paths))):
            with open(p, "rb") as f:
                h.update(f.read())
        return f"filesha-{h.hexdigest()[:12]}"
    except Exception:
        return "unknown"


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
        print(f"Warning: Failed to load cache from {run_dir}: {e}")
        return None


def should_skip(
    cfg, artifacts_dir: str = "artifacts"
) -> tuple[bool, str, Optional[Dict]]:
    """
    Check if a run should be skipped based on existing cache.

    Args:
        cfg: Configuration object
        artifacts_dir: Base directory for artifacts

    Returns:
        Tuple of (should_skip, run_dir, cached_outputs)
    """
    rid = run_id(cfg)
    run_dir = os.path.join(artifacts_dir, rid)

    # Check if results already exist
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        cached = load_cached_outputs(run_dir)
        if cached:
            return True, run_dir, cached

    return False, run_dir, None
