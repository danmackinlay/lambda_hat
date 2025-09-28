from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any

import jax.numpy as jnp
import numpy as np
from hashlib import sha256

# ---------- Target metadata ----------


@dataclass(frozen=True)
class TargetMeta:
    target_id: str
    created_at: float
    code_sha: str
    jax_enable_x64: bool
    pkg_versions: Dict[str, str]
    seed: int
    model_cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    training_cfg: Dict[str, Any]
    teacher_cfg: Dict[str, Any] | None
    dims: Dict[str, int]  # e.g. {"n": 100, "d": 4, "p": 45}
    hashes: Dict[str, str]  # e.g. {"theta": "sha256:..."}
    hostname: str
    metrics: Dict[str, float] = field(default_factory=dict)  # e.g. {"L0": 0.123}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------- Serialization for Haiku params (Flat NPZ) ----------

# Use '::' as a separator, as it's highly unlikely in module/param names.
_NPZ_SEP = "::"


def _flatten_params_dict(params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Converts Haiku params {'module': {'param': arr}} to flat NPZ format."""
    flat = {}
    # params is {'module_name': {'param_name': array}, ...}
    for module_name, module_params in params.items():
        # Ensure module_params is a dictionary (Haiku invariant for parameters)
        if not isinstance(module_params, dict):
            continue

        for param_name, param_value in module_params.items():
            # Basic check that the leaf is an array-like object
            if hasattr(param_value, "shape"):
                key = f"{module_name}{_NPZ_SEP}{param_name}"
                flat[key] = np.asarray(param_value)  # move to host np
            else:
                print(
                    f"Warning: Skipping non-array parameter during save: {module_name}/{param_name}"
                )
    return flat


def _unflatten_params_dict(flat: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Converts flat NPZ format back to Haiku params structure."""
    params: Dict[str, Dict[str, Any]] = {}
    # flat is {'module_name::param_name': array, ...}
    for key, value in flat.items():
        if _NPZ_SEP not in key:
            # Since backwards compatibility is not required, we fail fast on old formats.
            raise ValueError(
                f"Invalid key format in NPZ: {key}. Expected 'module::param'. Artifact may be legacy format."
            )

        module_name, param_name = key.split(_NPZ_SEP, 1)
        if module_name not in params:
            params[module_name] = {}
        params[module_name][param_name] = jnp.asarray(value)  # Move back to JAX array
    return params


def _hash_arrays(flat: Dict[str, np.ndarray]) -> str:
    # Order by key for deterministic hashing.
    items = [(k, flat[k]) for k in sorted(flat)]
    # Hash concatenated bytes with shapes/dtypes so collisions are unlikely.

    h = sha256()
    for k, arr in items:
        h.update(k.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes(order="C"))
    return "sha256:" + h.hexdigest()


# ---------- Public API ----------


# New explicit saver that does NOT reconstruct paths
def save_target_artifact_explicit(out_dir, X, Y, params0, meta):
    """Save target artifact to explicit directory (used by new Snakemake entrypoints)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Arrays
    np.savez_compressed(out_dir / "data.npz", X=np.asarray(X), Y=np.asarray(Y))

    # Params
    flat = _flatten_params_dict(params0)
    np.savez_compressed(out_dir / "params.npz", **flat)

    # Meta
    with open(out_dir / "meta.json", "w") as f:
        json.dump(asdict(meta), f, indent=2, sort_keys=True)

    return out_dir


# Optional explicit loader for target dir
def load_target_artifact_from_dir(target_dir):
    """Load target artifact directly from target directory."""
    target_dir = Path(target_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    data = np.load(target_dir / "data.npz")
    X, Y = data["X"], data["Y"]

    pz = np.load(target_dir / "params.npz")
    flat = {k: pz[k] for k in pz.files}
    params = _unflatten_params_dict(flat)

    with open(target_dir / "meta.json") as f:
        meta = json.load(f)

    return X, Y, params, meta, target_dir


def save_target_artifact(
    root: str | Path,
    target_id: str,
    X: np.ndarray,
    Y: np.ndarray,
    params: Dict[str, Any],
    meta: TargetMeta,
) -> Path:
    """Writes X, Y, params, meta under runs/targets/<target_id>."""
    tdir = Path(root) / "targets" / target_id
    _ensure_dir(tdir)

    # Arrays
    np.savez_compressed(tdir / "data.npz", X=np.asarray(X), Y=np.asarray(Y))

    # Params
    flat = _flatten_params_dict(params)
    # Store as a single NPZ; keys are parameter paths.
    np.savez_compressed(tdir / "params.npz", **flat)

    # Meta
    with open(tdir / "meta.json", "w") as f:
        json.dump(asdict(meta), f, indent=2, sort_keys=True)

    # Catalog append
    _append_catalog(Path(root), meta)

    return tdir


def load_target_artifact(root: str | Path, target_id_or_path: str | Path):
    """Returns (X, Y, params, meta_dict, tdir Path)."""
    # Resolve id/path
    tdir = Path(target_id_or_path)
    if not tdir.exists():
        tdir = Path(root) / "targets" / str(target_id_or_path)
    if not tdir.exists():
        raise FileNotFoundError(f"Target not found: {target_id_or_path}")

    data = np.load(tdir / "data.npz")
    X, Y = data["X"], data["Y"]

    pz = np.load(tdir / "params.npz")
    flat = {k: pz[k] for k in pz.files}
    params = _unflatten_params_dict(flat)

    with open(tdir / "meta.json") as f:
        meta = json.load(f)

    return X, Y, params, meta, tdir


def _append_catalog(root: Path, meta: TargetMeta):
    cat = root / "targets" / "_catalog.jsonl"
    _ensure_dir(cat.parent)
    with open(cat, "a") as f:
        f.write(json.dumps(asdict(meta), sort_keys=True) + "\n")


# ---- Sample manifest per target ----


def append_sample_manifest(
    root: str | Path,
    target_id: str,
    record: Dict[str, Any],
) -> None:
    """Append a line to runs/targets/<target_id>/_runs.jsonl."""
    tdir = Path(root) / "targets" / target_id
    _ensure_dir(tdir)
    with open(tdir / "_runs.jsonl", "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
