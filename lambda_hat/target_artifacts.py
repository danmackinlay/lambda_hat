from __future__ import annotations

import json, time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np

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
    dims: Dict[str, int]         # e.g. {"n": 100, "d": 4, "p": 45}
    hashes: Dict[str, str]       # e.g. {"theta": "sha256:..."}
    hostname: str
    metrics: Dict[str, float] = field(default_factory=dict)    # e.g. {"L0": 0.123}

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------- Simple dict<str, ndarray> serialization for Haiku params ----------

def _flatten_params_dict(params: Dict[str, Any], prefix: str = "") -> Dict[str, np.ndarray]:
    """Flattens a Haiku-style nested dict of arrays into { 'a/b/c': array }."""
    out: Dict[str, np.ndarray] = {}
    for k, v in params.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_flatten_params_dict(v, key))
        else:
            arr = np.asarray(v)  # move to host np
            out[key] = arr
    return out

def _unflatten_params_dict(flat: Dict[str, np.ndarray]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    for k, arr in flat.items():
        cur = root
        parts = k.split("/")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = jnp.asarray(arr)
    return root

def _hash_arrays(flat: Dict[str, np.ndarray]) -> str:
    # Order by key for deterministic hashing.
    items = [(k, flat[k]) for k in sorted(flat)]
    # Hash concatenated bytes with shapes/dtypes so collisions are unlikely.
    from hashlib import sha256
    h = sha256()
    for k, arr in items:
        h.update(k.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes(order="C"))
    return "sha256:" + h.hexdigest()

# ---------- Public API ----------

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
    """Append a line to runs/samples/<target_id>/_index.jsonl."""
    sdir = Path(root) / "samples" / target_id
    _ensure_dir(sdir)
    with open(sdir / "_index.jsonl", "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")