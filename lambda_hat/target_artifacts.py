from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

import equinox as eqx
import jax
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
    teacher_cfg: Dict[str, Any] | None
    dims: Dict[str, int]  # e.g. {"n": 100, "d": 4, "p": 45}
    hashes: Dict[str, str]  # e.g. {"theta": "sha256:..."}
    hostname: str
    metrics: Dict[str, float] = field(default_factory=dict)  # e.g. {"L0": 0.123}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------- Serialization for Equinox models ----------


def _serialize_model(model: Any, path: Path) -> None:
    """Serialize an Equinox model to disk.

    Args:
        model: Equinox module to serialize
        path: Directory to write model files (params.eqx + static.eqx)
    """
    path.mkdir(parents=True, exist_ok=True)

    # Split into trainable parameters and static structure
    params, static = eqx.partition(model, eqx.is_array)

    # Save parameters (arrays)
    with open(path / "params.eqx", "wb") as f:
        eqx.tree_serialise_leaves(f, params)

    # Save static structure
    with open(path / "static.eqx", "wb") as f:
        eqx.tree_serialise_leaves(f, static)


def _deserialize_model(model_template: Any, path: Path) -> Any:
    """Deserialize an Equinox model from disk.

    Args:
        model_template: Equinox module with correct structure (shapes only)
        path: Directory containing model files (params.eqx + static.eqx)

    Returns:
        Reconstructed Equinox module

    Note:
        Target artifacts are always saved as float32. We temporarily disable
        x64 mode during deserialization to ensure template leaves match the
        on-disk dtype, preventing Equinox dtype validation errors.
    """
    # Ensure template leaves are float32 during deserialization because
    # target params are saved on disk as float32
    x64_prev = jax.config.read("jax_enable_x64")
    try:
        jax.config.update("jax_enable_x64", False)

        # Split template into params and static
        params, static = eqx.partition(model_template, eqx.is_array)

        # Load parameters
        with open(path / "params.eqx", "rb") as f:
            params = eqx.tree_deserialise_leaves(f, params)

        # Load static structure
        with open(path / "static.eqx", "rb") as f:
            static = eqx.tree_deserialise_leaves(f, static)

        # Combine and return
        return eqx.combine(params, static)
    finally:
        jax.config.update("jax_enable_x64", x64_prev)


def _check_legacy_format(path: Path) -> bool:
    """Check if directory contains legacy Haiku format artifacts."""
    npz_params = path / "params.npz"
    eqx_params = path / "params.eqx"

    if npz_params.exists() and not eqx_params.exists():
        return True
    return False


def _hash_model(model: Any) -> str:
    """Hash an Equinox model for deterministic artifact tracking.

    Args:
        model: Equinox module to hash

    Returns:
        SHA256 hash string
    """
    params, _ = eqx.partition(model, eqx.is_array)
    leaves = jax.tree.leaves(params)

    h = sha256()
    for arr in leaves:
        arr_np = np.asarray(arr)
        h.update(str(arr_np.shape).encode())
        h.update(str(arr_np.dtype).encode())
        h.update(arr_np.tobytes(order="C"))

    return "sha256:" + h.hexdigest()


# ---------- Public API ----------


def save_target_artifact_explicit(out_dir, X, Y, params0, meta):
    """Save target artifact to explicit directory.

    Args:
        out_dir: Output directory path
        X: Input data array
        Y: Target data array
        params0: Equinox model (ERM solution)
        meta: TargetMeta metadata object
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Arrays (data)
    np.savez_compressed(out_dir / "data.npz", X=np.asarray(X), Y=np.asarray(Y))

    # Model (Equinox format)
    _serialize_model(params0, out_dir)

    # Metadata
    with open(out_dir / "meta.json", "w") as f:
        json.dump(asdict(meta), f, indent=2, sort_keys=True)

    return out_dir


# Optional explicit loader for target dir
def load_target_artifact_from_dir(target_dir, model_template=None):
    """Load target artifact directly from target directory.

    Args:
        target_dir: Path to target directory
        model_template: Equinox model template for reconstruction (required for new format)

    Returns:
        Tuple of (X, Y, model, meta_dict, target_dir)

    Raises:
        ValueError: If legacy Haiku format detected
        FileNotFoundError: If target directory doesn't exist
    """
    target_dir = Path(target_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    # Check for legacy format
    if _check_legacy_format(target_dir):
        raise ValueError(
            f"Legacy Haiku format detected at {target_dir}. "
            f"This artifact uses the old params.npz format. "
            f"Please rebuild the target with the new Equinox-based code."
        )

    # Load data
    data = np.load(target_dir / "data.npz")
    X, Y = data["X"], data["Y"]

    # Load metadata
    with open(target_dir / "meta.json") as f:
        meta = json.load(f)

    # Load Equinox model
    if model_template is None:
        raise ValueError(
            "model_template is required to load Equinox models. "
            "Build a model with the same architecture from metadata, then pass it here."
        )

    model = _deserialize_model(model_template, target_dir)

    return X, Y, model, meta, target_dir


def save_target_artifact(
    root: str | Path,
    target_id: str,
    X: np.ndarray,
    Y: np.ndarray,
    params: Any,  # Equinox model
    meta: TargetMeta,
) -> Path:
    """Writes X, Y, model, meta under runs/targets/<target_id>.

    Args:
        root: Root directory for artifacts
        target_id: Target identifier
        X: Input data
        Y: Target data
        params: Equinox model (ERM solution)
        meta: Target metadata

    Returns:
        Path to target directory
    """
    tdir = Path(root) / "targets" / target_id
    _ensure_dir(tdir)

    # Arrays (data)
    np.savez_compressed(tdir / "data.npz", X=np.asarray(X), Y=np.asarray(Y))

    # Model (Equinox format)
    _serialize_model(params, tdir)

    # Metadata
    with open(tdir / "meta.json", "w") as f:
        json.dump(asdict(meta), f, indent=2, sort_keys=True)

    # Catalog append
    _append_catalog(Path(root), meta)

    return tdir


def load_target_artifact(root: str | Path, target_id_or_path: str | Path, model_template=None):
    """Load target artifact (data + model + metadata).

    Args:
        root: Root directory for artifacts
        target_id_or_path: Target ID or full path to target directory
        model_template: Equinox model template for reconstruction (required)

    Returns:
        Tuple of (X, Y, model, meta_dict, tdir Path)

    Raises:
        ValueError: If legacy Haiku format detected or model_template not provided
        FileNotFoundError: If target not found
    """
    # Resolve id/path
    tdir = Path(target_id_or_path)
    if not tdir.exists():
        tdir = Path(root) / "targets" / str(target_id_or_path)
    if not tdir.exists():
        raise FileNotFoundError(f"Target not found: {target_id_or_path}")

    # Check for legacy format
    if _check_legacy_format(tdir):
        raise ValueError(
            f"Legacy Haiku format detected at {tdir}. "
            f"This artifact uses the old params.npz format. "
            f"Please rebuild the target with the new Equinox-based code."
        )

    # Load data
    data = np.load(tdir / "data.npz")
    X, Y = data["X"], data["Y"]

    # Load metadata
    with open(tdir / "meta.json") as f:
        meta = json.load(f)

    # Load Equinox model
    if model_template is None:
        raise ValueError(
            "model_template is required to load Equinox models. "
            "Build a model with the same architecture from metadata, then pass it here."
        )

    model = _deserialize_model(model_template, tdir)

    return X, Y, model, meta, tdir


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


def load_target_by_id(target_id: str, experiment: str, model_template, paths=None):
    """Load target artifact from new artifact system using target_id.

    This function replaces load_target_artifact() for the new artifact system.
    It searches for targets in the experiment's targets directory (which contains
    symlinks to the content-addressed store).

    Args:
        target_id: Target ID string (e.g., 'tgt_abc123')
        experiment: Experiment name
        model_template: Equinox model template for reconstruction (required)
        paths: Paths object (default: from environment)

    Returns:
        Tuple of (X, Y, model, meta_dict, payload_path)

    Raises:
        FileNotFoundError: If target not found in experiment
        ValueError: If model_template not provided or legacy format detected
    """
    if paths is None:
        from lambda_hat.artifacts import Paths

        paths = Paths.from_env()

    # Find target in experiment targets directory
    targets_dir = paths.experiments / experiment / "targets"
    if not targets_dir.exists():
        raise FileNotFoundError(
            f"No targets directory found for experiment '{experiment}' at {targets_dir}"
        )

    # Search for matching target_id in symlinked targets
    for target_link in targets_dir.glob("*"):
        if target_link.is_symlink() or target_link.is_dir():
            meta_path = target_link / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                        if meta.get("target_id") == target_id:
                            # Found it! Load using existing function
                            return load_target_artifact_from_dir(target_link, model_template)
                except (json.JSONDecodeError, KeyError):
                    continue

    raise FileNotFoundError(
        f"Target '{target_id}' not found in experiment '{experiment}'. "
        f"Available targets in {targets_dir}: {list(targets_dir.glob('*'))}"
    )
