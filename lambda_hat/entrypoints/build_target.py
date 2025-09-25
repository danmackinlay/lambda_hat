from __future__ import annotations

import time
from typing import Dict

import jax
from omegaconf import DictConfig, OmegaConf
import hydra

# Ensure resolvers are registered before Hydra composes config.
from lambda_hat import hydra_support  # noqa: F401
from lambda_hat.target_artifacts import (
    TargetMeta,
    save_target_artifact,
    _hash_arrays,
    _flatten_params_dict,
    load_target_artifact,
)
from lambda_hat.targets import build_target


def _pkg_versions() -> Dict[str, str]:
    import haiku as hk
    import jax
    import blackjax
    import numpy as np

    return {
        "jax": getattr(jax, "__version__", "unknown"),
        "haiku": getattr(hk, "__version__", "unknown"),
        "blackjax": getattr(blackjax, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
    }


def build_target_components(key, cfg: DictConfig):
    """Returns (X, Y, model, init_params, loss_fn, trained_params)."""

    # Convert OmegaConf DictConfig to the project's Config class
    # For now, we'll just use the existing build_target function
    # which expects a Config object
    target_bundle = build_target(key, cfg)

    # Extract components (using simplified TargetBundle attributes)
    # Data stored in f32 for efficiency, cast to f64 for precision during build
    from lambda_hat.losses import as_dtype

    X = as_dtype(target_bundle.X, "float64")
    Y = as_dtype(target_bundle.Y, "float64")
    # Params stored in f32, cast to f64 for precision
    trained_params = as_dtype(target_bundle.params0, "float64")
    model = target_bundle.model

    train_info = {"L0": float(target_bundle.L0)}

    return X, Y, model, trained_params, train_info


@hydra.main(config_path="../conf", config_name="workflow", version_base=None)
def main(cfg: DictConfig) -> None:
    # Resolve the target ID (prefer unified workflow's 'target_id', fall back to 'target.id')
    try:
        target_id = cfg.get("target_id") or cfg.target.id
    except Exception:
        print(
            "Critical Error: Failed to resolve target ID (expected 'target_id' or 'target.id')."
        )
        raise

    # --- IDEMPOTENCY CHECK ---
    try:
        # Attempt to load the artifact using the resolved target_id
        _, _, _, _, tdir = load_target_artifact(cfg.store.root, target_id)
        print(
            f"[build-target] Target {target_id} already exists at {tdir}. Skipping build."
        )
        return  # Exit successfully
    except FileNotFoundError:
        # If not found, proceed with building
        print(f"[build-target] Building new target {target_id}...")
    except Exception as e:
        # Handle potential corruption or other I/O errors during load check
        # We can safely print target_id here as it is already resolved.
        print(
            f"[build-target] Error checking for existing target {target_id}: {e}. Proceeding with build."
        )
    # -------------------------

    # Precision
    if cfg.jax.enable_x64:
        jax.config.update("jax_enable_x64", True)

    print("=== LLC: Build Target ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # RNG
    key = jax.random.PRNGKey(cfg.target.seed)

    # Build & train
    X, Y, model, theta, train_info = build_target_components(key, cfg)

    # Metadata & hashing
    flat = _flatten_params_dict(theta)
    theta_hash = _hash_arrays(flat)
    dims = {
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "p": sum(v.size for v in flat.values()),
    }

    meta = TargetMeta(
        target_id=target_id,
        created_at=time.time(),
        code_sha=cfg.runtime.code_sha,
        jax_enable_x64=bool(cfg.jax.enable_x64),
        pkg_versions=_pkg_versions(),
        seed=int(cfg.target.seed),
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        data_cfg=OmegaConf.to_container(cfg.data, resolve=True),
        training_cfg=OmegaConf.to_container(cfg.training, resolve=True),
        dims=dims,
        hashes={"theta": theta_hash},
        metrics={"L0": float(train_info.get("L0", 0))},
        hostname=cfg.runtime.hostname,
    )

    # Write artifact
    out_dir = save_target_artifact(cfg.store.root, target_id, X, Y, theta, meta)
    print(f"[build-target] wrote {out_dir}")
    print(f"[build-target] L0 = {train_info.get('L0', 0):.6f}")


if __name__ == "__main__":
    main()
