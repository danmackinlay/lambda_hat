from __future__ import annotations

import time
from typing import Any, Dict

import jax
from omegaconf import DictConfig, OmegaConf
import hydra

# Ensure resolvers are registered before Hydra composes config.
from lambda_hat import hydra_support  # noqa: F401
from lambda_hat.target_artifacts import TargetMeta, save_target_artifact, _hash_arrays, _flatten_params_dict

def _pkg_versions() -> Dict[str, str]:
    import haiku as hk, jax, blackjax, numpy as np
    return {
        "jax": getattr(jax, "__version__", "unknown"),
        "haiku": getattr(hk, "__version__", "unknown"),
        "blackjax": getattr(blackjax, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
    }

def build_target_components(key, cfg: DictConfig):
    """Returns (X, Y, model, init_params, loss_fn, trained_params)."""
    from lambda_hat.targets import build_target
    from lambda_hat.config import Config

    # Convert OmegaConf DictConfig to the project's Config class
    # For now, we'll just use the existing build_target function
    # which expects a Config object
    target_bundle = build_target(key, cfg)

    # Extract components
    X = target_bundle.X_f64
    Y = target_bundle.Y_f64
    trained_params = target_bundle.params0_f64
    model = target_bundle.model

    train_info = {"L0": float(target_bundle.L0)}

    return X, Y, model, trained_params, train_info

@hydra.main(config_path="../conf/target", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
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
    dims = {"n": int(X.shape[0]), "d": int(X.shape[1]), "p": sum(v.size for v in flat.values())}

    meta = TargetMeta(
        target_id=cfg.target.id,
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
    out_dir = save_target_artifact(cfg.store.root, cfg.target.id, X, Y, theta, meta)
    print(f"[build-target] wrote {out_dir}")
    print(f"[build-target] L0 = {train_info.get('L0', 0):.6f}")

if __name__ == "__main__":
    main()