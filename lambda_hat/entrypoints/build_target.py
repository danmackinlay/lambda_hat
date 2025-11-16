# lambda_hat/entrypoints/build_target.py
import argparse
import time
from typing import Dict

import jax
from omegaconf import OmegaConf

from lambda_hat import omegaconf_support  # noqa: F401
from lambda_hat.nn_eqx import count_params
from lambda_hat.target_artifacts import (
    TargetMeta,
    _hash_model,
    save_target_artifact_explicit,
)
from lambda_hat.targets import build_target


def _pkg_versions() -> Dict[str, str]:
    import blackjax
    import equinox as eqx
    import jax
    import numpy as np

    return {
        "jax": getattr(jax, "__version__", "unknown"),
        "equinox": getattr(eqx, "__version__", "unknown"),
        "blackjax": getattr(blackjax, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
    }


def main():
    ap = argparse.ArgumentParser("lambda-hat-build-target")
    ap.add_argument("--config-yaml", required=True, help="Path to composed YAML config")
    ap.add_argument("--target-id", required=True, help="Target ID string")
    ap.add_argument("--target-dir", required=True, help="Directory where to write artifacts")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config_yaml)

    # Fail-fast validation
    assert "target" in cfg and "seed" in cfg.target, "cfg.target.seed missing"
    assert "jax" in cfg and "enable_x64" in cfg.jax, "cfg.jax.enable_x64 missing"
    assert "store" in cfg and "root" in cfg.store, "cfg.store.root missing"

    jax.config.update("jax_enable_x64", bool(cfg.jax.enable_x64))

    print("=== LLC: Build Target ===")
    print(f"Target ID: {args.target_id}")
    print(f"Target Dir: {args.target_dir}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # RNG
    key = jax.random.PRNGKey(int(cfg.target.seed))

    # Build & train
    target_bundle, used_model_widths, used_teacher_widths = build_target(key, cfg)

    # Extract components for saving
    X = target_bundle.X
    Y = target_bundle.Y
    theta = target_bundle.params0
    L0 = target_bundle.L0

    # Log resolved widths for debugging
    print(f"Resolved widths: model={used_model_widths}, teacher={used_teacher_widths}")
    if used_teacher_widths is not None:
        print(f"teacher.widths={used_teacher_widths}")

    # Metadata & hashing (Equinox)
    theta_hash = _hash_model(theta)
    dims = {
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "p": count_params(theta),
    }

    # Create model_cfg with resolved widths injected
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["widths"] = used_model_widths

    # Create teacher_cfg with resolved widths injected (if teacher exists)
    teacher_cfg = None
    if hasattr(cfg, "teacher") and cfg.teacher is not None and cfg.teacher != {}:
        teacher_cfg = OmegaConf.to_container(cfg.teacher, resolve=True)
        # Sanitize: drop any unexpected fields defensively
        teacher_cfg = {
            k: teacher_cfg.get(k)
            for k in [
                "depth",
                "widths",
                "activation",
                "dropout_rate",
                "target_params",
                "hidden",
            ]
        }
        teacher_cfg["widths"] = used_teacher_widths

    meta = TargetMeta(
        target_id=args.target_id,
        created_at=time.time(),
        code_sha=cfg.runtime.code_sha,
        jax_enable_x64=bool(cfg.jax.enable_x64),
        pkg_versions=_pkg_versions(),
        seed=int(cfg.target.seed),
        model_cfg=model_cfg,
        data_cfg=OmegaConf.to_container(cfg.data, resolve=True),
        training_cfg=OmegaConf.to_container(cfg.training, resolve=True),
        teacher_cfg=teacher_cfg,
        dims=dims,
        hashes={"theta": theta_hash},
        metrics={"L0": float(L0)},
        hostname=cfg.runtime.hostname,
    )

    save_target_artifact_explicit(args.target_dir, X, Y, theta, meta)
    print(f"[build-target] wrote {args.target_dir}")
    print(f"[build-target] L0 = {L0:.6f}")


if __name__ == "__main__":
    main()
