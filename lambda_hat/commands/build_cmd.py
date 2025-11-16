# lambda_hat/commands/build_cmd.py
"""Build command - Stage A: Train neural network and create target artifact."""

import time
from typing import Dict, Optional

import jax
from omegaconf import OmegaConf

from lambda_hat import omegaconf_support  # noqa: F401
from lambda_hat.artifacts import ArtifactStore, Paths, RunContext, safe_symlink
from lambda_hat.nn_eqx import count_params
from lambda_hat.target_artifacts import (
    TargetMeta,
    _hash_model,
    save_target_artifact_explicit,
)
from lambda_hat.targets import build_target


def _pkg_versions() -> Dict[str, str]:
    """Get package versions for reproducibility tracking."""
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


def build_entry(config_yaml: str, target_id: str, experiment: Optional[str] = None) -> Dict:
    """Build target artifact from config.

    Args:
        config_yaml: Path to composed YAML config file
        target_id: Content-addressed target ID (e.g., 'tgt_abc123')
        experiment: Experiment name (optional, defaults from config then env)

    Returns:
        dict: Build results with keys:
            - urn: Artifact URN in content-addressed store
            - target_id: Target ID
            - run_id: Run ID for this build
            - L0: Initial loss value
            - experiment: Experiment name
    """
    cfg = OmegaConf.load(config_yaml)

    # Determine experiment name (CLI > config > env)
    experiment = experiment or cfg.get("experiment", None)

    # Initialize artifact system
    paths = Paths.from_env()
    paths.ensure()
    store = ArtifactStore(paths.store)

    # Create RunContext for this build
    ctx = RunContext.create(
        experiment=experiment, algo="build_target", paths=paths, tag=target_id
    )
    target_dir = ctx.scratch_dir / "target_payload"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Fail-fast validation
    assert "target" in cfg and "seed" in cfg.target, "cfg.target.seed missing"
    assert "jax" in cfg and "enable_x64" in cfg.jax, "cfg.jax.enable_x64 missing"

    jax.config.update("jax_enable_x64", bool(cfg.jax.enable_x64))

    print("=== LLC: Build Target ===")
    print(f"Target ID: {target_id}")
    print(f"Target Dir: {target_dir}")
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
        target_id=target_id,
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

    save_target_artifact_explicit(target_dir, X, Y, theta, meta)
    print(f"[build-target] wrote {target_dir}")
    print(f"[build-target] L0 = {L0:.6f}")

    # Commit to artifact store
    urn = store.put_dir(
        target_dir,
        a_type="target",
        meta={
            "target_id": target_id,
            "seed": int(cfg.target.seed),
            "model_cfg": model_cfg,
            "data_cfg": OmegaConf.to_container(cfg.data, resolve=True),
            "training_cfg": OmegaConf.to_container(cfg.training, resolve=True),
            "teacher_cfg": teacher_cfg,
        },
    )

    # Create symlinks in experiment
    target_short_id = urn.split(":")[-1][:12]
    safe_symlink(
        store.path_for(urn) / "payload",
        paths.experiments / ctx.experiment / "targets" / target_short_id,
    )
    safe_symlink(
        paths.experiments / ctx.experiment / "targets" / target_short_id,
        ctx.inputs_dir / "target",
    )

    # Write run manifest
    ctx.write_run_manifest(
        {
            "phase": "build_target",
            "outputs": [{"urn": urn, "role": "target"}],
            "target_id": target_id,
        }
    )

    print(f"[build-target] committed to store: {urn}")
    print(f"[build-target] experiment: {ctx.experiment}, run: {ctx.run_id}")

    return {
        "urn": urn,
        "target_id": target_id,
        "run_id": ctx.run_id,
        "L0": float(L0),
        "experiment": ctx.experiment,
    }
