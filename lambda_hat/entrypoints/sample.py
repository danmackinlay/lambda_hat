from __future__ import annotations

import json, time
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf

from lambda_hat import hydra_support  # ensure resolvers registered
from lambda_hat.target_artifacts import load_target_artifact, append_sample_manifest
from lambda_hat.sampling_runner import run_sampler  # Use extracted sampler runner
from lambda_hat.config import Config

def run_sampling_logic(cfg: DictConfig) -> None:
    """Executes the sampling logic. Reusable by different entry points."""
    if cfg.jax.enable_x64:
        jax.config.update("jax_enable_x64", True)

    print("=== LLC: Sample ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Load target
    X, Y, params, meta, tdir = load_target_artifact(cfg.store.root, cfg.target_id)

    # Guardrails
    if bool(cfg.jax.enable_x64) != bool(meta["jax_enable_x64"]):
        raise RuntimeError(f"Precision mismatch: sampler x64={cfg.jax.enable_x64}, target x64={meta['jax_enable_x64']}")

    # Use the existing targets module to build a target bundle
    from lambda_hat.targets import TargetBundle
    from lambda_hat.losses import make_loss_fns, as_dtype
    from lambda_hat.models import build_mlp_forward_fn

    # Recreate forward function from stored model config
    mcfg = meta["model_cfg"]

    # Infer widths if not stored directly
    if "widths" in mcfg and mcfg["widths"] is not None:
        widths = mcfg["widths"]
    else:
        from lambda_hat.models import infer_widths
        widths = infer_widths(
            in_dim=int(X.shape[-1]),
            out_dim=int(Y.shape[-1] if Y.ndim > 1 else 1),
            depth=mcfg.get("depth", 2),
            target_params=mcfg.get("target_params", None),
            fallback_width=mcfg.get("hidden", 32)
        )

    forward_fn = build_mlp_forward_fn(
        in_dim=int(X.shape[-1]),
        widths=widths,
        out_dim=int(Y.shape[-1] if Y.ndim > 1 else 1),
        activation=mcfg.get("activation", "relu"),
        bias=mcfg.get("bias", True),
        init=mcfg.get("init", "he"),
        skip=mcfg.get("skip_connections", False),
        residual_period=mcfg.get("residual_period", 2),
        layernorm=mcfg.get("layernorm", False),
    )

    # Get L0 from metadata
    L0 = float(meta.get("metrics", {}).get("L0", 0))
    if L0 == 0:
        # Fallback: compute L0 if not stored (compatibility)
        logits = jnp.dot(X, jnp.ones(X.shape[-1]))  # dummy computation
        L0 = 0.1  # placeholder - will be computed properly in target bundle

    # Convert params and data to proper dtypes
    params_f32 = as_dtype(params, jnp.float32)
    params_f64 = as_dtype(params, jnp.float64)
    X_f32 = jnp.asarray(X, dtype=jnp.float32)
    Y_f32 = jnp.asarray(Y, dtype=jnp.float32)
    X_f64 = jnp.asarray(X, dtype=jnp.float64)
    Y_f64 = jnp.asarray(Y, dtype=jnp.float64)

    # Guard: params shapes must match forward_fn expectations on a dummy input
    try:
        # Test with a dummy haiku model-like object
        class DummyModel:
            def apply(self, params, rng, x):
                return forward_fn(params, x)

        dummy_model = DummyModel()
        _ = dummy_model.apply(params_f32, None, X_f32[:1])
    except Exception as e:
        raise RuntimeError(f"Model/params mismatch for target {cfg.target_id}: {e}")

    # Create loss functions
    loss_full_f32, loss_mini_f32 = make_loss_fns(X_f32, Y_f32, forward_fn, jnp.float32)
    loss_full_f64, loss_mini_f64 = make_loss_fns(X_f64, Y_f64, forward_fn, jnp.float64)

    # Build target bundle for compatibility with existing code
    target = TargetBundle(
        d=meta["dims"]["p"],
        params0_f32=params_f32,
        params0_f64=params_f64,
        loss_full_f32=loss_full_f32,
        loss_minibatch_f32=loss_mini_f32,
        loss_full_f64=loss_full_f64,
        loss_minibatch_f64=loss_mini_f64,
        X_f32=X_f32,
        Y_f32=Y_f32,
        X_f64=X_f64,
        Y_f64=Y_f64,
        L0=L0,
        model={"forward": forward_fn}
    )

    # Run the selected sampler using existing machinery
    key = jax.random.PRNGKey(cfg.runtime.seed)
    t0 = time.time()

    # Pass the original cfg object
    result = run_sampler(cfg.sampler.name, cfg, target, key)

    dt = time.time() - t0

    # Save outputs to sample-specific directory
    sample_dir = Path(cfg.store.root) / "samples" / cfg.target_id / cfg.sampler.name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique run ID based on hyperparams
    from hashlib import md5
    hp_str = json.dumps(OmegaConf.to_container(cfg.sampler, resolve=True), sort_keys=True)
    run_id = md5(hp_str.encode()).hexdigest()[:8]
    run_dir = sample_dir / f"run_{run_id}"
    run_dir.mkdir(exist_ok=True)

    # Save traces if available
    import numpy as np
    traces = result.get("traces", None)
    if traces is not None:
        # Prefer ArviZ NetCDF if available; otherwise fall back to NPZ
        try:
            import arviz as az
            az_trace = az.from_dict(posterior=jax.tree.map(lambda x: np.asarray(x), traces))
            az.to_netcdf(az_trace, run_dir / "trace.nc")
        except Exception:
            # Fallback to NPZ with flattened structure
            flat = {}
            def _flatput(prefix, obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _flatput(f"{prefix}{k}/", v)
                else:
                    flat[prefix[:-1]] = np.asarray(obj)
            _flatput("", traces)
            np.savez_compressed(run_dir / "traces.npz", **flat)

    # Extract metrics
    metrics = {
        "elapsed_time": dt,
        "L0": L0,
        "n_samples": result.get("n_samples", 0),
    }

    # Add any computed metrics from result
    for k, v in result.items():
        if k not in ["samples", "elapsed_time"]:
            if isinstance(v, (int, float, str, bool)):
                metrics[k] = v
            elif hasattr(v, "item"):
                metrics[k] = v.item()

    # Save analysis
    with open(run_dir / "analysis.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Extract just the sampler hyperparams (everything under sampler.* except 'name')
    sp = OmegaConf.to_container(cfg.sampler, resolve=True) or {}
    if isinstance(sp, dict):
        sp.pop("name", None)
        sp.pop("chains", None)  # chains is workflow config, not sampler hyperparams

    # Record in manifest
    record = {
        "target_id": cfg.target_id,
        "sampler": cfg.sampler.name,
        "hyperparams": sp,
        "dtype64": bool(cfg.jax.enable_x64),
        "walltime_sec": dt,
        "artifact_path": str(run_dir),
        "metrics": metrics,
        "code_sha": cfg.runtime.code_sha,
        "created_at": time.time(),
    }
    append_sample_manifest(cfg.store.root, cfg.target_id, record)

    print(f"[sample] done in {dt:.2f}s → {run_dir}")

@hydra.main(config_path="../conf/sample", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    run_sampling_logic(cfg)

if __name__ == "__main__":
    main()