# lambda_hat/entrypoints/sample.py
import argparse
import json
import time
from pathlib import Path

import jax
from omegaconf import OmegaConf

from lambda_hat.analysis import (
    analyze_traces,
    create_arviz_diagnostics,
    create_combined_convergence_plot,
    create_work_normalized_variance_plot,
)
from lambda_hat.config import validate_teacher_cfg
from lambda_hat.losses import as_dtype, make_loss_fns
from lambda_hat.models import build_mlp_forward_fn, count_params
from lambda_hat.sampling_runner import run_sampler
from lambda_hat.target_artifacts import append_sample_manifest, load_target_artifact
from lambda_hat.targets import TargetBundle


def main():
    ap = argparse.ArgumentParser("lambda-hat-sample")
    ap.add_argument("--config-yaml", required=True, help="Path to composed YAML config")
    ap.add_argument("--target-id", required=True, help="Target ID string")
    ap.add_argument("--run-dir", required=True, help="Directory where to write run outputs")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config_yaml)

    # Fail-fast validation
    assert "sampler" in cfg and "name" in cfg.sampler, "cfg.sampler.name missing"
    assert "jax" in cfg and "enable_x64" in cfg.jax, "cfg.jax.enable_x64 missing"

    jax.config.update("jax_enable_x64", bool(cfg.jax.enable_x64))

    print("=== LLC: Sample ===")
    print(f"Target ID: {args.target_id}")
    print(f"Run Dir: {args.run_dir}")
    try:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    except Exception:
        # Fallback to non-resolved view to keep the run alive
        print(OmegaConf.to_yaml(cfg, resolve=False))

    # Load target artifact using store.root and target_id
    X, Y, params, meta, tdir = load_target_artifact(cfg.store.root, args.target_id)

    # Guardrails
    if bool(cfg.jax.enable_x64) != bool(meta["jax_enable_x64"]):
        raise RuntimeError(
            f"Precision mismatch: sampler x64={cfg.jax.enable_x64}, "
            f"target x64={meta['jax_enable_x64']}"
        )

    # Recreate forward function from stored model config
    mcfg = meta["model_cfg"]

    # Use persisted widths (should always be present in new artifacts)
    widths = mcfg.get("widths")
    assert widths is not None, "Artifact missing resolved model widths"

    # Validate teacher config if present
    if meta.get("teacher_cfg"):
        validate_teacher_cfg(meta["teacher_cfg"])

    model = build_mlp_forward_fn(
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
    L0 = meta.get("metrics", {}).get("L0")
    if L0 is None or L0 == 0:
        raise ValueError(
            f"L0 reference loss not found or zero in target {args.target_id}. "
            "Target artifact may be corrupted or from an old format. "
            "Please rebuild the target using lambda-hat-build-target."
        )
    L0 = float(L0)

    # Store params and data in f32 for memory efficiency (precision determined dynamically)
    params_f32 = as_dtype(params, "float32")
    X_f32 = as_dtype(X, "float32")
    Y_f32 = as_dtype(Y, "float32")

    # Determine loss type from Stage B config (if specified) or from metadata
    loss_type = OmegaConf.select(cfg, "posterior.loss") or "mse"

    # Load necessary data parameters from Stage A metadata
    data_cfg = meta.get("data_cfg", {})
    noise_scale = data_cfg.get("noise_scale", 0.1)
    student_df = data_cfg.get("student_df", 4.0)

    # Loss fns (float32)
    loss_full_f32, loss_minibatch_f32 = make_loss_fns(
        model.apply,
        X_f32,
        Y_f32,
        loss_type=loss_type,
        noise_scale=noise_scale,
        student_df=student_df,
    )

    # Build TargetBundle with required fields
    d = count_params(params_f32)
    target_bundle = TargetBundle(
        d=d,
        params0=params_f32,
        loss_full=loss_full_f32,
        loss_minibatch=loss_minibatch_f32,
        X=X_f32,
        Y=Y_f32,
        L0=L0,
        model=model,
    )

    # Run the sampler via the new runner
    sampler_name = cfg.sampler.name
    key = jax.random.PRNGKey(int(cfg.runtime.seed))
    t0 = time.time()
    result = run_sampler(sampler_name, cfg, target_bundle, key)
    dt = time.time() - t0

    # Analysis: traces -> (metrics, idata)
    traces = result["traces"]  # dict of arrays (C, T)
    timings = result["timings"]  # {"adaptation":..., "sampling":..., "total":...}
    n_data = X_f32.shape[0]
    beta = float(result["beta"])
    metrics, idata = analyze_traces(
        traces, L0=L0, n_data=n_data, beta=beta, warmup=0, timings=timings
    )

    # Write artifacts
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(run_dir / "trace.nc")
    (run_dir / "analysis.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    # Create diagnostic plots
    create_arviz_diagnostics({sampler_name: idata}, run_dir)
    create_combined_convergence_plot({sampler_name: idata}, run_dir)
    create_work_normalized_variance_plot({sampler_name: idata}, run_dir)

    # Manifest entry
    sp = OmegaConf.to_container(cfg.sampler, resolve=True)
    append_sample_manifest(
        cfg.store.root,
        args.target_id,
        {
            "target_id": args.target_id,
            "sampler": sampler_name,
            "hyperparams": sp,
            "dtype64": bool(cfg.jax.enable_x64),
            "walltime_sec": dt,
            "artifact_path": str(run_dir),
            "metrics": metrics,
            "code_sha": cfg.runtime.code_sha,
            "created_at": time.time(),
        },
    )

    print(f"[sample] wrote trace.nc & analysis.json in {dt:.2f}s → {run_dir}")


if __name__ == "__main__":
    main()
