from __future__ import annotations

import json
import time
from pathlib import Path
from hashlib import md5

import jax
import hydra
from omegaconf import DictConfig, OmegaConf
import arviz as az

from lambda_hat.target_artifacts import load_target_artifact, append_sample_manifest
from lambda_hat.sampling_runner import run_sampler  # Use extracted sampler runner
from lambda_hat.targets import TargetBundle
from lambda_hat.losses import make_loss_fns, as_dtype
from lambda_hat.analysis import analyze_traces
from lambda_hat.models import build_mlp_forward_fn
from lambda_hat.models import infer_widths



def run_sampling_logic(cfg: DictConfig) -> None:
    """Executes the sampling logic. Reusable by different entry points."""
    if cfg.jax.enable_x64:
        jax.config.update("jax_enable_x64", True)

    print("=== LLC: Sample ===")
    try:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    except Exception:
        # Fallback to non-resolved view to keep the run alive
        print(OmegaConf.to_yaml(cfg, resolve=False))

    # Load target
    X, Y, params, meta, tdir = load_target_artifact(cfg.store.root, cfg.target_id)

    # Guardrails
    if bool(cfg.jax.enable_x64) != bool(meta["jax_enable_x64"]):
        raise RuntimeError(
            f"Precision mismatch: sampler x64={cfg.jax.enable_x64}, target x64={meta['jax_enable_x64']}"
        )

    # Use the existing targets module to build a target bundle

    # Recreate forward function from stored model config
    mcfg = meta["model_cfg"]

    # Infer widths if not stored directly
    if "widths" in mcfg and mcfg["widths"] is not None:
        widths = mcfg["widths"]
    else:
        widths = infer_widths(
            in_dim=int(X.shape[-1]),
            out_dim=int(Y.shape[-1] if Y.ndim > 1 else 1),
            depth=mcfg.get("depth", 2),
            target_params=mcfg.get("target_params", None),
            fallback_width=mcfg.get("hidden", 32),
        )

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

    # Get L0 from metadata (dangerous fallback removed - fail explicitly)
    L0 = meta.get("metrics", {}).get("L0")
    if L0 is None or L0 == 0:
        raise ValueError(
            f"L0 reference loss not found or zero in target {cfg.target_id}. "
            "Target artifact may be corrupted or from an old format. "
            "Please rebuild the target using lambda-hat-build-target."
        )
    L0 = float(L0)

    # Store params and data in f32 for memory efficiency (precision determined dynamically)
    params_f32 = as_dtype(params, "float32")
    X_f32 = as_dtype(X, "float32")
    Y_f32 = as_dtype(Y, "float32")

    # Guard: Use model.apply directly. (Removes DummyModel usage, lines 87-95)
    try:
        test_key = jax.random.PRNGKey(0)  # Dummy key for model validation
        _ = model.apply(params_f32, test_key, X_f32[:1])
    except Exception as e:
        raise RuntimeError(f"Model/params mismatch for target {cfg.target_id}: {e}")

    # Determine loss type from Stage B config (if specified) or from metadata
    loss_type = getattr(cfg.get("posterior", {}), "loss", "mse")

    # Load necessary data parameters from Stage A metadata
    data_cfg = meta.get("data_cfg", {})
    noise_scale = data_cfg.get("noise_scale", 0.1)
    student_df = data_cfg.get("student_df", 4.0)

    # Create loss functions in f32 (precision cast dynamically in sampling_runner)
    loss_full_f32, loss_mini_f32 = make_loss_fns(
        model.apply,
        X_f32,
        Y_f32,
        loss_type=loss_type,
        noise_scale=noise_scale,
        student_df=student_df,
    )

    # Build target bundle for compatibility with existing code
    target = TargetBundle(
        d=meta["dims"]["p"],
        params0=params_f32,
        loss_full=loss_full_f32,
        loss_minibatch=loss_mini_f32,
        X=X_f32,
        Y=Y_f32,
        L0=L0,
        model=model,  # Pass the Haiku object directly
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
    hp_str = json.dumps(
        OmegaConf.to_container(cfg.sampler, resolve=True), sort_keys=True
    )
    run_id = md5(hp_str.encode()).hexdigest()[:8]
    run_dir = sample_dir / f"run_{run_id}"
    run_dir.mkdir(exist_ok=True)

    # --- Analysis and Trace Saving (Enhanced Path) ---
    traces = result.get("traces", {})
    timings = result.get("timings", {})
    metrics = {
        "elapsed_time": dt,
        "L0": L0,
    }
    n_data = X.shape[0]  # Use X loaded from artifact
    beta = result["beta"]

    if "Ln" in traces:
        # Determine warmup and recording frequency (needed for analysis)
        sampler_name = cfg.sampler.name
        sampler_specific_cfg = getattr(cfg.sampler, sampler_name, {})
        record_every = 1
        if sampler_name == "sgld":
            # SGLD records less frequently
            record_every = getattr(sampler_specific_cfg, "eval_every", 10)
            warmup = getattr(sampler_specific_cfg, "warmup", 0)
            warmup_steps_in_trace = warmup // max(1, record_every)
        else:
            # HMC/MCLMC traces are already post-warmup
            warmup_steps_in_trace = 0

        # Use the new comprehensive analysis function
        llc_metrics, idata = analyze_traces(
            traces,
            L0,
            n_data=n_data,
            beta=beta,
            warmup=warmup_steps_in_trace,
            timings=timings,
        )

        metrics.update(llc_metrics)
        metrics["n_samples"] = traces["Ln"].shape[1] - warmup_steps_in_trace

        # Save traces using ArviZ InferenceData from analyze_traces
        az.to_netcdf(idata, run_dir / "trace.nc")

        # Create diagnostic plots immediately
        from lambda_hat.analysis import (
            create_arviz_diagnostics,
            create_combined_convergence_plot,
            create_work_normalized_variance_plot,
        )

        create_arviz_diagnostics({cfg.sampler.name: idata}, run_dir)
        create_combined_convergence_plot({cfg.sampler.name: idata}, run_dir)
        create_work_normalized_variance_plot({cfg.sampler.name: idata}, run_dir)
    else:
        print("[WARNING] No 'Ln' found in traces. Analysis and trace saving skipped.")

    # Add any remaining computed metrics from result
    for k, v in result.items():
        if k not in ["traces", "timings", "elapsed_time", "beta"]:
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
