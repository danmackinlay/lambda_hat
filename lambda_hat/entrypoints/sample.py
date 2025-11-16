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
from lambda_hat.artifacts import ArtifactStore, Paths, RunContext
from lambda_hat.config import validate_teacher_cfg
from lambda_hat.losses import as_dtype, make_loss_fns
from lambda_hat.nn_eqx import build_mlp, count_params
from lambda_hat.sampling_runner import run_sampler
from lambda_hat.target_artifacts import append_sample_manifest, load_target_artifact
from lambda_hat.targets import TargetBundle


def main():
    ap = argparse.ArgumentParser("lambda-hat-sample")
    ap.add_argument("--config-yaml", required=True, help="Path to composed YAML config")
    ap.add_argument("--target-id", required=True, help="Target ID string")
    ap.add_argument(
        "--experiment",
        required=False,
        help="Experiment name (defaults from config then env)",
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config_yaml)

    # Determine experiment name and sampler name early
    experiment = args.experiment or cfg.get("experiment", None)
    sampler_name = cfg.sampler.name

    # Initialize artifact system
    paths = Paths.from_env()
    paths.ensure()
    store = ArtifactStore(paths.store)

    # Create RunContext for this sampling run
    ctx = RunContext.create(
        experiment=experiment,
        algo=sampler_name,
        paths=paths,
        tag=args.target_id[:8],  # Use short target ID as tag
    )
    run_dir = ctx.run_dir

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

    # Recreate model template from metadata (needed for Equinox deserialization)
    # We need to build the model BEFORE loading to get the correct structure
    mcfg = OmegaConf.load(args.config_yaml).get("model", {})
    meta_path = Path(cfg.store.root) / "targets" / args.target_id / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # Guardrails
    if bool(cfg.jax.enable_x64) != bool(meta["jax_enable_x64"]):
        raise RuntimeError(
            f"Precision mismatch: sampler x64={cfg.jax.enable_x64}, "
            f"target x64={meta['jax_enable_x64']}"
        )

    # Build model template from metadata
    model_cfg = meta["model_cfg"]
    widths = model_cfg.get("widths")
    assert widths is not None, "Artifact missing resolved model widths"

    # Validate teacher config if present
    if meta.get("teacher_cfg"):
        validate_teacher_cfg(meta["teacher_cfg"])

    # Create model template for loading (dummy key for structure only)
    model_template = build_mlp(
        in_dim=model_cfg.get("in_dim", 2),
        widths=widths,
        out_dim=model_cfg.get("out_dim", 1),
        activation=model_cfg.get("activation", "relu"),
        bias=model_cfg.get("bias", True),
        layernorm=model_cfg.get("layernorm", False),
        key=jax.random.PRNGKey(0),  # Dummy key, will be overwritten
    )

    # Load target artifact with model template
    X, Y, model, meta, tdir = load_target_artifact(
        cfg.store.root, args.target_id, model_template=model_template
    )
    params = model  # For Equinox, model IS the params

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
    # Equinox models are called directly: model(x), not model.apply(params, None, x)
    predict_fn = lambda m, x: m(x)
    loss_full_f32, loss_minibatch_f32 = make_loss_fns(
        predict_fn,
        X_f32,
        Y_f32,
        loss_type=loss_type,
        noise_scale=noise_scale,
        student_df=student_df,
    )

    # Flatten params for VI (required by TargetBundle even if HMC/SGLD don't use it)
    params0_flat, unravel_fn = jax.flatten_util.ravel_pytree(params_f32)

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
        params0_flat=params0_flat,
        unravel_fn=unravel_fn,
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
    work = result.get("work")  # {"n_full_loss":..., "n_minibatch_grads":...}
    n_data = X_f32.shape[0]
    beta = float(result["beta"])

    # Extract sampler flavour from work dict, or infer from sampler name
    sampler_flavour = None
    if work is not None:
        sampler_flavour = work.get("sampler_flavour")

    metrics, idata = analyze_traces(
        traces,
        L0=L0,
        n_data=n_data,
        beta=beta,
        warmup=0,
        timings=timings,
        work=work,
        sampler_flavour=sampler_flavour,
    )

    # Write artifacts
    run_dir.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(run_dir / "trace.nc")
    (run_dir / "analysis.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    # TensorBoard logging (Stage 2) - VI only for now
    if (
        sampler_name == "vi"
        and hasattr(cfg.sampler.vi, "tensorboard")
        and cfg.sampler.vi.tensorboard
    ):
        from tensorboardX import SummaryWriter

        tb_dir = ctx.tb_dir
        writer = SummaryWriter(str(tb_dir))

        # Log scalar traces over optimization steps
        num_chains, num_steps = traces["elbo"].shape
        for step in range(num_steps):
            # Average across chains for cleaner plots
            writer.add_scalar("vi/elbo", float(traces["elbo"][:, step].mean()), step)
            writer.add_scalar("vi/elbo_like", float(traces["elbo_like"][:, step].mean()), step)
            writer.add_scalar("vi/logq", float(traces["logq"][:, step].mean()), step)
            writer.add_scalar("vi/radius2", float(traces["radius2"][:, step].mean()), step)
            writer.add_scalar(
                "vi/resp_entropy", float(traces["resp_entropy"][:, step].mean()), step
            )
            writer.add_scalar(
                "vi/cumulative_fge", float(traces["cumulative_fge"][:, step].mean()), step
            )

            # Control variate metrics (constant across steps, but log anyway)
            if "Eq_Ln_mc" in traces:
                writer.add_scalar("vi/Eq_Ln_mc", float(traces["Eq_Ln_mc"][:, step].mean()), step)
            if "Eq_Ln_cv" in traces:
                writer.add_scalar("vi/Eq_Ln_cv", float(traces["Eq_Ln_cv"][:, step].mean()), step)
            if "variance_reduction" in traces:
                writer.add_scalar(
                    "vi/variance_reduction",
                    float(traces["variance_reduction"][:, step].mean()),
                    step,
                )

            # Stage 2 enhanced diagnostics
            if "pi_min" in traces:
                writer.add_scalar("vi/pi_min", float(traces["pi_min"][:, step].mean()), step)
                writer.add_scalar("vi/pi_max", float(traces["pi_max"][:, step].mean()), step)
                writer.add_scalar(
                    "vi/pi_entropy", float(traces["pi_entropy"][:, step].mean()), step
                )
            if "D_sqrt_min" in traces:
                writer.add_scalar(
                    "vi/D_sqrt_min", float(traces["D_sqrt_min"][:, step].mean()), step
                )
                writer.add_scalar(
                    "vi/D_sqrt_max", float(traces["D_sqrt_max"][:, step].mean()), step
                )
                writer.add_scalar(
                    "vi/D_sqrt_med", float(traces["D_sqrt_med"][:, step].mean()), step
                )
            if "grad_norm" in traces:
                writer.add_scalar("vi/grad_norm", float(traces["grad_norm"][:, step].mean()), step)
            if "A_col_norm_max" in traces:
                writer.add_scalar(
                    "vi/A_col_norm_max", float(traces["A_col_norm_max"][:, step].mean()), step
                )

        # Log final metrics
        writer.add_scalar("vi/llc_mean", metrics["llc_mean"], num_steps)
        writer.add_scalar("vi/L0", L0, num_steps)
        if work:
            if "lambda_hat_mean" in work:
                writer.add_scalar("vi/lambda_hat_mean", work["lambda_hat_mean"], num_steps)
            if "Eq_Ln_mean" in work:
                writer.add_scalar("vi/Eq_Ln_final", work["Eq_Ln_mean"], num_steps)
            if "Ln_wstar" in work:
                writer.add_scalar("vi/Ln_wstar", work["Ln_wstar"], num_steps)

        writer.close()
        print(f"[sample] TensorBoard logs written to {tb_dir}")

    # Create diagnostic plots
    create_arviz_diagnostics({sampler_name: idata}, run_dir)
    create_combined_convergence_plot({sampler_name: idata}, run_dir)
    create_work_normalized_variance_plot({sampler_name: idata}, run_dir)

    # Legacy manifest entry (keep for now, may remove later)
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

    # Write run manifest
    ctx.write_run_manifest(
        {
            "phase": "sample",
            "inputs": [],  # Could add target URN reference if needed
            "target_id": args.target_id,
            "sampler": sampler_name,
            "hyperparams": sp,
            "dtype64": bool(cfg.jax.enable_x64),
            "walltime_sec": dt,
            "metrics": metrics,
        }
    )

    print(f"[sample] wrote trace.nc & analysis.json in {dt:.2f}s → {run_dir}")
    print(f"[sample] experiment: {ctx.experiment}, run: {ctx.run_id}")


if __name__ == "__main__":
    main()
