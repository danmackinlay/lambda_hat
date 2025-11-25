# lambda_hat/commands/sample_cmd.py
"""Sample command - Stage B: Run MCMC/VI sampler on target.

Workers always produce raw traces (traces_raw.npz + manifest.json) only.
Diagnostics are deferred to Stage C (diagnose command) via the golden path.
"""

import json
import logging
import time
from typing import Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from lambda_hat.artifacts import Paths, RunContext
from lambda_hat.config import validate_teacher_cfg
from lambda_hat.equinox_adapter import ensure_dtype
from lambda_hat.logging_config import configure_logging
from lambda_hat.losses import make_loss_fns
from lambda_hat.nn_eqx import build_mlp, count_params
from lambda_hat.sampling_runner import run_sampler
from lambda_hat.target_artifacts import load_target_artifact_from_dir
from lambda_hat.targets import TargetBundle

log = logging.getLogger(__name__)


def sample_entry(
    config_yaml: str,
    target_id: str,
    experiment: Optional[str] = None
) -> Dict:
    """Run sampler on target artifact.

    Args:
        config_yaml: Path to composed YAML config file
        target_id: Target ID to sample from
        experiment: Experiment name (optional, defaults from config then env)

    Returns:
        dict: Sample results with keys:
            - run_id: Run ID for this sample
            - run_dir: Path to run directory
            - metrics: Analysis metrics dict
            - experiment: Experiment name
    """
    cfg = OmegaConf.load(config_yaml)

    # Determine experiment name and sampler name early
    experiment = experiment or cfg.get("experiment", None)
    sampler_name = cfg.sampler.name

    # Configure logging at entrypoint
    configure_logging()

    # Initialize artifact system
    paths = Paths.from_env()
    paths.ensure()

    # Create RunContext for this sampling run
    ctx = RunContext.create(
        experiment=experiment,
        algo=sampler_name,
        paths=paths,
        tag=target_id[:8],  # Use short target ID as tag
    )
    run_dir = ctx.run_dir

    # Fail-fast validation
    assert "sampler" in cfg and "name" in cfg.sampler, "cfg.sampler.name missing"
    assert "jax" in cfg and "enable_x64" in cfg.jax, "cfg.jax.enable_x64 missing"

    # Set JAX precision per-task from config (before any PRNG or JIT usage)
    want_x64 = bool(cfg.jax.enable_x64)
    jax.config.update("jax_enable_x64", want_x64)
    log.info("sample_entry: jax_enable_x64 set to %s", want_x64)

    # Temporarily disable x64 for model template creation and target loading
    # (targets are always saved as float32, so template must be float32)
    jax.config.update("jax_enable_x64", False)

    log.info("=== LLC: Sample ===")
    log.info("Target ID: %s", target_id)
    log.info("Run Dir: %s", run_dir)
    try:
        log.debug("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    except Exception:
        # Fallback to non-resolved view to keep the run alive
        log.debug("Config (unresolved):\n%s", OmegaConf.to_yaml(cfg, resolve=False))

    # Resolve target from experiment (new artifact system)
    # Find target in experiment's targets directory (symlinks to content-addressed store)
    targets_dir = paths.experiments / experiment / "targets"
    target_payload = None
    target_meta = None

    # Search for matching target_id
    for target_link in targets_dir.glob("*"):
        meta_path = target_link / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                if meta.get("target_id") == target_id:
                    target_payload = target_link
                    target_meta = meta
                    break

    if target_payload is None:
        raise FileNotFoundError(
            f"Target {target_id} not found in experiment {experiment}. "
            f"Available targets: {list(targets_dir.glob('*'))}"
        )

    # Build model template from metadata
    model_cfg = target_meta["model_cfg"]
    widths = model_cfg.get("widths")
    assert widths is not None, "Artifact missing resolved model widths"

    # Validate teacher config if present
    if target_meta.get("teacher_cfg"):
        validate_teacher_cfg(target_meta["teacher_cfg"])

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

    # Load target artifact with model template (template and saved model both float32)
    X, Y, model, meta, tdir = load_target_artifact_from_dir(
        target_payload, model_template=model_template
    )

    # Restore x64 state to config value (from want_x64 set earlier)
    jax.config.update("jax_enable_x64", want_x64)

    # Cast loaded model and data to sampler's required dtype using equinox_adapter
    # Determine dtype from sampler config (default: float64 if x64, else float32)
    sampler_dtype_str = OmegaConf.select(cfg, f"sampler.{cfg.sampler.name}.dtype")
    if sampler_dtype_str is None:
        sampler_dtype_str = "float64" if cfg.jax.enable_x64 else "float32"

    target_dtype = jnp.float32 if sampler_dtype_str == "float32" else jnp.float64

    # Use equinox_adapter for safe dtype casting (handles static leaves correctly)
    model = ensure_dtype(model, target_dtype)
    X = X.astype(target_dtype)
    Y = Y.astype(target_dtype)

    params = model  # For Equinox, model IS the params

    # Get L0 from metadata
    L0 = meta.get("metrics", {}).get("L0")
    if L0 is None or L0 == 0:
        raise ValueError(
            f"L0 reference loss not found or zero in target {target_id}. "
            "Target artifact may be corrupted or from an old format. "
            "Please rebuild the target using lambda-hat build."
        )
    L0 = float(L0)

    # Determine loss type from Stage B config (if specified) or from metadata
    loss_type = OmegaConf.select(cfg, "posterior.loss") or "mse"

    # Load necessary data parameters from Stage A metadata
    data_cfg = meta.get("data_cfg", {})
    noise_scale = data_cfg.get("noise_scale", 0.1)
    student_df = data_cfg.get("student_df", 4.0)

    # Loss fns (use sampler's dtype)
    # Equinox models are called directly: model(x), not model.apply(params, None, x)
    def predict_fn(m, x):
        return m(x)

    loss_full, loss_minibatch = make_loss_fns(
        predict_fn,
        X,
        Y,
        loss_type=loss_type,
        noise_scale=noise_scale,
        student_df=student_df,
    )

    # Flatten params for VI (required by TargetBundle even if HMC/SGLD don't use it)
    # For Equinox models, extract only array leaves (not activation functions, etc.)
    trainable_params, _ = eqx.partition(params, eqx.is_array)
    params0_flat, unravel_fn = jax.flatten_util.ravel_pytree(trainable_params)

    # Build TargetBundle with required fields
    d = count_params(params)
    target_bundle = TargetBundle(
        d=d,
        params0=params,
        loss_full=loss_full,
        loss_minibatch=loss_minibatch,
        X=X,
        Y=Y,
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
    n_data = X.shape[0]
    beta = float(result["beta"])

    # Extract sampler flavour from work dict, or infer from sampler name
    sampler_flavour = None
    if work is not None:
        sampler_flavour = work.get("sampler_flavour")

    # Workers ALWAYS write raw traces only (single golden format)
    # Diagnostics are deferred to controller (diagnose command or --diagnose flag)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Convert all array-like values to NumPy arrays for NPZ storage
    traces_arrays = {k: np.asarray(v) for k, v in traces.items()}

    # Write compressed NPZ (binary, efficient storage)
    np.savez_compressed(run_dir / "traces_raw.npz", **traces_arrays)
    log.info("[sample] Wrote traces_raw.npz (raw traces only)")

    # No metrics computed during sampling (deferred to diagnostics stage)
    metrics = {}

    # TensorBoard logging (Stage 2) - VI only for now
    if (
        sampler_name == "vi"
        and hasattr(cfg.sampler.vi, "tensorboard")
        and cfg.sampler.vi.tensorboard
    ):
        _write_tensorboard_logs(ctx, traces, metrics, work)

    # Diagnostics deferred to Stage C (diagnose command)
    # Sampling (Stage B) only creates traces_raw.npz + manifest.json
    # Run diagnostics: lambda-hat diagnose --run-dir <path>
    # Or use --diagnose flag: lambda-hat sample --diagnose ...

    # Write run manifest (includes metadata needed for later reconstruction)
    sp = OmegaConf.to_container(cfg.sampler, resolve=True)
    ctx.write_run_manifest(
        {
            "phase": "sample",
            "inputs": [],  # Could add target URN reference if needed
            "target_id": target_id,
            "sampler": sampler_name,
            "hyperparams": sp,
            "dtype64": bool(cfg.jax.enable_x64),
            "walltime_sec": dt,
            "metrics": metrics,  # Empty dict for now; filled by diagnose stage
            # Metadata needed for reconstruction
            "L0": L0,
            "n_data": n_data,
            "beta": beta,
            "warmup": 0,
            "timings": timings,
            "work": work,
            "sampler_flavour": sampler_flavour,
        }
    )

    log.info("[sample] Wrote raw traces in %.2fs → %s", dt, run_dir)
    log.info("[sample] Experiment: %s, Run: %s", ctx.experiment, ctx.run_id)

    return {
        "run_id": ctx.run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "experiment": ctx.experiment,
    }


def _write_tensorboard_logs(ctx, traces, metrics, work):
    """Write TensorBoard logs for VI runs (helper function).

    Args:
        ctx: RunContext with tb_dir
        traces: Trace dict from VI sampler
        metrics: Analysis metrics
        work: Work metrics dict
    """
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
        writer.add_scalar("vi/resp_entropy", float(traces["resp_entropy"][:, step].mean()), step)
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
            writer.add_scalar("vi/pi_entropy", float(traces["pi_entropy"][:, step].mean()), step)
        if "D_sqrt_min" in traces:
            writer.add_scalar("vi/D_sqrt_min", float(traces["D_sqrt_min"][:, step].mean()), step)
            writer.add_scalar("vi/D_sqrt_max", float(traces["D_sqrt_max"][:, step].mean()), step)
            writer.add_scalar("vi/D_sqrt_med", float(traces["D_sqrt_med"][:, step].mean()), step)
        if "grad_norm" in traces:
            writer.add_scalar("vi/grad_norm", float(traces["grad_norm"][:, step].mean()), step)
        if "A_col_norm_max" in traces:
            writer.add_scalar(
                "vi/A_col_norm_max", float(traces["A_col_norm_max"][:, step].mean()), step
            )

    # Log final metrics
    L0 = metrics.get("L0", 0.0)  # Extract L0 from metrics if available
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
    log.info("[sample] TensorBoard logs written to %s", tb_dir)
