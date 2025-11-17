"""Method trial runner for Optuna hyperparameter optimization.

Executes approximate LLC estimation methods (SGLD, VI, MCLMC) with
given hyperparameters and returns LLC estimate for Optuna optimization.
"""

import json
import logging
import time
from pathlib import Path

import jax
from omegaconf import OmegaConf

from lambda_hat.analysis import analyze_traces
from lambda_hat.config import validate_teacher_cfg
from lambda_hat.losses import as_dtype, make_loss_fns
from lambda_hat.nn_eqx import build_mlp, count_params
from lambda_hat.sampling_runner import run_sampler
from lambda_hat.targets import TargetBundle, build_target
from lambda_hat.workflow_utils import compose_build_cfg

log = logging.getLogger(__name__)


def run_method_trial(
    problem_spec, method_cfg, ref_llc, out_metrics_json, budget_sec=6000, seed=None
):
    """Run approximate method trial with given hyperparameters.

    This runner creates a target from problem_spec, runs the specified method
    (SGLD/VI/MCLMC) with hyperparameters from method_cfg, and returns LLC estimate
    for comparison against ref_llc.

    Args:
        problem_spec: Dict with keys {model, data, teacher, seed, overrides?}
                      Same format as targets in config/experiments.yaml
        method_cfg: Dict with keys {name, <hyperparams>}
                    e.g., {"name": "sgld", "eta0": 0.01, "gamma": 0.5, "batch": 128}
                    e.g., {"name": "vi", "lr": 0.001, "M": 8, "r": 2}
        ref_llc: float, reference LLC from HMC (for logging/diagnostics)
        out_metrics_json: Path to write trial metrics JSON
        budget_sec: Wall-time budget in seconds (default: 6000 = 100 minutes)
        seed: Random seed for method (default: None, uses method_cfg or 12345)

    Returns:
        dict: Trial metrics with keys:
            - llc_hat: float, estimated LLC
            - se_hat: float or None, standard error if available
            - runtime_sec: float, actual runtime
            - error: float, |llc_hat - ref_llc| for logging
            - diagnostics: dict with ESS, work metrics, etc.
    """
    method_name = method_cfg["name"]
    log.info("[Method Trial] Starting %s trial", method_name)
    log.info("  Problem: %s", problem_spec)
    log.info("  Hyperparams: %s", method_cfg)
    log.info("  Budget: %ds (%.1fmin)", budget_sec, budget_sec / 60)
    log.info("  Reference LLC: %.4f", ref_llc)
    log.info("  Output: %s", out_metrics_json)

    # Determine seed
    if seed is None:
        seed = method_cfg.get("seed", 12345)

    # Build target using existing workflow infrastructure
    # Precision: use f32 for SGLD/VI, f64 for MCLMC (following existing patterns)
    enable_x64 = method_name in ("mclmc",)  # MCLMC uses f64, others use f32
    build_cfg = compose_build_cfg(problem_spec, jax_enable_x64=enable_x64)
    jax.config.update("jax_enable_x64", enable_x64)

    log.info("  Precision: %s", "float64" if enable_x64 else "float32")

    # Build target (returns TargetBundle and widths)

    key_build = jax.random.PRNGKey(int(problem_spec.get("seed", 42)))
    target_bundle_init, used_model_widths, used_teacher_widths = build_target(key_build, build_cfg)

    # Extract initial data
    X = target_bundle_init.X
    Y = target_bundle_init.Y
    params = target_bundle_init.params0
    L0 = float(target_bundle_init.L0)

    # Create metadata dict (simplified)
    meta = {
        "model_cfg": {
            **OmegaConf.to_container(build_cfg.model, resolve=True),
            "widths": used_model_widths,
        },
        "data_cfg": OmegaConf.to_container(build_cfg.data, resolve=True),
        "teacher_cfg": (
            {
                **OmegaConf.to_container(build_cfg.teacher, resolve=True),
                "widths": used_teacher_widths,
            }
            if used_teacher_widths is not None
            else None
        ),
        "metrics": {"L0": L0},
        "jax_enable_x64": enable_x64,
    }

    # Validate teacher config if present
    if meta.get("teacher_cfg"):
        validate_teacher_cfg(meta["teacher_cfg"])

    # Recreate model
    mcfg = meta["model_cfg"]
    widths = mcfg.get("widths")
    assert widths is not None, "Target missing resolved model widths"

    # Create model template for loading (dummy key for structure only)
    model = build_mlp(
        in_dim=int(X.shape[-1]),
        widths=widths,
        out_dim=int(Y.shape[-1] if Y.ndim > 1 else 1),
        activation=mcfg.get("activation", "relu"),
        bias=mcfg.get("bias", True),
        layernorm=mcfg.get("layernorm", False),
        key=jax.random.PRNGKey(0),  # Dummy key, will be overwritten
    )

    # Cast to appropriate precision
    dtype = "float64" if enable_x64 else "float32"
    X_cast = as_dtype(X, dtype)
    Y_cast = as_dtype(Y, dtype)
    params_cast = as_dtype(params, dtype)

    # Create loss functions
    data_cfg = meta.get("data_cfg", {})
    loss_type = "mse"  # Default for trials
    noise_scale = data_cfg.get("noise_scale", 0.1)
    student_df = data_cfg.get("student_df", 4.0)

    # Equinox models are called directly: model(x), not model.apply(params, None, x)
    predict_fn = lambda m, x: m(x)
    loss_full, loss_minibatch = make_loss_fns(
        predict_fn,
        X_cast,
        Y_cast,
        loss_type=loss_type,
        noise_scale=noise_scale,
        student_df=student_df,
    )

    # Flatten params for VI (required by TargetBundle interface)
    params0_flat, unravel_fn = jax.flatten_util.ravel_pytree(params_cast)

    # Build TargetBundle
    d = count_params(params_cast)
    target_bundle = TargetBundle(
        d=d,
        params0=params_cast,
        loss_full=loss_full,
        loss_minibatch=loss_minibatch,
        X=X_cast,
        Y=Y_cast,
        L0=L0,
        model=model,
        params0_flat=params0_flat,
        unravel_fn=unravel_fn,
    )

    # Create sampler config from method_cfg
    # Map method_cfg hyperparameters to sampler config structure
    sampler_cfg = _build_sampler_cfg(method_name, method_cfg, seed)

    # Run method with time budget enforcement
    n_data = X_cast.shape[0]
    log.info("[Method Trial] Running %s (d=%d, n=%d)...", method_name, d, n_data)
    key = jax.random.PRNGKey(seed)

    t0 = time.time()
    try:
        result = run_sampler(method_name, sampler_cfg, target_bundle, key)
    except Exception as e:
        log.error("[Method Trial] FAILED: %s", e)
        # Return failure metrics for Optuna
        runtime_sec = time.time() - t0
        fail_metrics = {
            "llc_hat": float("inf"),  # Large value to penalize failure
            "se_hat": None,
            "runtime_sec": runtime_sec,
            "error": float("inf"),
            "diagnostics": {"failed": True, "error_msg": str(e)},
        }
        _write_metrics(out_metrics_json, fail_metrics)
        return fail_metrics
    finally:
        runtime_sec = time.time() - t0

    log.info("[Method Trial] %s completed in %.1fs", method_name, runtime_sec)

    # Analyze traces to extract LLC
    traces = result["traces"]
    timings = result["timings"]
    work = result.get("work")
    beta = float(result["beta"])

    # Extract sampler flavour from work dict if available
    sampler_flavour = None
    if work is not None:
        sampler_flavour = work.get("sampler_flavour")

    metrics_analyzed, idata = analyze_traces(
        traces,
        L0=L0,
        n_data=n_data,
        beta=beta,
        warmup=0,  # Samplers handle warmup internally
        timings=timings,
        work=work,
        sampler_flavour=sampler_flavour or method_name,
    )

    # Extract LLC estimate and SE
    llc_hat = float(metrics_analyzed["llc_mean"])
    se_hat = metrics_analyzed.get("llc_se")

    # Compute error vs reference
    error = abs(llc_hat - ref_llc)

    # Diagnostics
    diagnostics = {
        "ess": metrics_analyzed.get("ess"),
        "rhat": metrics_analyzed.get("rhat"),
        "ess_per_sec": metrics_analyzed.get("ess_per_sec"),
        "n_samples": traces["llc"].shape[1] if "llc" in traces else None,
        "n_chains": traces["llc"].shape[0] if "llc" in traces else None,
        "work": work,
    }

    # Build metrics dict
    trial_metrics = {
        "llc_hat": llc_hat,
        "se_hat": float(se_hat) if se_hat is not None else None,
        "runtime_sec": runtime_sec,
        "error": error,
        "diagnostics": diagnostics,
        "L0": L0,
        "beta": beta,
        "gamma": float(result["gamma"]),
        "method": method_name,
        "hyperparams": method_cfg,
    }

    # Write to output JSON (idempotent)
    _write_metrics(out_metrics_json, trial_metrics)

    log.info(
        "[Method Trial] LLC_hat = %.4f ± %.4f", llc_hat, se_hat if se_hat else 0
    )
    log.info(
        "[Method Trial] Error vs ref = %.4f (%.1f%%)", error, error / ref_llc * 100
    )
    log.info("[Method Trial] Wrote metrics to %s", out_metrics_json)

    return trial_metrics


def _build_sampler_cfg(method_name, method_cfg, seed):
    """Build OmegaConf sampler config from method hyperparameters.

    Args:
        method_name: Sampler name ("sgld", "vi", "mclmc")
        method_cfg: Dict with method-specific hyperparameters
        seed: Random seed

    Returns:
        OmegaConf DictConfig for run_sampler
    """
    # Base config structure
    cfg = {
        "sampler": {"name": method_name},
        "posterior": {
            "gamma": 0.001,  # Standard localizer
            "beta_mode": "1_over_log_n",
            "loss": "mse",
        },
        "jax": {"enable_x64": method_name in ("mclmc",)},
        "runtime": {"seed": seed},
    }

    # Method-specific hyperparameters
    if method_name == "sgld":
        cfg["sampler"]["sgld"] = {
            "num_chains": method_cfg.get("num_chains", 4),
            "warmup": method_cfg.get("warmup", 1000),
            "steps": method_cfg.get("steps", 10000),
            "eta0": method_cfg.get("eta0", 0.01),
            "gamma": method_cfg.get("gamma", 0.5),
            "batch_size": method_cfg.get("batch", 128),
            "precond_type": method_cfg.get("precond_type", "rmsprop"),
            "precond_update_prob": method_cfg.get("precond_update_prob", 0.01),
        }
    elif method_name == "vi":
        cfg["sampler"]["vi"] = {
            "num_chains": method_cfg.get("num_chains", 4),
            "algo": method_cfg.get("algo", "mfa"),  # mfa or flow
            "lr": method_cfg.get("lr", 0.001),
            "steps": method_cfg.get("steps", 5000),
            "batch_size": method_cfg.get("batch_size", 256),
            "eval_every": method_cfg.get("eval_every", 50),
            "eval_samples": method_cfg.get("eval_samples", 64),
            # MFA-specific
            "M": method_cfg.get("M", 8),
            "r": method_cfg.get("r", 2),
            "whitening_mode": method_cfg.get("whitening_mode", "adam"),
            "clip_global_norm": method_cfg.get("clip_global_norm", 5.0),
        }
    elif method_name == "mclmc":
        cfg["sampler"]["mclmc"] = {
            "num_chains": method_cfg.get("num_chains", 4),
            "warmup": method_cfg.get("warmup", 1000),
            "steps": method_cfg.get("steps", 10000),
            "L": method_cfg.get("L", 1.0),
            "step_size": method_cfg.get("step_size", 0.01),
            "target_acceptance": method_cfg.get("target_accept", 0.65),
        }
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return OmegaConf.create(cfg)


def _write_metrics(out_path, metrics):
    """Write metrics to JSON file (idempotent).

    Args:
        out_path: Path to output JSON
        metrics: Dict of metrics to write
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
