"""HMC reference runner for Optuna hyperparameter optimization.

Computes high-quality LLC reference estimates using HMC for comparison
against approximate methods (SGLD, VI, MCLMC).
"""

import json
import time
from pathlib import Path

import jax
from omegaconf import OmegaConf

from lambda_hat.analysis import analyze_traces
from lambda_hat.config import validate_teacher_cfg
from lambda_hat.losses import as_dtype, make_loss_fns
from lambda_hat.nn_eqx import build_mlp, count_params
from lambda_hat.sampling_runner import run_sampler
from lambda_hat.targets import TargetBundle
from lambda_hat.workflow_utils import compose_build_cfg, target_id_for


def run_hmc_reference(problem_spec, out_ref_json, budget_sec=36000, seed=42):
    """Compute HMC reference LLC estimate for a problem.

    This runner creates a target from problem_spec, runs HMC with generous
    settings to obtain a high-quality reference LLC estimate, and writes
    the result to out_ref_json.

    Args:
        problem_spec: Dict with keys {model, data, teacher, seed, overrides?}
                      Same format as targets in config/experiments.yaml
        out_ref_json: Path to write reference metrics JSON
        budget_sec: Wall-time budget in seconds (default: 36000 = 10 hours)
        seed: Random seed for HMC (default: 42)

    Returns:
        dict: Reference metrics with keys:
            - llc_ref: float, reference LLC estimate
            - se_ref: float or None, standard error if available
            - diagnostics: dict with ESS, R-hat, etc.
            - runtime_sec: float, actual runtime
            - n_samples: int, number of HMC samples
    """
    print(f"[HMC Reference] Starting for problem: {problem_spec}")
    print(f"  Budget: {budget_sec}s ({budget_sec / 3600:.1f}h)")
    print(f"  Output: {out_ref_json}")

    # Build target using existing workflow infrastructure
    # Enable x64 for HMC (high precision)
    # Note: JAX_ENABLE_X64 is set by executor's worker_init, not at runtime
    build_cfg = compose_build_cfg(problem_spec, jax_enable_x64=True)

    # Compute target ID for caching (optional, for logging/debugging)
    tid = target_id_for(build_cfg)
    print(f"  Target ID: {tid}")

    # Build target: model + dataset
    # Reuse logic from commands/build_cmd.py
    from lambda_hat.targets import build_target

    # Build target (returns TargetBundle and widths)
    key_build = jax.random.PRNGKey(int(problem_spec.get("seed", 42)))
    target_bundle_init, used_model_widths, used_teacher_widths = build_target(key_build, build_cfg)

    # Extract initial data from target bundle
    X = target_bundle_init.X
    Y = target_bundle_init.Y
    params = target_bundle_init.params0
    L0_init = target_bundle_init.L0

    # Create metadata dict (simplified from build_target entrypoint)
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
        "metrics": {"L0": float(L0_init)},
        "jax_enable_x64": True,
    }

    # Get L0 from metadata
    L0 = meta.get("metrics", {}).get("L0")
    if L0 is None or L0 == 0:
        raise ValueError("L0 reference loss not found or zero in target metadata")
    L0 = float(L0)

    # Recreate model
    mcfg = meta["model_cfg"]
    widths = mcfg.get("widths")
    assert widths is not None, "Target missing resolved model widths"

    if meta.get("teacher_cfg"):
        validate_teacher_cfg(meta["teacher_cfg"])

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

    # Cast to f64 for HMC (precision)
    X_f64 = as_dtype(X, "float64")
    Y_f64 = as_dtype(Y, "float64")
    params_f64 = as_dtype(params, "float64")

    # Create loss functions
    data_cfg = meta.get("data_cfg", {})
    loss_type = "mse"  # HMC reference uses MSE for stability
    noise_scale = data_cfg.get("noise_scale", 0.1)
    student_df = data_cfg.get("student_df", 4.0)

    # Equinox models are called directly: model(x), not model.apply(params, None, x)
    predict_fn = lambda m, x: m(x)
    loss_full_f64, loss_minibatch_f64 = make_loss_fns(
        predict_fn,
        X_f64,
        Y_f64,
        loss_type=loss_type,
        noise_scale=noise_scale,
        student_df=student_df,
    )

    # Flatten params for VI (required by TargetBundle interface)
    params0_flat, unravel_fn = jax.flatten_util.ravel_pytree(params_f64)

    # Build TargetBundle
    d = count_params(params_f64)
    target_bundle = TargetBundle(
        d=d,
        params0=params_f64,
        loss_full=loss_full_f64,
        loss_minibatch=loss_minibatch_f64,
        X=X_f64,
        Y=Y_f64,
        L0=L0,
        model=model,
        params0_flat=params0_flat,
        unravel_fn=unravel_fn,
    )

    # Create HMC config with generous settings for reference quality
    # Use longer chains and more samples than typical runs
    n_data = X_f64.shape[0]
    hmc_cfg = OmegaConf.create(
        {
            "sampler": {
                "name": "hmc",
                "hmc": {
                    "num_chains": 4,  # Multiple chains for convergence diagnostics
                    "warmup": 2000,  # Generous warmup
                    "steps": 10000,  # Long chains for low SE
                    "target_acceptance": 0.8,  # High acceptance for quality
                    "num_integration_steps": 20,  # Reasonable trajectory length
                },
            },
            "posterior": {
                "gamma": 0.001,  # Standard localizer
                "beta_mode": "1_over_log_n",
                "beta0": 1.0,  # Unused when beta_mode="1_over_log_n", but must be set
                "prior_radius": None,  # No prior radius constraint
                "loss": "mse",
            },
            "jax": {"enable_x64": True},
            "runtime": {"seed": seed},
        }
    )

    # Run HMC with time budget enforcement
    print(f"[HMC Reference] Running HMC (d={d}, n={n_data})...")
    key = jax.random.PRNGKey(seed)

    t0 = time.time()
    try:
        result = run_sampler("hmc", hmc_cfg, target_bundle, key)
    except Exception as e:
        print(f"[HMC Reference] FAILED: {e}")
        raise
    finally:
        runtime_sec = time.time() - t0

    print(f"[HMC Reference] HMC completed in {runtime_sec:.1f}s")

    # Analyze traces to extract LLC
    traces = result["traces"]
    timings = result["timings"]
    work = result.get("work")
    beta = float(result["beta"])

    metrics, idata = analyze_traces(
        traces,
        L0=L0,
        n_data=n_data,
        beta=beta,
        warmup=0,  # HMC already did warmup, traces are post-warmup
        timings=timings,
        work=work,
        sampler_flavour="hmc",
    )

    # Extract reference LLC and SE
    llc_ref = float(metrics["llc_mean"])
    se_ref = metrics.get("llc_se")  # May be None if not computed

    # Diagnostics
    diagnostics = {
        "ess": metrics.get("ess"),
        "rhat": metrics.get("rhat"),
        "ess_per_sec": metrics.get("ess_per_sec"),
        "n_samples": traces["llc"].shape[1] if "llc" in traces else None,
        "n_chains": traces["llc"].shape[0] if "llc" in traces else None,
    }

    # Build reference dict
    ref = {
        "llc_ref": llc_ref,
        "se_ref": float(se_ref) if se_ref is not None else None,
        "diagnostics": diagnostics,
        "runtime_sec": runtime_sec,
        "L0": L0,
        "beta": beta,
        "gamma": float(result["gamma"]),
        "problem_spec": problem_spec,
    }

    # Write to output JSON (idempotent)
    out_path = Path(out_ref_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ref, indent=2))

    print(f"[HMC Reference] LLC_ref = {llc_ref:.4f} ± {se_ref:.4f if se_ref else 0:.4f}")
    print(f"[HMC Reference] Wrote reference to {out_ref_json}")

    return ref
