# llc/tasks.py
"""
Experiment task runner. All execution paths now use batched samplers exclusively.
Safe for pickling and cluster/cloud execution.
"""

from __future__ import annotations
from typing import Dict, Any


# IMPORTANT: keep imports inside the function if you want to minimize process import overhead
def run_experiment_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one experiment config and return a small, JSON-serializable result.
    Safe to pickle (top-level def) and safe to import on cluster/cloud.

    Always delegates to pipeline.run_one with save_artifacts controlling I/O.
    Returns uniform shape with cfg, run_dir, and llc_{sampler} values.

    Expects canonical nested payload: {"cfg": {...}, "meta": {...}, ...}
    """
    import os
    import sys
    import platform
    import traceback
    from dataclasses import fields
    from llc.config import Config, config_schema_hash
    from llc.pipeline import run_one
    from llc.util.json_safe import json_safe

    # Track stage for better error reporting
    stage = "init"

    try:
        # Require canonical nested payload structure - fail fast on unexpected format
        if "cfg" not in payload:
            raise ValueError(
                f"Invalid payload structure: missing 'cfg' key. "
                f"Expected nested payload {{'cfg': {{...}}, 'meta': {{...}}, ...}}, "
                f"got keys: {list(payload.keys())}"
            )
        cfg_dict = payload["cfg"]
        meta = payload.get("meta", {})

        # Ensure worker process honors GPU intent (SLURM/Submitit)
        gpu_mode = meta.get("gpu_mode", cfg_dict.get("gpu_mode", "off"))
        if gpu_mode != "off":
            os.environ.setdefault("JAX_PLATFORMS", "cuda")
        else:
            os.environ.setdefault("JAX_PLATFORMS", "cpu")

        # Print startup banner with device info
        import jax
        banner_info = {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "devices": [str(d) for d in jax.devices()],
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename,
        }
        banner_text = f"[llc worker] start {banner_info}"
        print(banner_text, file=sys.stdout, flush=True)

        stage = "config"
        cfg_dict_clean = dict(cfg_dict)
        # control flags must be at payload root level (strict enforcement)
        save_artifacts = bool(payload.get("save_artifacts", False))
        skip_if_exists = bool(payload.get("skip_if_exists", True))
        # Remove any control flags that leaked into config
        cfg_dict_clean.pop("save_artifacts", None)
        cfg_dict_clean.pop("skip_if_exists", None)
        provided_schema = cfg_dict_clean.pop("config_schema", None)

        # Drop unknown keys to tolerate remote/client skew (but be loud).
        allowed = {f.name for f in fields(Config)}
        dropped = sorted(set(cfg_dict_clean) - allowed)
        cfg_kwargs = {k: v for k, v in cfg_dict_clean.items() if k in allowed}

        # Schema handshake: if provided, must match.
        local_schema = config_schema_hash()
        if provided_schema and provided_schema != local_schema:
            raise RuntimeError(
                f"[stage={stage}] Config schema mismatch between client and worker.\n"
                f"  client schema: {provided_schema}\n"
                f"  worker schema: {local_schema}\n"
                "Redeploy the Modal app or use object-based remote function to auto-deploy."
            )
        if dropped:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"[llc] dropping unknown config keys: {dropped}")

        cfg = Config(**cfg_kwargs)

        # Compute run_id early for error reporting
        from llc.cache import run_id
        computed_run_id = run_id(cfg)

        # Print and persist config snapshot for debugging/reruns
        stage = "config_snapshot"
        import json
        cfg_json = json.dumps(cfg.__dict__, indent=2, default=str)
        print(f"[llc worker] cfg rid={computed_run_id}\n{cfg_json}", file=sys.stdout, flush=True)

        # Also persist alongside submitit logs if path is known
        log_dir = os.environ.get("SUBMITIT_LOGS_FOLDER") or "slurm_logs"
        try:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"{computed_run_id}_cfg.json"), "w") as f:
                f.write(cfg_json)
        except Exception:
            pass

        # Write worker info file if saving artifacts
        if save_artifacts:
            stage = "write_worker_info"
            from pathlib import Path
            run_dir = f"runs/{computed_run_id}"
            try:
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                (Path(run_dir) / "worker_info.txt").write_text(banner_text)
            except Exception:
                pass  # Non-critical, don't fail

        # Single path: pipeline is source of truth. save_artifacts governs I/O.
        stage = "pipeline"
        print(f"STAGE: pipeline.run_one", file=sys.stdout, flush=True)
        out = run_one(cfg, save_artifacts=save_artifacts, skip_if_exists=skip_if_exists, stage_callback=lambda s: print(f"STAGE: {s}", file=sys.stdout, flush=True))

        stage = "prepare_result"
        # Uniform, JSON-serializable result shape.
        result: Dict[str, Any] = {
            "cfg": cfg_dict,
            "run_dir": out.run_dir or "",
            "run_id": computed_run_id,  # Always include for better tracking
            "status": "ok",
        }
        for s in ("sgld", "sghmc", "hmc", "mclmc"):
            k = f"{s}_llc_mean"
            if k in out.metrics:
                result[f"llc_{s}"] = float(out.metrics[k])

        # Ensure JSON-safe result
        result_safe = json_safe(result)
        result_safe.setdefault("meta", json_safe(meta))

        # Contract: ok must include run_dir
        if not result_safe.get("run_dir"):
            return {
                "status": "error",
                "error_type": "ProtocolError",
                "error": "run_experiment_task produced no run_dir on success",
                "meta": result_safe.get("meta", {}),
            }

        print(f"[llc worker] complete", file=sys.stdout, flush=True)
        return result_safe

    except Exception as e:
        # Enhanced error reporting with stage and run_id
        tb = traceback.format_exc()
        error_dict = {
            "status": "error",
            "stage": stage,
            "run_id": locals().get("computed_run_id", "unknown"),
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": tb,
            "meta": json_safe(payload.get("meta", {})),
        }
        print(f"[llc worker] error at stage={stage}: {e}", file=sys.stderr, flush=True)
        return error_dict
