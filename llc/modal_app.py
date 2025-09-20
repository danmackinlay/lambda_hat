# modal_app.py
import os
import io
import tarfile
import traceback
import time
import modal


def _gpu_list_from_env():
    """Get GPU types from environment variable with fallback to default."""
    val = os.environ.get("LLC_MODAL_GPU_LIST", "").strip()
    if not val:
        return "L40S"  # default stays simple
    # Allow comma-separated list; Modal accepts str or List[str]
    lst = [x.strip() for x in val.split(",") if x.strip()]
    return lst if len(lst) > 1 else (lst[0] if lst else "L40S")

# --- Base image: all build steps before adding local code ---
base = (
    modal.Image.debian_slim(python_version="3.11")
    # Install all deps first
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["modal"])
    # Set container env BEFORE adding local sources
    .env(
        {
            "JAX_ENABLE_X64": "true",
            "MPLBACKEND": "Agg",
            # Optional: set LLC_CODE_VERSION here if you want to override the file hash
            # "LLC_CODE_VERSION": "deploy-123",
        }
    )
)

# CPU image: add local code last
image = base.add_local_python_source("llc")

# GPU image: do the GPU install first, then add local code last
gpu_image = base.pip_install(
    "jax[cuda12_local]", force_build=True
).add_local_python_source("llc")

# Create persistent volume for runs
runs_volume = modal.Volume.from_name("llc-runs", create_if_missing=True)

app = modal.App("llc-experiments", image=image)

# Fast preflight check to detect funding/billing issues immediately
@app.function(timeout=30)
def ping():
    """Quick health check to detect Modal billing/quota issues early."""
    return {"ok": True}

# Tunable timeouts and retry settings via environment variables
TIMEOUT_S = int(os.environ.get("LLC_MODAL_TIMEOUT_S", 3 * 60 * 60))  # Default 3 hours
RETRIES = modal.Retries(
    max_retries=int(os.environ.get("LLC_MODAL_MAX_RETRIES", 3)),
    backoff_coefficient=float(os.environ.get("LLC_MODAL_BACKOFF", 2.0)),
    initial_delay=int(os.environ.get("LLC_MODAL_INITIAL_DELAY_S", 10)),
)


def _remote_impl(cfg_dict: dict, *, gpu_label: str) -> dict:
    """Shared body for CPU and GPU functions."""
    started = time.time()
    stage = "start"

    try:
        # Intentional local imports:
        # - Keep JAX out of module import time (faster cold start)
        # - Avoid importing SDKs globally in the image hydrate phase
        import jax
        jax.config.update("jax_enable_x64", True)
        from llc.tasks import run_experiment_task
        import shutil

        # Make caching effective across calls: write runs directly to the volume.
        stage = "setup"
        cfg_dict = dict(cfg_dict)  # copy
        if os.path.isdir("/runs"):
            print(f"[Modal {gpu_label}] Volume /runs exists, setting runs_dir=/runs")
            cfg_dict.setdefault("save_artifacts", True)
            cfg_dict["runs_dir"] = "/runs"
        else:
            print(f"[Modal {gpu_label}] Warning: Volume /runs not found, using default runs_dir")

        stage = "run_experiment_task"
        result = run_experiment_task(cfg_dict)
        result["status"] = "ok"
        result["duration_s"] = time.time() - started

        # Package artifacts + persist to volume
        run_dir = result.get("run_dir")
        if run_dir and os.path.isdir(run_dir):
            # Normalize to get run ID
            rid = os.path.basename(run_dir.rstrip("/"))

            # (1) Create a tar.gz first while the directory definitely exists
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                tf.add(run_dir, arcname=rid)
            result["artifact_tar"] = buf.getvalue()
            result["run_id"] = rid

            # (2) Persist to volume under /runs/<rid> for remote browsing
            try:
                vol_dir = f"/runs/{rid}"
                if os.path.exists(vol_dir):
                    shutil.rmtree(vol_dir)
                shutil.copytree(run_dir, vol_dir)
                runs_volume.commit()  # persist changes (Modal volumes)
                result["run_dir"] = vol_dir
            except Exception as e:
                # Non-fatal; we still have the tar
                print(f"[Modal {gpu_label}] Warning: failed to persist to volume: {e}")

        return result

    except Exception as e:
        # Provide deterministic run_id for later pull-runs even on failure
        from llc.cache import run_id as rid_from_cfg
        from llc.config import Config

        # Convert dict to Config for run_id calculation
        cfg = Config(**cfg_dict)
        rid = rid_from_cfg(cfg)

        return {
            "status": "error",
            "run_id": rid,
            "stage": stage,
            "error_type": e.__class__.__name__,
            "error": str(e)[:2000],
            "traceback": "".join(traceback.format_exc())[-4000:],
            "duration_s": time.time() - started,
        }


# CPU entrypoint: decorated and public
@app.function(
    gpu=None,
    timeout=TIMEOUT_S,
    volumes={"/runs": runs_volume},
    retries=RETRIES,
)
def run_experiment_remote(cfg_dict: dict) -> dict:
    """CPU remote entrypoint with shared implementation."""
    return _remote_impl(cfg_dict, gpu_label="CPU")


# GPU entrypoint: decorated and public
@app.function(
    image=gpu_image,
    gpu=_gpu_list_from_env(),
    timeout=TIMEOUT_S,
    volumes={"/runs": runs_volume},
    retries=RETRIES,
)
def run_experiment_remote_gpu(cfg_dict: dict) -> dict:
    """GPU remote entrypoint with shared implementation."""
    return _remote_impl(cfg_dict, gpu_label="GPU")


# --- SDK helpers for artifact management ---

@app.function(volumes={"/runs": runs_volume})
def list_runs(prefix: str = "/runs") -> list[str]:
    """List run directories on the Modal volume (server-side)."""
    import os

    try:
        entries = sorted(
            f"/runs/{name}"
            for name in os.listdir(prefix)
            if os.path.isdir(os.path.join(prefix, name))
        )
        return entries
    except FileNotFoundError:
        return []


@app.function(volumes={"/runs": runs_volume})
def export_run(run_id: str) -> bytes:
    """Tar.gz a single run dir under /runs and return it as bytes."""
    import io
    import os
    import tarfile

    run_path = f"/runs/{run_id}" if not run_id.startswith("/runs/") else run_id
    if not os.path.isdir(run_path):
        raise FileNotFoundError(f"Not found: {run_path}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(run_path, arcname=os.path.basename(run_path))
    return buf.getvalue()