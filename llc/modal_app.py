# modal_app.py
import modal

# --- image: install from pyproject.toml + modal extra ---
image = (
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
    # LAST: mount local source so code edits don't rebuild the image
    .add_local_python_source("llc")
)

# For GPU support, users can create a custom image:
# gpu_image = image.pip_install("jax[cuda12_local]", force_build=True)
# Then use: @app.function(image=gpu_image, gpu="L40S")

# Create persistent volume for artifacts
artifacts_volume = modal.Volume.from_name("llc-artifacts", create_if_missing=True)

app = modal.App("llc-experiments", image=image)


@app.function(
    gpu=None,
    timeout=3
    * 60
    * 60,  # generous 3 hours, jobs may terminate early & we're OK with it
    volumes={"/artifacts": artifacts_volume},
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=10.0,
    ),
)
def run_experiment_remote(cfg_dict: dict) -> dict:
    """
    Remote entrypoint: identical signature to local task but with artifact support.
    """
    # Intentional local imports:
    # - Keep JAX out of module import time (faster cold start)
    # - Avoid importing SDKs globally in the image hydrate phase
    # Ensure x64 at runtime too (belt & suspenders; do this before heavy JAX use)
    import jax

    jax.config.update("jax_enable_x64", True)
    import os
    import shutil
    from llc.tasks import run_experiment_task

    # Make caching effective across calls: write artifacts directly to the volume.
    cfg_dict = dict(cfg_dict)
    if os.path.isdir("/artifacts"):
        print("[Modal] Volume /artifacts exists, setting artifacts_dir=/artifacts")
        cfg_dict.setdefault("save_artifacts", True)
        cfg_dict["artifacts_dir"] = "/artifacts"
    else:
        print(
            "[Modal] Warning: Volume /artifacts not found, using default artifacts_dir"
        )

    result = run_experiment_task(cfg_dict)

    # If a run_dir exists, package it and return the bytes too
    if result.get("run_dir"):
        import io
        import tarfile

        run_dir = result["run_dir"]
        # Normalize to get run ID
        rid = os.path.basename(run_dir.rstrip("/"))

        if os.path.isdir(run_dir):
            # (1) Create a tar.gz first while the directory definitely exists
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                tf.add(run_dir, arcname=rid)
            result["artifact_tar"] = buf.getvalue()
            result["run_id"] = rid

            # (2) Persist to volume under /artifacts/<rid> for remote browsing
            try:
                vol_dir = f"/artifacts/{rid}"
                if os.path.exists(vol_dir):
                    shutil.rmtree(vol_dir)
                shutil.copytree(run_dir, vol_dir)
                artifacts_volume.commit()
                result["run_dir"] = vol_dir
            except Exception as e:
                # Non-fatal; we still have the tar
                print(f"[Modal] Warning: failed to persist to volume: {e}")
                pass

    return result


# --- SDK helpers for artifact management ---


@app.function(volumes={"/artifacts": artifacts_volume})
def list_artifacts(prefix: str = "/artifacts") -> list[str]:
    """List artifact directories on the Modal volume (server-side)."""
    import os

    try:
        entries = sorted(
            f"/artifacts/{name}"
            for name in os.listdir(prefix)
            if os.path.isdir(os.path.join(prefix, name))
        )
        return entries
    except FileNotFoundError:
        return []


@app.function(volumes={"/artifacts": artifacts_volume})
def export_artifacts(run_id: str) -> bytes:
    """Tar.gz a single run dir under /artifacts and return it as bytes."""
    import io
    import os
    import tarfile

    run_path = (
        f"/artifacts/{run_id}" if not run_id.startswith("/artifacts/") else run_id
    )
    if not os.path.isdir(run_path):
        raise FileNotFoundError(f"Not found: {run_path}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(run_path, arcname=os.path.basename(run_path))
    return buf.getvalue()
