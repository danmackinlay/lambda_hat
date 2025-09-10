# modal_app.py
import modal

# --- image: install from pyproject.toml + modal extra ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install all dependencies from pyproject.toml with modal extra
    # (includes JAX via blackjax dependency - CPU by default)
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["modal"])
    .add_local_python_source(".")  # ships your repo code
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
    from llc.tasks import run_experiment_task

    # Run task and get result with run_dir
    result = run_experiment_task(cfg_dict)

    # Sync artifacts to volume if run_dir was created
    if "run_dir" in result and result["run_dir"]:
        import shutil
        import os

        run_dir = result["run_dir"]
        if os.path.exists(run_dir):
            # Copy to volume mount
            volume_run_dir = f"/artifacts/{os.path.basename(run_dir)}"
            if os.path.exists(volume_run_dir):
                shutil.rmtree(volume_run_dir)
            shutil.copytree(run_dir, volume_run_dir)

            # Update result to point to volume location
            result["run_dir"] = volume_run_dir

        # Commit volume changes
        artifacts_volume.commit()

    return result
