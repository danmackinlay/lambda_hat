# modal_app.py
import modal

# --- image: install from pyproject.toml + modal extra ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install all deps first
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["modal"])
    # Set container env BEFORE adding local sources
    .env({"JAX_ENABLE_X64": "true", "MPLBACKEND": "Agg"})
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
    # Ensure x64 at runtime too (belt & suspenders; do this before heavy JAX use)
    import jax
    import os
    import shutil

    jax.config.update("jax_enable_x64", True)

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

    from llc.tasks import run_experiment_task

    result = run_experiment_task(cfg_dict)

    # Persist artifacts: if already in /artifacts (the volume), don't copy.
    if "run_dir" in result and result["run_dir"]:
        run_dir = result["run_dir"]
        print(f"[Modal] Handling artifacts: run_dir={run_dir}")

        try:
            if run_dir.startswith("/artifacts/"):
                # Already on the volume → just commit metadata
                print("[Modal] Artifacts already on volume, just committing")
                artifacts_volume.commit()
            else:
                # Local tmp → copy into the volume once
                print(f"[Modal] Copying from local tmp {run_dir} to volume")
                if os.path.exists(run_dir):
                    volume_run_dir = f"/artifacts/{os.path.basename(run_dir)}"
                    print(f"[Modal] Target volume path: {volume_run_dir}")

                    # Ensure paths are not identical
                    if os.path.abspath(run_dir) == os.path.abspath(volume_run_dir):
                        print(
                            "[Modal] Source and destination are identical, skipping copy"
                        )
                        artifacts_volume.commit()
                    else:
                        if os.path.exists(volume_run_dir):
                            print(f"[Modal] Removing existing {volume_run_dir}")
                            shutil.rmtree(volume_run_dir)
                        print(f"[Modal] Copying {run_dir} -> {volume_run_dir}")
                        shutil.copytree(run_dir, volume_run_dir)
                        result["run_dir"] = volume_run_dir
                        artifacts_volume.commit()
                else:
                    print(
                        f"[Modal] Warning: run_dir {run_dir} does not exist, skipping copy"
                    )
        except Exception as e:
            print(f"[Modal] Error handling artifacts: {e}")
            # Still try to commit in case there are partial changes
            try:
                artifacts_volume.commit()
            except Exception as e2:
                print(f"[Modal] Error committing volume: {e2}")
    else:
        print("[Modal] No run_dir in result, skipping artifact handling")

    return result
