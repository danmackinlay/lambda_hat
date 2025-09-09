# modal_app.py
import modal

# --- image: keep it simple. Add your libs; GPU optional ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # core
        "numpy", "scipy", "tqdm", "pandas", "matplotlib",
        "arviz~=0.22.0", "blackjax~=1.2.5", "optax>=0.2.5",
        # jax: pick one; CPU (safe) or CUDA variant if you want GPUs
        "jax[cpu]>=0.4.26",
        # if you need GPU: replace the above with the matching CUDA wheel,
        # e.g. "jax[cuda12_local]" and ensure the correct cuda libs are present.
    )
    .add_local_python_source(".")   # ships your repo code
)

# Create persistent volume for artifacts
artifacts_volume = modal.Volume.from_name("llc-artifacts", create_if_missing=True)

app = modal.App("llc-experiments", image=image)

@app.function(
    gpu=None, 
    timeout=60*60,   # set gpu="L40S" if you want GPUs
    volumes={"/artifacts": artifacts_volume},
    retries=modal.Retries(
        max_retries=2,
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