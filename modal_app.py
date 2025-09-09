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

app = modal.App("llc-experiments", image=image)

@app.function(gpu=None, timeout=60*60)   # set gpu="L40S" if you want GPUs
def run_experiment_remote(cfg_dict: dict) -> dict:
    """
    Remote entrypoint: identical signature to local task.
    """
    from llc.tasks import run_experiment_task
    return run_experiment_task(cfg_dict)