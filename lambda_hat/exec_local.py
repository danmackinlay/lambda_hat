import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

def _worker_entry(cfg, gpu_id):
    """Worker entry point that sets env vars BEFORE importing JAX."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"

    # Import AFTER env vars so JAX picks the right platform
    from .run import run_one
    return run_one(cfg, save_artifacts=True, skip_if_exists=True)

def map_local(cfgs: List, *, gpus: List[int] = None) -> List[dict]:
    """One cfg per task. If gpus provided, round-robin set CUDA_VISIBLE_DEVICES."""
    outs = [None] * len(cfgs)

    max_workers = len(gpus) if gpus else None
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker_entry, cfg, (gpus[i % len(gpus)] if gpus else None)): i
                for i, cfg in enumerate(cfgs)}
        for f in as_completed(futs):
            outs[futs[f]] = f.result()

    return outs